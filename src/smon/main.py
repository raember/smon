#!/usr/bin/python3
# smon - slurm info script for GPU-HPC users
# Copyright © 2023  Raphael Emberger

import argparse
import os
import pickle
import re
import tempfile
from datetime import datetime
from pathlib import Path
from socket import gethostname
from traceback import format_exc
from typing import Tuple

from pandas import DataFrame, Series
from psutil import Process, NoSuchProcess
from smon.docker_info import get_running_containers, container_to_string
from smon.log import msg2, msg1, warn3, msg3, msg4, msg5, err5, err3, \
    FMT_INFO1, FMT_INFO2, FMT_GOOD1, FMT_GOOD2, FMT_WARN1, FMT_WARN2, FMT_BAD1, FMT_BAD2, FMT_RST, err1, warn1, warn2, \
    MAGENTA, LIGHT_GRAY, log_tree
from smon.nvidia import nvidia_smi_gpu, nvidia_smi_compute, NVIDIA_CLOCK_SPEED_THROTTLE_REASONS, gpu_to_string
from smon.slurm import jobid_to_pids, is_sjob_setup_sane, slurm_job_to_string, get_node, \
    get_partition, suggest_n_gpu_srun_cmd, res_to_str, get_jobs_pretty, get_statistics, res_to_srun_cmd
from smon.util import is_docker_container, get_container_id_from, is_slurm_session, process_to_string, \
    strtdelta, strmbytes, strgbytes

sjobs: DataFrame = None
node: Series = None
partition: Series = None
stats: Series = None
stats_type: DataFrame = None
stats_user: DataFrame = None
gpu_info: DataFrame = None
gpu_processes: DataFrame = None
containers: DataFrame = None

CMD_PREFIX = f'\033[1;{MAGENTA}m❯\033[m'
PERCENTAGE_WARN1 = 80
PERCENTAGE_WARN2 = 90
STR_LEN_MAX = 90


def get_args() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '-a', '--all',
        action='store_true', default=False, dest='show_all',
        help='Show statistics about all users on the node')
    ap.add_argument(
        '-e', '--extended',
        action='store_true', default=False, dest='extended',
        help='Show extended analysis about all components on the node. Implies --all')
    ap.add_argument(
        '-u', '--user',
        action='store', dest='user',
        help='Show only SLURM jobs of given username. Cannot be set together with --all')
    ap.add_argument(
        '-j', '--jobid',
        action='store', dest='jobid', type=int, default=0,
        help='Show only SLURM job of give id. Cannot be set together with --all')
    return ap


def main(show_all=False, extended=False, user=None, jobid=0, pkl_fp: Path = None, list_dumps=False) -> Tuple[
    DataFrame, Series, Series, Series, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame]:
    if list_dumps:
        msg1('Listing error dumps on current host')
        tmp_p = Path('/tmp')
        files = []
        for fp in tmp_p.glob('smon_*.pkl'):
            _, host, user, _ = fp.name.split('_', maxsplit=3)
            files.append({
                'path': fp,
                'user': user,
                'host': host,
                'mtime': datetime.fromtimestamp(os.path.getmtime(fp)),
                'ctime': datetime.fromtimestamp(os.path.getctime(fp)),
            })
        files = DataFrame(files)
        if len(files) > 0:
            for user in files['user'].unique():
                msg2(f'{FMT_INFO1}{user}{FMT_RST}')
                user_dumps = files[files['user'] == user]
                for _, (path, _, host, mtime, ctime) in user_dumps.sort_values(by=['mtime'],
                                                                               ascending=False).iterrows():
                    msg3(f'{path} - {str(mtime)}')
        exit(0)

    data = {}
    is_dump = False
    if pkl_fp is not None:
        msg1('Loading state from dump file')
        data = pickle.load(open(pkl_fp, 'rb'))
        is_dump = True
        msg2(f"Created by {FMT_INFO2}{data['env']['USER']}{FMT_RST}"
             f" on {FMT_INFO2}{data['host']}{FMT_RST}"
             f", {FMT_INFO2}{data['timestamp']}{FMT_RST}")
        args = data['args']
        show_all = args['show_all']
        extended = args['extended']
        user = args['user']
        jobid = args['jobid']
    # Are we inside a SLURM session?
    if not is_dump:
        SJOB_NAME = os.getenv('SLURM_JOB_NAME')
        SJOB_ID = os.getenv('SLURM_JOB_ID')
        SJOB_USER = os.getenv('SLURM_JOB_USER')
    else:
        SJOB_NAME = data.get('env', {}).get('SLURM_JOB_NAME')
        SJOB_ID = data.get('env', {}).get('SLURM_JOB_ID')
        SJOB_USER = data.get('env', {}).get('SLURM_JOB_USER')
    print_prog_args = extended
    if we_are_inside_slurm := SJOB_NAME is not None and SJOB_ID is not None and SJOB_USER is not None:
        warn1(
            f"Cannot see all resources from within SLURM job {SJOB_NAME}:{SJOB_ID}! Only showing information for current SLURM job!")
        extended = False
        show_all = False
        jobid = int(SJOB_ID)
        user = SJOB_USER
    show_all |= extended
    if show_all and (jobid > 0 or user is not None):
        err1('Cannot show all SLURM jobs and still honor --user or --jobid')
        exit(1)

    if user is None:
        user = data.get('env', {}).get('USER', os.getenv('USER'))
    addressee = []
    if jobid > 0:
        addressee.append(f'SLURM job {FMT_INFO1}#{jobid}{FMT_RST}')
    else:
        addressee.append(f'{FMT_INFO1}{"all" if show_all else user}{FMT_RST}')
    # If we want the extended information, we need to run the sjob loop,
    # but honoring the all vs user options or just run it quietly
    run_all_sjobs_but_quietly = extended and not show_all
    if is_dump:
        args['show_all'] = show_all
        args['extended'] = extended
        args['user'] = user
        args['jobid'] = jobid
        msg2(f"Arguments: {args}")
        msg2(f"Error: {FMT_BAD2}{repr(data['exception'])}{FMT_RST}")
        print(f"{FMT_BAD1}{data['stacktrace']}{FMT_RST}")

    msg1(f'Crunching data for {", ".join(addressee)}')
    # SLURM
    global sjobs, node, partition, stats, stats_type, stats_user, gpu_info, gpu_processes, containers
    sjobs = data.get('sjobs') if data.get('sjobs') is not None else get_jobs_pretty()
    node = data.get('node') if data.get('node') is not None else get_node()
    cpu_total = node['cpus']
    cpu_used = node['alloc_cpus']
    cpu_free = cpu_total - cpu_used
    mem_free = node['free_mem']
    gpu_total = node['gres']['gpu']
    gpu_used = node['gres_used']['gpu']
    gpu_free = gpu_total - gpu_used
    partition = data.get('partition') if data.get('partition') is not None else get_partition()
    stats, stats_type, stats_user = (data.get('stats'), data.get('stats_type'), data.get('stats_user')) if data.get(
        'stats') is not None else get_statistics()
    # NVIDIA
    gpu_info = data.get('gpu_info') if data.get('gpu_info') is not None else nvidia_smi_gpu().set_index('index')
    gpu_processes = data.get('gpu_processes') if data.get('gpu_processes') is not None else nvidia_smi_compute()
    # Docker
    containers = data.get('containers') if data.get('containers') is not None else get_running_containers(gpu_info)

    # Preprocess
    if not is_dump:
        sjobs['is_interactive_bash_session'] = None
        sjobs['is_sane'] = None
        sjobs['ppid'] = None
        sjobs['is_using_gpu'] = False

    # Trim the search space
    sjobs2 = sjobs.copy()
    if jobid > 0:
        sjobs2 = sjobs2.loc[jobid].to_frame().transpose()
    elif not show_all:
        sjobs2 = sjobs2.loc[sjobs2['user'] == user]

    pending_sjobs = sjobs2[sjobs2['job_state'] == 'PENDING']
    running_sjobs = sjobs2[sjobs2['job_state'] == 'RUNNING']

    # Extend dataframes
    if not is_dump:
        gpu_info['in_use'] = None
        gpu_info['pids'] = None
        gpu_processes['container'] = None
        gpu_processes['conda'] = None
        gpu_processes['cpu_util'] = None
        gpu_processes['create_time'] = None
        gpu_processes['cpu_cnt'] = None
        gpu_processes['ram'] = None

    # Keep track of already identified/processed information
    gpu_info2 = gpu_info.copy()
    gpu_processes2 = gpu_processes.copy()
    containers2 = containers.copy()

    # Check all SLURM jobs
    for job_id, sjob in running_sjobs.iterrows():
        sjob_pids = jobid_to_pids(job_id)
        sjob_pids_list = sjob_pids['PID'].values.tolist()
        is_user = show_all and sjob['user'] == user and not run_all_sjobs_but_quietly
        fmt_info = FMT_INFO2 if is_user else FMT_INFO1
        fmt_good = FMT_GOOD2 if is_user else FMT_GOOD1
        fmt_warn = FMT_WARN2 if is_user else FMT_WARN1
        fmt_bad = FMT_BAD2 if is_user else FMT_BAD1

        # Display basic information about SLURM job
        msg1(slurm_job_to_string(sjob, job_id, fmt_info))

        with log_tree(3, include_root_hook=False) as tree:
            # Check if SLURM job has been set up correctly
            is_interactive_bash_session = True
            if not is_dump:
                try:
                    sjob_main_pid = Process(sjob['alloc_sid'])
                    is_interactive_bash_session = sjob_main_pid.children()[0].cmdline()[-1] != 'bash'
                    if not is_interactive_bash_session:
                        tree.log_leaf(f'{fmt_warn}Is an interactive bash session')
                    is_sane, slurm_ppid = is_sjob_setup_sane(sjob_main_pid)
                except NoSuchProcess as ex_nsp:
                    tree.log_leaf(f"{fmt_warn}SLURM job session process (PID:{ex_nsp.pid}) does not exist!")
                    is_sane = False
                    slurm_ppid = None
            else:
                is_sane = sjobs.loc[job_id, 'is_sane']
                slurm_ppid = sjobs.loc[job_id, 'ppid']
                is_interactive_bash_session = sjobs.loc[job_id, 'is_interactive_bash_session']
            if not is_sane:
                if slurm_ppid is not None:
                    tree.log_leaf(
                        f'{fmt_bad}SLURM job was not set up inside a screen/tmux session, but inside "{slurm_ppid.name()}"!')
                else:
                    tree.log_leaf(f'{fmt_warn}SLURM session cannot be determined!')
            sjobs2.loc[job_id, 'is_interactive_bash_session'] = is_interactive_bash_session
            sjobs2.loc[job_id, 'is_sane'] = is_sane
            sjobs2.loc[job_id, 'ppid'] = slurm_ppid.pid if hasattr(slurm_ppid, 'pid') else slurm_ppid
            sjobs.loc[job_id, 'is_interactive_bash_session'] = is_interactive_bash_session
            sjobs.loc[job_id, 'is_sane'] = is_sane
            sjobs.loc[job_id, 'ppid'] = slurm_ppid.pid if hasattr(slurm_ppid, 'pid') else slurm_ppid

            # Show resources allocated
            sjob_gres = sjob['GRES']
            res = sjob['tres_req']
            res_cpu = int(res.get("cpu", len(sjob["CPU_IDs"])))
            res_mem = float(res.get("mem", sjob["pn_min_memory"]).rstrip('G'))
            res_gpu = len(sjob_gres)
            tree.log_leaf(f'Reserved {res_to_str(fmt_info, res_cpu, fmt_info, res_mem, fmt_info, res_gpu, node)}' +
                          (f' (GPU: {", ".join(map(str, sjob_gres))})' if len(sjob_gres) != 0 else ''))

            # Show GPU allocations
            # if len(sjob_gres) == 0 and not is_queueing:
            #     warn3(f'{fmt_warn}No GPUs were allocated. Was this intended?')

            # The GPU IDs/indices are shifted because of limited visibility
            for gpu_id, gpu_id_i in zip(sjob_gres, range(len(sjob_gres))):
                gpu_id_internal = gpu_id
                if we_are_inside_slurm:
                    gpu_id_internal = gpu_id_i
                gpu_info_ = gpu_info2.loc[gpu_id_internal]
                # Print stats
                with tree.add_node() as gpu_node:
                    gpu_node.log(gpu_to_string(gpu_info_, gpu_id, fmt_info))

                    # Determine throttling
                    throttling_reasons = gpu_info_[NVIDIA_CLOCK_SPEED_THROTTLE_REASONS]
                    for throttling_reason in throttling_reasons.index:
                        if throttling_reasons[throttling_reason] == 'Active':
                            reason = throttling_reason.split('.')[-1].replace('_', ' ').title()
                            reason = reason.replace('Sw', 'SW').replace('Hw', 'HW').replace('Gpu', 'GPU')
                            if throttling_reason != 'clocks_throttle_reasons.gpu_idle':
                                gpu_node.log_leaf(f'{fmt_warn}Throttling: {reason}')

                    # Check GPU usage and setup (i.e. container or venv/conda)
                    gpu_uuid = gpu_info_['uuid']
                    gpu_procs_actual = gpu_processes[gpu_processes['gpu_uuid'] == gpu_uuid]
                    if len(gpu_procs_actual) == 0:
                        gpu_node.log_leaf(f'{fmt_bad}GPU not in use')
                        gpu_info.loc[gpu_id_internal, 'in_use'] = False
                        gpu_info2.loc[gpu_id_internal, 'in_use'] = False
                        # Find containers that see the exact list of this slurm job's resources
                        with gpu_node.add_node() as container_node:
                            for cont_id, cont in containers.iterrows():
                                if cont['GPUs'] == sjob_gres:
                                    container_node.log(
                                        f'Possibly linked to {container_to_string(cont, cont_id, fmt_info)}')
                    else:
                        sjobs.loc[job_id, 'is_using_gpu'] = True
                        gpu_info.loc[gpu_id_internal, 'in_use'] = True
                        gpu_info2.loc[gpu_id_internal, 'in_use'] = True
                    gpu_info.at[gpu_id_internal, 'pids'] = gpu_procs_actual['pid'].to_list()
                    gpu_info2.at[gpu_id_internal, 'pids'] = gpu_procs_actual['pid'].to_list()
                    # Check all associated GPU processes
                    for gpu_proc_pid, gpu_proc in gpu_procs_actual.iterrows():
                        gpu_processes2 = gpu_processes2.drop(gpu_proc_pid)
                        proc_name = f'{CMD_PREFIX} \033[{LIGHT_GRAY}m{gpu_proc["process_name"]}{FMT_RST}'
                        start_time, cpu_cnt, proc_cpu_util, proc_ram, proc_ram_perc = None, None, None, None, None
                        does_pid_exist = True
                        vram = gpu_proc["used_gpu_memory [MiB]"]
                        vram_perc = vram / gpu_info_["memory.total [MiB]"]
                        fmt_vram = fmt_good
                        if vram_perc > PERCENTAGE_WARN2:
                            fmt_vram = fmt_bad
                        elif vram_perc > PERCENTAGE_WARN1:
                            fmt_vram = fmt_warn
                        proc_details = f'{fmt_vram}{strmbytes(gpu_proc["used_gpu_memory [MiB]"])}{FMT_RST} VRAM'
                        proc = None
                        if not is_dump:
                            try:
                                proc = Process(gpu_proc['pid'])
                                if not print_prog_args:
                                    proc_name = ' '.join(proc.cmdline())
                                    if len(proc_name) > STR_LEN_MAX:
                                        proc_name = proc_name[:STR_LEN_MAX - 3] + '...'
                                else:
                                    proc_name = proc.name()
                                proc_name = f'{CMD_PREFIX} \033[{LIGHT_GRAY}m{proc_name}{FMT_RST}'
                                start_time = datetime.fromtimestamp(proc.create_time())
                                gpu_processes.loc[gpu_proc_pid, 'create_time'] = start_time
                                proc_mem = proc.memory_info()
                                proc_ram = float(proc_mem.rss) / 1000 ** 3
                                # proc_ram_virt = float(proc_mem.vms) / 1000 ** 3
                                gpu_processes.loc[gpu_proc_pid, 'ram'] = proc_ram
                                proc_ram_perc = proc_ram / res_mem * 100
                                cpu_cnt = proc.cpu_num()
                                gpu_processes.loc[gpu_proc_pid, 'cpu_cnt'] = cpu_cnt
                                proc_cpu_util = proc.cpu_percent(0.2)
                                gpu_processes.loc[gpu_proc_pid, 'cpu_util'] = proc_cpu_util
                                sjob_proc = proc.parent()
                            except NoSuchProcess:
                                # warn4(f"GPU process {gpu_proc['pid']} could not be found!")
                                proc_name = f'{fmt_bad}{gpu_proc["process_name"]}{FMT_RST}'
                                sjob_proc = None
                                does_pid_exist = False
                        else:
                            start_time = gpu_processes.loc[gpu_proc_pid, 'create_time']
                            proc_ram = gpu_processes.loc[gpu_proc_pid, 'ram']
                            proc_ram_perc = proc_ram / res_mem * 100
                            cpu_cnt = gpu_processes.loc[gpu_proc_pid, 'cpu_cnt']
                            proc_cpu_util = gpu_processes.loc[gpu_proc_pid, 'cpu_util']
                            sjob_proc = Process(1)
                        if proc_ram is not None:
                            ram_s = f'{proc_ram:.1f}' if int(proc_ram) != proc_ram else int(proc_ram)
                            fmt_ram = fmt_good
                            if proc_ram_perc > PERCENTAGE_WARN2:
                                fmt_ram = fmt_bad
                            elif proc_ram_perc > PERCENTAGE_WARN1:
                                fmt_ram = fmt_warn
                            proc_details += f', {fmt_ram}{ram_s}{FMT_RST}/{strgbytes(res_mem, False)} RAM'
                            # ({proc_ram_perc:.1f}%)'
                        if cpu_cnt is not None:
                            proc_details += f', {fmt_info}{proc_cpu_util:.1f}%{FMT_RST}, {fmt_info}{cpu_cnt}{FMT_RST} CPUs'
                        proc_details += ''
                        if start_time is not None:
                            proc_details += f', started {strtdelta(datetime.now() - start_time)} ago'

                        pid_str = str(gpu_proc["pid"])
                        with gpu_node.add_node([MAGENTA]) as proc_node:
                            proc_node.log(f'[{fmt_info}{pid_str}{FMT_RST}] {proc_name}{FMT_RST}')
                            if print_prog_args and proc is not None:
                                # Print out all args of the running job
                                proc_args = proc.cmdline()[1:]
                                is_first_arg = True
                                arg = ''
                                for proc_arg in proc_args:
                                    arg_line = (arg + ' ' + proc_arg).strip()
                                    if len(arg_line) > STR_LEN_MAX and not is_first_arg:
                                        proc_node.log(
                                            f'\033[{";".join(map(str, proc_node.fmt))}m│{FMT_RST} {" " * len(pid_str)}   \033[{LIGHT_GRAY}m' + arg + FMT_RST)
                                        arg = proc_arg
                                    else:
                                        arg = arg_line.strip()
                                    is_first_arg = False
                                if arg != '':
                                    proc_node.log(
                                        f'\033[{";".join(map(str, proc_node.fmt))}m│{FMT_RST} {" " * len(pid_str)}   \033[{LIGHT_GRAY}m' + arg + FMT_RST)
                            proc_node.log_leaf(proc_details)
                        while sjob_proc is not None:  # Check up the process tree
                            if is_slurm_session(sjob_proc, sjob_pids_list):
                                if gpu_processes.loc[gpu_proc_pid, 'conda']:
                                    gpu_node.log_leaf(
                                        f'Running in a baremetal {gpu_processes.loc[gpu_proc_pid, "conda"]} environment')
                                else:
                                    gpu_node.log_leaf('Running in baremetal environment',
                                                      4)  # This shouldn't really ever happen
                                break
                            elif 'conda' in sjob_proc.cmdline()[0]:
                                # Possibly inside conda baremetal.
                                cmd = sjob_proc.cmdline()[0]
                                conda_cmd = re.search(r'/([^/]*conda[^/]*)/', cmd).group(1)
                                gpu_processes.loc[gpu_proc_pid, 'conda'] = conda_cmd
                            elif is_docker_container(sjob_proc):
                                # Running inside docker
                                if not is_dump:
                                    container_id = get_container_id_from(sjob_proc)
                                    gpu_processes.loc[gpu_proc_pid, 'container'] = container_id
                                    # We don't want conda inside docker getting flagged as baremetal-conda
                                    gpu_processes.loc[gpu_proc_pid, 'conda'] = None
                                else:
                                    container_id = gpu_processes.loc[gpu_proc_pid, 'container']
                                container_info = containers.loc[container_id]
                                proc_node.log_leaf(
                                    f'Running inside docker: {container_to_string(container_info, container_id, fmt_info)}')
                                # Check sanity
                                gpu_excess = [gpuid for gpuid in container_info['GPUs'] if gpuid not in sjob_gres]
                                if len(gpu_excess) > 0:
                                    proc_node.log_leaf(
                                        f'{fmt_bad}Container sees more GPUs ({gpu_excess}) than allowed by SLURM job!{FMT_RST}')
                                # TODO: Add more sanity checks for containers
                                containers2 = containers2.drop(container_id, errors='ignore')
                                gpu_processes.loc[gpu_proc_pid, 'container'] = container_id
                                break
                            sjob_proc = sjob_proc.parent()
                        if does_pid_exist and sjob_proc is None:
                            proc_node.log_leaf(
                                f'{fmt_bad}Process is neither within a docker nor inside a SLURM job!{FMT_RST}')

                    # Remove from the list of GPUs
                    gpu_info2 = gpu_info2.drop(gpu_id_internal)

    if len(pending_sjobs) > 0:
        msg1(f"{'=' * 10}  SJOBS IN QUEUE  {'=' * 10}")
        for job_id, sjob in pending_sjobs.iterrows():
            is_user = show_all and sjob['user'] == user and not run_all_sjobs_but_quietly
            fmt_info = FMT_INFO2 if is_user else FMT_INFO1
            fmt_good = FMT_GOOD2 if is_user else FMT_GOOD1
            fmt_warn = FMT_WARN2 if is_user else FMT_WARN1
            fmt_bad = FMT_BAD2 if is_user else FMT_BAD1

            # Display basic information about SLURM job
            msg1(slurm_job_to_string(sjob, job_id, fmt_info))
            sjob_main_pid = Process(sjob['alloc_sid'])
            is_interactive_bash_session = sjob_main_pid.children()[0].cmdline()[-1] != 'bash'
            if not is_interactive_bash_session:
                warn2('Is an interactive bash session')

            sjob_gres = list(range(int(sjob.get('tres_per_node', 'gpu:0').split(':')[-1])))
            res = sjob['tres_req']
            res_cpu = int(res.get("cpu", len(sjob["CPU_IDs"])))
            fmt_cpu = fmt_info if res_cpu <= cpu_free else fmt_warn
            res_mem = int(res.get("mem", sjob["pn_min_memory"]).rstrip('G'))
            fmt_mem = fmt_info if res_mem <= mem_free else fmt_warn
            res_gpu = len(sjob_gres)
            fmt_gpu = fmt_info if res_gpu <= gpu_free else fmt_warn
            msg2(
                f'Requesting {res_to_str(fmt_cpu, res_cpu, fmt_mem, res_mem, fmt_gpu, res_gpu, node)}')

            # Get acceptable parameters
            new_cpu = min(res_cpu, cpu_free)
            fmt_cpu = fmt_good if new_cpu != res_cpu else fmt_info
            new_mem = min(res_mem, mem_free)
            fmt_mem = fmt_good if new_mem != res_mem else fmt_info
            new_gpu = min(res_gpu, gpu_free)
            fmt_gpu = fmt_good if new_gpu != res_gpu else fmt_info
            should_be_accepted = fmt_cpu == fmt_info and fmt_mem == fmt_info and fmt_gpu == fmt_info
            if not should_be_accepted:
                msg3(f'Would be processed if instead reserving'
                     f' {res_to_str(fmt_cpu, new_cpu, fmt_mem, new_mem, fmt_gpu, new_gpu, node)}')
                msg3("$ " + res_to_srun_cmd(new_cpu, new_mem, new_gpu, job_name=sjob['name'], command=sjob['command']))
            else:
                msg3('Job is expected to be allocated the requested resources soon.')

    if extended:
        print()
        msg1('Starting analysis of remaining containers')
        for cont_id, container in containers2.iterrows():
            msg2(container_to_string(container, str(cont_id)))
            gpu_excess = container['GPUs']
            if len(gpu_excess) > 0:
                warn3(f'Container sees more GPUs ({gpu_excess}) than allowed!')

        if len(gpu_info2) > 0:
            msg1('Overview of GPUs not in use at the moment (according to SLURM)')
            for gpu_id, gpu_info_ in gpu_info2.iterrows():
                msg2(gpu_to_string(gpu_info_, gpu_id, FMT_INFO1))
                for _, gpu_proc_ in gpu_processes2[gpu_processes2['gpu_uuid'] == gpu_info_['uuid']].iterrows():
                    err3(f'Process {gpu_proc_.pid} is using GPU!')
                    if not is_dump:
                        gpu_proc = Process(gpu_proc_.pid)
                        msg4(process_to_string(gpu_proc))
                        sjob_proc = gpu_proc.parent()
                        sjob_name = sjob_proc.name()
                        gpu_processes.loc[gpu_proc_pid, 'name'] = sjob_name
                        start_time = datetime.fromtimestamp(gpu_proc.create_time())
                        gpu_processes.loc[gpu_proc_pid, 'create_time'] = start_time
                        proc_cpu_util = gpu_proc.cpu_percent(0.2) / 100 * gpu_proc.cpu_num()
                        gpu_processes.loc[gpu_proc_pid, 'cpu_util'] = proc_cpu_util
                    else:
                        msg4(gpu_proc_)
                        sjob_proc = True
                        sjob_name = gpu_processes.loc[gpu_proc_pid, 'name']
                        start_time = gpu_processes.loc[gpu_proc_pid, 'create_time']
                        proc_cpu_util = gpu_processes.loc[gpu_proc_pid, 'cpu_util']
                    while sjob_proc is not None:  # Check up the process tree
                        if gpu_processes.loc[gpu_proc_pid, 'container'] is not None or is_docker_container(sjob_proc):
                            # Running inside docker
                            if not is_dump:
                                container_id = get_container_id_from(sjob_proc)
                            else:
                                container_id = gpu_processes.loc[gpu_proc_pid, 'container']
                            container_info = containers.loc[container_id]
                            msg5(
                                f'Running inside docker: {container_to_string(container_info, container_id, FMT_INFO1)}')
                            break
                        elif sjob_name in ['tmux: server', 'screen', 'bash']:
                            msg5(process_to_string(sjob_proc))
                        elif sjob_name == '':
                            msg5("Unclear where this process used to be.")
                            break
                        sjob_proc = sjob_proc.parent()
                    if gpu_proc is None:
                        err5('Process is neither within a docker nor inside a SLURM job!')

    msg1('General information')
    # GPUs
    #   SLURM
    free_gpu_str_slurm = f'{FMT_INFO1}{gpu_free}{FMT_RST}'
    #   NVIDIA
    reserved_nvidia_gpu_n = len(gpu_processes['gpu_uuid'].unique())
    # free_gpu_str_nvidia = f'{FMT_INFO1}{len(all_gpu_ids) - reserved_nvidia_gpu_n}{FMT_RST}'
    # #   Docker containers
    # reserved_docker_gpu_ids = []
    # list(map(reserved_docker_gpu_ids.extend, containers['GPUs'].tolist()))
    # reserved_docker_gpu_ids = list(set(reserved_docker_gpu_ids))
    # free_gpu_str_docker = f'{FMT_INFO1}{len(all_gpu_ids) - len(reserved_docker_gpu_ids)}{FMT_RST}'
    # CPUs
    n_cpus = node['cpus']
    available_cpus = n_cpus - node['alloc_cpus']
    free_cpu_str = f'{FMT_INFO1}{available_cpus}{FMT_RST}'
    # RAM
    ram_total = node['real_memory']
    running_sjobs = sjobs[sjobs['job_state'] == 'RUNNING']
    available_ram = int(ram_total - sum(running_sjobs['pn_min_memory'].to_list())) / 1024
    free_ram_str = f"{FMT_INFO1}{strgbytes(available_ram, False)}{FMT_RST}"

    msg2(f"Free resources: {free_gpu_str_slurm}/{gpu_total} GPUs"
         f" ({FMT_INFO1}{reserved_nvidia_gpu_n}{FMT_RST}/{gpu_used} using the GPU)"
         # f" (nvidia: {free_gpu_str_nvidia}/{n_gpus}, docker: {free_gpu_str_docker}/{n_gpus})"
         f", {free_cpu_str}/{n_cpus} CPUs"
         f", {free_ram_str}/{strmbytes(node['real_memory'], False)} RAM")

    if gpu_free > 0 and len(gpu_info) > 0:
        msg1('Suggested srun command for single-GPU job:')
        msg2(f"$ {suggest_n_gpu_srun_cmd(vram=int(gpu_info.loc[0, 'memory.total [MiB]'] / 1000), node_info=node)}")

    return sjobs, node, partition, stats, stats_type, stats_user, gpu_info, gpu_processes, containers


if __name__ == '__main__':
    args = get_args().parse_args().__dict__
    try:
        main(**args)
    except Exception as e:
        err1(f'An error occurred: {e}')
        stack = format_exc()
        print(f"{FMT_BAD1}{stack}{FMT_RST}")
        dump = {
            'sjobs': sjobs,
            'node': node,
            'partition': partition,
            'stats': stats,
            'stats_type': stats_type,
            'stats_user': stats_user,
            'gpu_info': gpu_info,
            'gpu_processes': gpu_processes,
            'containers': containers,
            # Meta data
            'args': args,
            'env': {
                'SLURM_JOB_NAME': os.getenv('SLURM_JOB_NAME'),
                'SLURM_JOB_ID': os.getenv('SLURM_JOB_ID'),
                'SLURM_JOB_USER': os.getenv('SLURM_JOB_USER'),
                'USER': os.getenv('USER'),
            },
            'exception': e,
            'stacktrace': stack,
            'timestamp': datetime.now(),
            'host': gethostname(),
        }
        with tempfile.NamedTemporaryFile(prefix=f'smon_{gethostname()}_{os.getenv("USER")}_', suffix='.pkl',
                                         delete=False) as tfp:
            pickle.dump(dump, tfp)
            os.chmod(tfp.name, 0o664)
            msg1(
                f'Created a dump file at {FMT_INFO2}{tfp.name}{FMT_RST} - Please send this to {FMT_INFO2}embe{FMT_RST}.')
