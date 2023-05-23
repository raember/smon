#!/usr/bin/python3

import argparse
import os
from datetime import datetime

from psutil import Process
from smon.docker_info import get_running_containers, container_to_string
from smon.log import msg2, msg1, err2, warn3, msg3, msg4, msg5, warn4, err5, err3, \
    FMT_INFO1, FMT_INFO2, FMT_GOOD1, FMT_GOOD2, FMT_WARN1, FMT_WARN2, FMT_BAD1, FMT_BAD2, FMT_RST, err1, warn1
from smon.nvidia import nvidia_smi_gpu, nvidia_smi_compute, NVIDIA_CLOCK_SPEED_THROTTLE_REASONS, gpu_to_string
from smon.slurm import jobid_to_pids, is_sjob_setup_sane, slurm_job_to_string, get_jobs, \
    get_partition, suggest_n_gpu_srun_cmd, res_to_str, SJOBS, NODE, get_statistics, GPU_TOTAL, GPU_USED, GPU_FREE, \
    MEM_FREE, CPU_FREE, res_to_srun_cmd
from smon.util import is_docker_container, get_container_id_from, is_slurm_session, process_to_string, \
    strtdelta, strmbytes


def get_args() -> dict:
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
    return ap.parse_args().__dict__


def main(show_all=False, extended=False, user=None, jobid=0):
    # Are we inside a SLURM session?
    SJOB_NAME = os.getenv('SLURM_JOB_NAME')
    SJOB_ID = os.getenv('SLURM_JOB_ID')
    SJOB_USER = os.getenv('SLURM_JOB_USER')
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
        user = os.environ.get('USER')
    addressee = []
    if jobid > 0:
        addressee.append(f'SLURM job {FMT_INFO1}#{jobid}{FMT_RST}')
    else:
        addressee.append(f'{FMT_INFO1}{"all" if show_all else user}{FMT_RST}')
    # If we want the extended information, we need to run the sjob loop,
    # but honoring the all vs user options or just run it quietly
    run_all_sjobs_but_quietly = extended and not show_all

    msg1(f'Crunching data for {", ".join(addressee)}')
    # SLURM
    sjobs = SJOBS
    node = NODE
    partition = list(get_partition().values())[0]
    statistics = get_statistics()
    # NVIDIA
    gpu_info = nvidia_smi_gpu().set_index('index')
    gpu_processes = nvidia_smi_compute()
    # Docker
    containers = get_running_containers(gpu_info)

    # Preprocess
    sjobs['is_sane'] = None
    sjobs['ppid'] = None

    # Trim the search space
    sjobs2 = sjobs.copy()
    if jobid > 0:
        sjobs2 = sjobs2.loc[jobid].to_frame().transpose()
    elif not show_all:
        sjobs2 = sjobs2.loc[sjobs2['user'] == user]

    pending_sjobs = sjobs2[sjobs2['job_state'] == 'PENDING']
    running_sjobs = sjobs2[sjobs2['job_state'] == 'RUNNING']

    # Extend dataframes
    gpu_info['in_use'] = None
    gpu_info['pids'] = None
    gpu_processes['containers'] = None

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

        # Check if SLURM job has been set up correctly
        is_sane, slurm_ppid = is_sjob_setup_sane(Process(sjob['alloc_sid']))
        if not is_sane:
            err2(
                f'{fmt_bad}SLURM job was not set up inside a screen/tmux session, but inside "{slurm_ppid.name()}"!')
        sjobs2.loc[job_id, 'is_sane'] = is_sane
        sjobs2.loc[job_id, 'ppid'] = slurm_ppid.pid

        # Show resources allocated
        sjob_gres = sjob['GRES']
        res = sjob['tres_req']
        res_cpu = int(res.get("cpu", len(sjob["CPU_IDs"])))
        res_mem = int(res.get("mem", sjob["pn_min_memory"]).rstrip('G'))
        res_gpu = len(sjob_gres)
        msg2(
            f'Reserved {res_to_str(fmt_info, res_cpu, fmt_info, res_mem, fmt_info, res_gpu)}' +
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
            msg3(gpu_to_string(gpu_info_, gpu_id, fmt_info))

            # Determine throttling
            throttling_reasons = gpu_info_[NVIDIA_CLOCK_SPEED_THROTTLE_REASONS]
            for throttling_reason in throttling_reasons.index:
                if throttling_reasons[throttling_reason] == 'Active':
                    reason = throttling_reason.split('.')[-1].replace('_', ' ').title()
                    reason = reason.replace('Sw', 'SW').replace('Hw', 'HW').replace('Gpu', 'GPU')
                    if throttling_reason != 'clocks_throttle_reasons.gpu_idle':
                        warn3(f'{fmt_warn}Throttling: {reason}')

            # Check GPU usage and setup (i.e. container or venv/conda)
            gpu_uuid = gpu_info_['uuid']
            gpu_procs_actual = gpu_processes[gpu_processes['gpu_uuid'] == gpu_uuid]
            if len(gpu_procs_actual) == 0:
                warn4(f'GPU not in use')
                gpu_info.loc[gpu_id_internal, 'in_use'] = False
                gpu_info2.loc[gpu_id_internal, 'in_use'] = False
                # Find containers that see the exact list of this slurm job's resources
                for cont_id, cont in containers.iterrows():
                    if cont['GPUs'] == sjob_gres:
                        msg5(f'Possibly linked to {container_to_string(cont, cont_id, fmt_info)}')
            else:
                gpu_info.loc[gpu_id_internal, 'in_use'] = True
                gpu_info2.loc[gpu_id_internal, 'in_use'] = True
            gpu_info.at[gpu_id_internal, 'pids'] = gpu_procs_actual.index.to_list()
            gpu_info2.at[gpu_id_internal, 'pids'] = gpu_procs_actual.index.to_list()
            # Check all associated GPU processes
            for gpu_proc_pid, gpu_proc in gpu_procs_actual.iterrows():
                gpu_processes2 = gpu_processes2.drop(gpu_proc_pid)
                proc = Process(gpu_proc['pid'])
                start_time = datetime.utcfromtimestamp(proc.create_time())
                proc_cpu_util = proc.cpu_percent(0.2) / 100 * proc.cpu_num()
                msg4(
                    f'"{proc.name()}" ({fmt_info}{strmbytes(gpu_proc["used_gpu_memory [MiB]"])}{FMT_RST}, {fmt_info}{proc_cpu_util:.1f}%{FMT_RST} CPU), started {strtdelta(datetime.utcnow() - start_time)} ago')
                get_jobs()
                sjob_proc = proc.parent()
                while sjob_proc is not None:  # Check up the process tree
                    if is_slurm_session(sjob_proc, sjob_pids_list):
                        msg5('Running inside SLURM job')
                        break
                    elif is_docker_container(sjob_proc):
                        # Running inside docker
                        container_id = get_container_id_from(sjob_proc)
                        container_info = containers.loc[container_id]
                        msg5(
                            f'Running inside docker: {container_to_string(container_info, container_id, fmt_info)}')
                        # Check sanity
                        gpu_excess = [gpuid for gpuid in container_info['GPUs'] if gpuid not in sjob_gres]
                        if len(gpu_excess) > 0:
                            err5(f'Container sees more GPUs ({gpu_excess}) than allowed by SLURM job!')
                        # TODO: Add more sanity checks for containers
                        containers2 = containers2.drop(container_id, errors='ignore')
                        gpu_processes.loc[gpu_proc_pid, 'container'] = container_id
                        break
                    sjob_proc = sjob_proc.parent()
                if proc is None:
                    err5('Process is neither within a docker not inside a SLURM job!')

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

            sjob_gres = list(range(int(sjob.get('tres_per_node', 'gpu:0').split(':')[-1])))
            res = sjob['tres_req']
            res_cpu = int(res.get("cpu", len(sjob["CPU_IDs"])))
            fmt_cpu = fmt_info if res_cpu <= CPU_FREE else fmt_warn
            res_mem = int(res.get("mem", sjob["pn_min_memory"]).rstrip('G'))
            fmt_mem = fmt_info if res_mem <= MEM_FREE else fmt_warn
            res_gpu = len(sjob_gres)
            fmt_gpu = fmt_info if res_gpu <= GPU_FREE else fmt_warn
            msg2(
                f'Requesting {res_to_str(fmt_cpu, res_cpu, fmt_mem, res_mem, fmt_gpu, res_gpu)}')

            # Get acceptable parameters
            new_cpu = min(res_cpu, CPU_FREE)
            fmt_cpu = fmt_good if new_cpu != res_cpu else fmt_info
            new_mem = min(res_mem, MEM_FREE)
            fmt_mem = fmt_good if new_mem != res_mem else fmt_info
            new_gpu = min(res_gpu, GPU_FREE)
            fmt_gpu = fmt_good if new_gpu != res_gpu else fmt_info
            should_be_accepted = fmt_cpu == fmt_info and fmt_mem == fmt_info and fmt_gpu == fmt_info
            if not should_be_accepted:
                msg3(f'Would be processed if instead reserving'
                     f' {res_to_str(fmt_cpu, new_cpu, fmt_mem, new_mem, fmt_gpu, new_gpu)}')
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

        msg1('Overview of GPUs not in use at the moment (according to SLURM)')
        for gpu_id, gpu_info_ in gpu_info2.iterrows():
            msg2(gpu_to_string(gpu_info_, gpu_id, FMT_INFO1))
            for _, gpu_proc_ in gpu_processes2[gpu_processes2['gpu_uuid'] == gpu_info_['uuid']].iterrows():
                err3(f'Process {gpu_proc_.pid} is using GPU!')
                gpu_proc = Process(gpu_proc_.pid)
                msg4(process_to_string(gpu_proc))
                sjob_proc = gpu_proc.parent()
                while sjob_proc is not None:  # Check up the process tree
                    if is_docker_container(sjob_proc):
                        # Running inside docker
                        container_id = get_container_id_from(sjob_proc)
                        container_info = containers.loc[container_id]
                        msg5(f'Running inside docker: {container_to_string(container_info, container_id, FMT_INFO1)}')
                        break
                    elif sjob_proc.name() in ['tmux: server', 'screen', 'bash']:
                        msg5(process_to_string(sjob_proc))
                    sjob_proc = sjob_proc.parent()
                if gpu_proc is None:
                    err5('Process is neither within a docker not inside a SLURM job!')

    msg1('General information')
    # GPUs
    #   SLURM
    free_gpu_str_slurm = f'{FMT_INFO1}{GPU_FREE}{FMT_RST}'
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
    available_ram = int(ram_total - sum(sjobs['pn_min_memory'].to_list())) / 1024
    free_ram_str = f"{FMT_INFO1}{strmbytes(available_ram, False)}{FMT_RST}"

    msg2(f"Free resources: {free_gpu_str_slurm}/{GPU_TOTAL} GPUs"
         f" ({FMT_INFO1}{reserved_nvidia_gpu_n}{FMT_RST}/{GPU_USED} using the GPU)"
         # f" (nvidia: {free_gpu_str_nvidia}/{n_gpus}, docker: {free_gpu_str_docker}/{n_gpus})"
         f", {free_cpu_str}/{n_cpus} CPUs"
         f", {free_ram_str}/{strmbytes(node['real_memory'], False)} RAM")

    if GPU_FREE > 0:
        msg1('Suggested srun command für single-GPU job:')
        msg2(f"$ {suggest_n_gpu_srun_cmd(vram=int(gpu_info.loc[0, 'memory.total [MiB]'] / 1000))}")

    exit()


if __name__ == '__main__':
    main(**get_args())
