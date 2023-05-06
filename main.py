#!/usr/bin/python3

import argparse
import multiprocessing
import os
from datetime import datetime

from psutil import Process, virtual_memory

from src.docker import get_running_containers, container_to_string
from src.logging import msg2, msg1, err2, warn3, msg3, msg4, msg5, warn4, err5, err3, \
    FMT_INFO1, FMT_INFO2, FMT_GOOD1, FMT_GOOD2, FMT_WARN1, FMT_WARN2, FMT_BAD1, FMT_BAD2, FMT_RST, err1
from src.nvidia import nvidia_smi_gpu, nvidia_smi_compute, NVIDIA_CLOCK_SPEED_THROTTLE_REASONS, gpu_to_string
from src.slurm import scontrol_show_job_pretty, jobid_to_pids, is_sjob_setup_sane, slurm_job_to_string
from src.util import is_docker_container, get_container_id_from, is_slurm_session, process_to_string, \
    round_down


def main(show_all=False, extended=False, user=None, jobid=0):
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
    sjobs = scontrol_show_job_pretty()
    gpu_info = nvidia_smi_gpu().set_index('index')
    gpu_processes = nvidia_smi_compute()
    containers = get_running_containers(gpu_info)

    # Preprocess
    sjobs['is_sane'] = None
    sjobs['ppid'] = None

    # Trim the search space
    sjobs2 = sjobs.copy()
    if jobid > 0:
        sjobs2 = sjobs2.loc[jobid].to_frame().transpose()
    elif not show_all:
        sjobs2 = sjobs2.loc[sjobs2['User'] == user]

    # Extend dataframes
    gpu_info['in_use'] = None
    gpu_info['pids'] = None
    gpu_processes['containers'] = None

    # Keep track of already identified/processed information
    gpu_info2 = gpu_info.copy()
    gpu_processes2 = gpu_processes.copy()
    containers2 = containers.copy()

    # Check all SLURM jobs
    for job_id, sjob in sjobs2.iterrows():
        sjob_pids = jobid_to_pids(job_id)
        sjob_pids_list = sjob_pids['PID'].values.tolist()
        is_user = show_all and sjob["User"] == user
        fmt_info = FMT_INFO2 if is_user else FMT_INFO1
        fmt_good = FMT_GOOD2 if is_user else FMT_GOOD1
        fmt_warn = FMT_WARN2 if is_user else FMT_WARN1
        fmt_bad = FMT_BAD2 if is_user else FMT_BAD1

        # Display basic information about SLURM job
        msg1(slurm_job_to_string(sjob, job_id, fmt_info))
        is_queueing = sjob['JobState'] == 'PENDING'
        is_cancelled = sjob['JobState'] == 'CANCELLED'

        if not is_cancelled:
            # Check if SLURM job has been set up correctly
            is_sane, slurm_ppid = is_sjob_setup_sane(Process(sjob['Sid']))
            if is_sane:
                msg2(f'{fmt_good}SLURM job has been set up correctly{FMT_RST}')
            else:
                err2(
                    f'{fmt_bad}SLURM job was not set up inside a screen/tmux session, but inside "{slurm_ppid.name()}"!')
            sjobs2.loc[job_id, 'is_sane'] = is_sane
            sjobs2.loc[job_id, 'ppid'] = slurm_ppid.pid
        else:
            sjobs2.loc[job_id, 'is_sane'] = True
            sjobs2.loc[job_id, 'ppid'] = 0

        # Show resources allocated
        gres = sjob['GRES']
        if is_queueing:
            gres = list(range(sjob['TresPerNode']))
        res = {}
        for tres in sjob['TRES'].split(','):
            k, v = tres.split('=', maxsplit=1)
            res[k] = v
        msg2(
            f'Reserved {fmt_info}{res.get("cpu", len(sjob["CPU_IDs"]))}{FMT_RST} CPUs'  # ({sjob["NumTasks"]}×{sjob["CPUs/Task"]})'
            f', min {fmt_info}{res.get("mem", sjob["MinMemoryNode"])}{FMT_RST} RAM'
            f', and {fmt_info}{len(gres)}{FMT_RST} GPU' + ('' if len(gres) == 1 else 's') +
            (f' (GPU: {", ".join(map(str, gres))})' if len(gres) != 0 and not is_queueing else ''))

        # Show GPU allocations
        if len(gres) == 0 and not is_queueing:
            warn3(f'{fmt_warn}No GPUs were allocated. Was this intended?')
        if not is_queueing:
            for gpu_id in gres:
                gpu_info_ = gpu_info2.loc[gpu_id]
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
                    gpu_info.loc[gpu_id, 'in_use'] = False
                    gpu_info2.loc[gpu_id, 'in_use'] = False
                    # Find containers that see the exact list of this slurm job's resources
                    for cont_id, cont in containers.iterrows():
                        if cont['GPUs'] == gres:
                            msg5(f'Possibly linked to {container_to_string(cont, cont_id, fmt_info)}')
                else:
                    gpu_info.loc[gpu_id, 'in_use'] = True
                    gpu_info2.loc[gpu_id, 'in_use'] = True
                gpu_info.at[gpu_id, 'pids'] = gpu_procs_actual.index.to_list()
                gpu_info2.at[gpu_id, 'pids'] = gpu_procs_actual.index.to_list()
                # Check all associated GPU processes
                for gpu_proc_pid, gpu_proc in gpu_procs_actual.iterrows():
                    gpu_processes2 = gpu_processes2.drop(gpu_proc_pid)
                    proc = Process(gpu_proc['pid'])
                    start_time = datetime.utcfromtimestamp(proc.create_time())
                    msg4(
                        f'"{proc.name()}" ({fmt_info}{gpu_proc["used_gpu_memory [MiB]"]}MiB{FMT_RST}), started {datetime.utcnow() - start_time} ago')
                    pproc = proc.parent()
                    while pproc is not None:  # Check up the process tree
                        if is_slurm_session(pproc, sjob_pids_list):
                            msg5('Running inside SLURM job')
                            break
                        elif is_docker_container(pproc):
                            # Running inside docker
                            container_id = get_container_id_from(pproc)
                            container_info = containers.loc[container_id]
                            msg5(
                                f'Running inside docker: {container_to_string(container_info, container_id, fmt_info)}')
                            # Check sanity
                            gpu_excess = [gpuid for gpuid in container_info['GPUs'] if gpuid not in gres]
                            if len(gpu_excess) > 0:
                                err5(f'Container sees more GPUs ({gpu_excess}) than allowed by SLURM job!')
                            # TODO: Add more sanity checks for containers
                            containers2 = containers2.drop(container_id, errors='ignore')
                            gpu_processes.loc[gpu_proc_pid, 'container'] = container_id
                            break
                        pproc = pproc.parent()
                    if proc is None:
                        err5('Process is neither within a docker not inside a SLURM job!')

                # Remove from the list of GPUs
                gpu_info2 = gpu_info2.drop(gpu_id)
    if extended:
        print()
        msg1('Starting analysis of remaining containers')
        for cont_id, container in containers2.iterrows():
            msg2(container_to_string(container, str(cont_id)))
            gpu_excess = container['GPUs']
            if len(gpu_excess) > 0:
                err3(f'Container sees more GPUs ({gpu_excess}) than allowed!')

        msg1('Overview of GPUs not in use at the moment (according to SLURM)')
        for gpu_id, gpu_info_ in gpu_info2.iterrows():
            msg2(gpu_to_string(gpu_info_, gpu_id, FMT_INFO1))
            for _, gpu_proc_ in gpu_processes2[gpu_processes2['gpu_uuid'] == gpu_info_['uuid']].iterrows():
                err3(f'Process {gpu_proc_.pid} is using GPU!')
                gpu_proc = Process(gpu_proc_.pid)
                msg4(process_to_string(gpu_proc))
                pproc = gpu_proc.parent()
                while pproc is not None:  # Check up the process tree
                    if is_docker_container(pproc):
                        # Running inside docker
                        container_id = get_container_id_from(pproc)
                        container_info = containers.loc[container_id]
                        msg5(f'Running inside docker: {container_to_string(container_info, container_id, FMT_INFO1)}')
                        break
                    elif pproc.name() in ['tmux: server', 'screen', 'bash']:
                        msg5(process_to_string(pproc))
                    pproc = pproc.parent()
                if gpu_proc is None:
                    err5('Process is neither within a docker not inside a SLURM job!')

    msg1('General information')
    # GPUs
    n_gpus = len(gpu_info)
    free_gpu_ids = list(range(n_gpus))
    #   SLURM
    reserved_slurm_gpu_ids = []
    list(map(reserved_slurm_gpu_ids.extend, sjobs['GRES'].tolist()))
    available_gpus = len(free_gpu_ids) - len(reserved_slurm_gpu_ids)
    free_gpu_str_slurm = f'{FMT_INFO1}{available_gpus}{FMT_RST}'
    #   NVIDIA
    reserved_nvidia_gpu_n = len(gpu_processes['gpu_uuid'].unique())
    free_gpu_str_nvidia = f'{FMT_INFO1}{len(free_gpu_ids) - reserved_nvidia_gpu_n}{FMT_RST}'
    #   Docker containers
    reserved_docker_gpu_ids = []
    list(map(reserved_docker_gpu_ids.extend, containers['GPUs'].tolist()))
    reserved_docker_gpu_ids = list(set(reserved_docker_gpu_ids))
    free_gpu_str_docker = f'{FMT_INFO1}{len(free_gpu_ids) - len(reserved_docker_gpu_ids)}{FMT_RST}'
    # CPUs
    n_cpus = multiprocessing.cpu_count()
    available_cpus = n_cpus - sum(sjobs['MinCPUsNode'].to_list())
    free_cpu_str = f'{FMT_INFO1}{available_cpus}{FMT_RST}'
    # RAM
    ram_total = int(virtual_memory().total / 1024 / 1024 / 1024)
    available_ram = ram_total - sum(sjobs['MinMemoryNode'].to_list())
    free_ram_str = f"{FMT_INFO1}{int(available_ram)}G{FMT_RST}"

    msg2(f"Free resources: {free_gpu_str_slurm}/{n_gpus} GPUs"
         f" (nvidia: {free_gpu_str_nvidia}/{n_gpus}, docker: {free_gpu_str_docker}/{n_gpus})"
         f", {free_cpu_str}/{n_cpus} CPUs"
         f", {free_ram_str}/{ram_total}G RAM")

    if available_gpus > 0:
        msg1('Suggested srun command für single-GPU job:')
        sugg_cpu = int(available_cpus / available_gpus)
        sugg_cpu = min(round_down(sugg_cpu), n_cpus // n_gpus)
        sugg_mem = int(available_ram / available_gpus)
        sugg_mem = min(round_down(sugg_mem, 0b111), ram_total // n_gpus)
        msg2(
            f'srun --pty --ntasks=1 --cpus-per-task={sugg_cpu} --mem={sugg_mem}G --gres=gpu:1 --job-name={FMT_INFO1}<jobname>{FMT_RST} bash')

    exit()


if __name__ == '__main__':
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
    args = ap.parse_args()
    main(**args.__dict__)
