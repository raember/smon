# smon - slurm info script for GPU-HPC users
# Copyright © 2023  Raphael Emberger

import grp
import math
import pwd
import re
import subprocess
from datetime import timedelta, datetime
from typing import Tuple, List

from pandas import DataFrame, Series
from psutil import Process
from pyslurm import job, partition, powercap, statistics, node
from smon.log import FMT_RST, FMT_INFO1
from smon.util import round_down, strtdelta

COLUMNS = [
    'account', 'accrue_time', 'admin_comment', 'alloc_node', 'alloc_sid', 'array_job_id', 'array_task_id',
    'array_task_str', 'array_max_tasks', 'assoc_id', 'batch_flag', 'batch_features', 'batch_host', 'billable_tres',
    'bitflags', 'boards_per_node', 'burst_buffer', 'burst_buffer_state', 'command', 'comment', 'contiguous',
    'core_spec', 'cores_per_socket', 'cpus_per_task', 'cpus_per_tres', 'cpu_freq_gov', 'cpu_freq_max', 'cpu_freq_min',
    'dependency', 'derived_ec', 'eligible_time', 'end_time', 'exc_nodes', 'exit_code', 'features', 'group_id', 'job_id',
    'job_state', 'last_sched_eval', 'licenses', 'max_cpus', 'max_nodes', 'mem_per_tres', 'name', 'network', 'nodes',
    'nice', 'ntasks_per_core', 'ntasks_per_core_str', 'ntasks_per_node', 'ntasks_per_socket', 'ntasks_per_socket_str',
    'ntasks_per_board', 'num_cpus', 'num_nodes', 'num_tasks', 'partition', 'mem_per_cpu', 'min_memory_cpu',
    'mem_per_node', 'min_memory_node', 'pn_min_memory', 'pn_min_cpus', 'pn_min_tmp_disk', 'power_flags', 'priority',
    'profile', 'qos', 'reboot', 'req_nodes', 'req_switch', 'requeue', 'resize_time', 'restart_cnt', 'resv_name',
    'run_time', 'run_time_str', 'sched_nodes', 'shared', 'show_flags', 'sockets_per_board', 'sockets_per_node',
    'start_time', 'state_reason', 'std_err', 'std_in', 'std_out', 'submit_time', 'suspend_time', 'system_comment',
    'time_limit', 'time_limit_str', 'time_min', 'threads_per_core', 'tres_alloc_str', 'tres_bind', 'tres_freq',
    'tres_per_job', 'tres_per_node', 'tres_per_socket', 'tres_per_task', 'tres_req_str', 'user_id', 'wait4switch',
    'wckey', 'work_dir', 'cpus_allocated', 'cpus_alloc_layout', 'CPU_IDs', 'Mem', 'GRES'
]


def _id_to_list(s: str) -> List[int]:
    l = []
    if s == '' or s == 'nan' or (isinstance(s, float) and math.isnan(s)):
        return l
    for ran in s.split(','):
        if '-' in ran:
            a, b = ran.split('-')
            l += list(range(int(a), int(b) + 1))
        else:
            l.append(int(ran))
    return l


def get_jobs() -> DataFrame:
    """
    Queries information about current slurm jobs from scontrol.

    May contain jobs that are COMPLETED.
    May lack entries for GRES and TresPerNode if no GPU was reserved.
    Is susceptible to injection attacks like putting valid key-value pairs into the job name.
    :return: A table of the accumulated slurm job information.
    :rtype: DataFrame
    """
    df = DataFrame(job().get()).transpose()
    if len(df) == 0:
        df = DataFrame(columns=COLUMNS)
        return df
    # Now grab the GRES data
    df['CPU_IDs'] = None
    df['Mem'] = None
    df['GRES'] = None
    sproc = subprocess.Popen(['/usr/bin/scontrol', 'show', 'job', '-do'], stdout=subprocess.PIPE)
    for sjob_line in sproc.stdout.readlines():
        line = sjob_line.decode().strip('\n')
        if line == 'No jobs in the system':
            break
        m_job_id = re.match(r'^JobId=(\d+)', line)
        job_id = int(m_job_id.group(1))
        m_data = re.search(r'CPU_IDs=([\d\-,]+) Mem=(\d+) GRES=(\S+)', line[-500:])
        cpu_ids = []
        mem = 0
        gres = []
        if m_data is not None:
            cpu_ids = _id_to_list(m_data.group(1))
            mem = int(m_data.group(2))
            gres = _id_to_list(m_data.group(3).strip(')').split(':')[-1])
        df.at[job_id, 'CPU_IDs'] = cpu_ids
        df.at[job_id, 'Mem'] = mem
        df.at[job_id, 'GRES'] = gres
    return df


def _query_to_dict(s: str) -> dict:
    d = {}
    if s != 'None':
        for tres in s.split(','):
            k, v = tres.split('=', maxsplit=1)
            d[k] = v
    return d


def get_jobs_pretty() -> DataFrame:
    """
    Queries scontrol and formats the received data into more useful datatypes.

    Note that some conversions may be unstable.
    :return: The processed list of slurm job information.
    :rtype: DataFrame
    """
    df = get_jobs()
    df.alloc_sid = df.alloc_sid.astype(int)
    df.assoc_id = df.assoc_id.astype(int)
    df.batch_flag = df.batch_flag.astype(int)
    df.billable_tres = df.billable_tres.astype(float)
    df.bitflags = df.bitflags.astype(int)
    df.boards_per_node = df.boards_per_node.astype(int)
    df.command = df.command.astype(str)
    df.comment = df.comment.map(lambda c: '' if c is None else c).astype(str)
    df.contiguous = df.contiguous.astype(str)
    df.cpus_per_task = df.cpus_per_task.astype(int)
    df.eligible_time = df.eligible_time.astype(int)
    df.end_time = df.end_time.astype(int)
    df.group_id = df.group_id.astype(int)
    df['group'] = df.group_id.map(lambda gid: grp.getgrgid(gid).gr_name)
    df.job_id = df.job_id.astype(int)
    df.job_state = df.job_state.astype(str)
    df.max_cpus = df.max_cpus.astype(int)
    df.max_nodes = df.max_nodes.astype(int)
    df.name = df.name.astype(str)
    df.nodes = df.nodes.astype(str)
    df.nice = df.nice.astype(int)
    df.ntasks_per_core_str = df.ntasks_per_core_str.astype(str)
    df.ntasks_per_board = df.ntasks_per_board.astype(int)
    df.num_cpus = df.num_cpus.astype(int)
    df.num_nodes = df.num_nodes.astype(int)
    df.num_tasks = df.num_tasks.astype(int)
    df.partition = df.partition.astype(str)
    df.mem_per_cpu = df.mem_per_cpu.astype(bool)
    df.mem_per_node = df.mem_per_node.astype(bool)
    df.min_memory_node = df.min_memory_node.astype(int)
    df.pn_min_memory = df.pn_min_memory.astype(int)
    df.pn_min_cpus = df.pn_min_cpus.astype(int)
    df.pn_min_tmp_disk = df.pn_min_tmp_disk.astype(int)
    df.power_flags = df.power_flags.astype(int)
    df.priority = df.priority.astype(int)
    df.profile = df.profile.astype(int)
    df.reboot = df.reboot.astype(int)
    df.req_switch = df.req_switch.astype(int)
    df.requeue = df.requeue.astype(bool)
    df.resize_time = df.resize_time.astype(int)
    df.restart_cnt = df.restart_cnt.astype(int)
    df.run_time = df.run_time.astype(int)
    df.run_time_str = df.run_time_str.astype(str)
    df.shared = df.shared.astype(str)
    df.show_flags = df.show_flags.astype(int)
    df.sockets_per_board = df.sockets_per_board.astype(int)
    df.start_time = df.start_time.astype(int)
    df.submit_time = df.submit_time.astype(int)
    df.suspend_time = df.suspend_time.astype(int)
    df.time_limit_str = df.time_limit_str.astype(str)
    df.time_min = df.time_min.astype(int)
    df.tres_alloc_str = df.tres_alloc_str.astype(str)
    df['tres_alloc'] = df.tres_alloc_str.map(_query_to_dict)
    df.tres_per_node = df.tres_per_node.astype(str)
    df['gpus'] = df.tres_per_node.map(lambda s: int(s.split(':')[-1]) if s != 'None' else 0)
    df.tres_req_str = df.tres_req_str.astype(str)
    df['tres_req'] = df.tres_req_str.map(_query_to_dict)
    df.user_id = df.user_id.astype(int)
    df['user'] = df.user_id.map(lambda uid: pwd.getpwuid(uid).pw_name)
    df.wait4switch = df.wait4switch.astype(int)
    df.work_dir = df.work_dir.astype(str)
    return df


def get_node() -> Series:
    node_info = Series(list(node().get().values())[0])

    def split_str_list_to_dict(l: List[str]) -> dict:
        d = {}
        for entry in l:
            key, val = entry.split(':', maxsplit=1)
            d[key] = int(val) if val.isdigit() else val
        return d

    node_info['gres'] = split_str_list_to_dict(node_info['gres'])
    node_info['gres_used'] = split_str_list_to_dict(node_info['gres_used'])
    return node_info


def get_partition() -> Series:
    d = list(partition().get().values())[0]
    flags = d['flags']
    del d['flags']
    for key, val in flags.items():
        d[f'flag_{key}'] = val
    return Series(d)


def get_powercap() -> Series:
    return Series(powercap().get())


def get_statistics() -> Tuple[Series, DataFrame, DataFrame]:
    stats = statistics().get()
    rpc_type_stats = stats['rpc_type_stats']
    del stats['rpc_type_stats']
    rpc_user_stats = stats['rpc_user_stats']
    del stats['rpc_user_stats']
    return Series(stats), DataFrame(rpc_type_stats).transpose(), DataFrame(rpc_user_stats).transpose()


def jobid_to_pids(jobid: int) -> DataFrame:
    cmd = subprocess.run(('/usr/bin/scontrol', 'listpids', str(jobid)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pids = []
    for line in cmd.stdout.decode().splitlines()[1:]:
        pid, job_id_2, step_id, local_id, global_id = tuple(
            map(int, re.sub(r'\s+', ',', line.strip(' ')).replace('-', '0').split(',')))
        assert job_id_2 == jobid
        pids.append({
            'PID': pid,
            'JOBID': job_id_2,
            'STEPID': step_id,
            'LOCALID': local_id,
            'GLOBALID': global_id,
        })
    df = DataFrame(pids, columns=['PID', 'JOBID', 'STEPID', 'LOCALID', 'GLOBALID'])
    return df


def is_sjob_setup_sane(sid: Process) -> Tuple[bool, Process]:
    ppid = sid.parent()
    default_ppid = ppid
    while ppid is not None:
        if ppid.name() in ['screen', 'tmux: server']:
            return True, ppid
        ppid = ppid.parent()
    return False, default_ppid


def slurm_job_to_string(sjob: Series, job_id: int, fmt_info: str = FMT_INFO1) -> str:
    is_not_running = sjob['job_state'] != 'RUNNING'
    reason = f" ({sjob['state_reason']})"
    state = f" {fmt_info}{sjob['job_state']}{FMT_RST}{reason if reason != ' (None)' else ''}"
    td_since_submit = datetime.now() - datetime.fromtimestamp(sjob["submit_time"])
    td_since_started = timedelta(seconds=int(sjob["run_time"]))
    pw = pwd.getpwuid(sjob["user_id"])
    name = ''
    if pw.pw_gecos != '':
        name = f'{fmt_info}{pw.pw_gecos}{FMT_RST} ({fmt_info}{pw.pw_name}{FMT_RST})'
    else:
        name = f'{fmt_info}{pw.pw_name}{FMT_RST}'
    return f'SLURM job' \
           f'{state if is_not_running else ""}' \
           f' {fmt_info}#{job_id}{FMT_RST}:' \
           f' "{fmt_info}{sjob["name"]}{FMT_RST}"' \
           f' by {name}' \
           f' ({f"submitted {strtdelta(td_since_submit)}" if is_not_running else f"started {strtdelta(td_since_started)}"} ago):' \
           f' {fmt_info}{sjob["command"]}{FMT_RST}'


def suggest_n_gpu_srun_cmd(n_gpu: int = 1, job_name: str = None, command: str = 'bash',
                           vram: int = -1, node_info: Series = None) -> str:
    assert node_info is not None
    cpu_total = node_info['cpus']
    cpu_used = node_info['alloc_cpus']
    cpu_free = cpu_total - cpu_used
    mem_total = int(node_info['real_memory'] / 1024)
    mem_free = node_info['free_mem']
    # Why is the sum of reserved memory different from NODE['free_mem'] / 1024?
    # SJOBS[SJOBS['job_state'] == 'RUNNING']['pn_min_memory'].sum() / 1024  # or 'min_memory_node'
    # Because oversubscription of memory is allowed!
    gpu_total = node_info['gres']['gpu']
    cpu_spare_per_gpu = 1
    max_cpu_per_gpu = int(cpu_total / gpu_total - cpu_spare_per_gpu)
    mem_spare_per_gpu = 1
    max_mem_per_gpu = mem_total / gpu_total - mem_spare_per_gpu
    if vram < 0:
        vram = max_mem_per_gpu
    gpu_factor = max(1, n_gpu)
    # CPU
    cpu_sugg_per_gpu = int(cpu_free / gpu_factor) - cpu_spare_per_gpu
    cpu_sugg = round_down(min(cpu_sugg_per_gpu, max_cpu_per_gpu) * gpu_factor, 0b111)
    # RAM
    mem_sugg_per_gpu = mem_free / 1024 / gpu_factor
    mem_sugg = round_down(int(min(mem_sugg_per_gpu, max_mem_per_gpu, vram * 1.5) * gpu_factor), 0b1111)
    return res_to_srun_cmd(cpu_sugg, mem_sugg, n_gpu, job_name, command)


def res_to_str(fmt_cpu: str, n_cpu: int, fmt_mem: str, mem: float, fmt_gpu: str, n_gpu: int, node_info: Series,
               total: bool = False) -> str:
    if int(mem) == mem:
        mem = int(mem)
    cpu_total = node_info['cpus']
    s_mem = f'{mem:.1f}' if isinstance(mem, float) else str(mem)
    mem_total = int(node_info['real_memory'] / 1024)
    gpu_total = node_info['gres']['gpu']
    return f'{fmt_cpu}{n_cpu}{FMT_RST}{f"/{cpu_total}" if total else ""} CPU{"s" if n_cpu != 1 else ""}, ' \
           f'{fmt_mem}{s_mem}{"" if total else "G"}{FMT_RST}{f"/{mem_total}G" if total else ""} RAM, ' \
           f'{fmt_gpu}{n_gpu}{FMT_RST}{f"/{gpu_total}" if total else ""} GPU{"s" if n_gpu != 1 else ""}'


def res_to_srun_cmd(n_cpu: int, mem: int, n_gpu: int, job_name: str = None, command: str = 'bash') -> str:
    if int(mem) == mem:
        mem = int(mem)
    s_mem = f'{mem:.1f}' if isinstance(mem, float) else str(mem)
    if job_name is None:
        job_name = f'{FMT_INFO1}<jobname>{FMT_RST}'
    return f'srun --pty --ntasks=1 --cpus-per-task={n_cpu} --mem={s_mem}G --gres=gpu:{n_gpu} --job-name={job_name} {command}'


def is_interactive_bash_session(proc: Process) -> bool:
    if proc.name() == 'srun':
        return proc.cmdline()[-1] in ['bash', 'sh', 'mysecureshell', 'tmux', 'screen']
    else:
        for child in proc.children():
            if is_interactive_bash_session(child):
                return True
        return False
