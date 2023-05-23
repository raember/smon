import grp
import math
import pwd
import re
import subprocess
from datetime import timedelta, datetime
from typing import Tuple, List

from pandas import DataFrame, Series
from psutil import Process
from pyslurm import job, node, partition, powercap, statistics
from smon.log import FMT_RST, FMT_INFO1
from smon.util import round_down, strtdelta

NODE = list(node().get().values())[0]
CPU_TOTAL = NODE['cpus']
CPU_USED = NODE['alloc_cpus']
CPU_FREE = CPU_TOTAL - CPU_USED
MEM_TOTAL = int(NODE['real_memory'] / 1024)
MEM_FREE = NODE['free_mem']
MEM_USED = MEM_TOTAL - MEM_FREE
# Why is the sum of reserved memory different from NODE['free_mem'] / 1024?
# SJOBS[SJOBS['job_state'] == 'RUNNING']['pn_min_memory'].sum() / 1024  # or 'min_memory_node'
# Because oversubscription of memory is allowed!
GPU_TOTAL = int(NODE['gres'][0].split(':')[-1])
GPU_USED = int(NODE['gres_used'][0].split(':')[-1])
GPU_FREE = GPU_TOTAL - GPU_USED
CPU_SPARE_PER_GPU = 1
MAX_CPU_PER_GPU = int(CPU_TOTAL / GPU_TOTAL - CPU_SPARE_PER_GPU)
MEM_SPARE_PER_GPU = 1
MAX_MEM_PER_GPU = MEM_TOTAL / GPU_TOTAL - MEM_SPARE_PER_GPU


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
    # Now grab the GRES data
    df['CPU_IDs'] = None
    df['Mem'] = None
    df['GRES'] = None
    sproc = subprocess.Popen(['scontrol', 'show', 'job', '-do'], stdout=subprocess.PIPE)
    for sjob_line in sproc.stdout.readlines():
        line = sjob_line.decode().strip('\n')
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


SJOBS = get_jobs_pretty()


def get_partition() -> dict:
    return partition().get()


def get_powercap() -> dict:
    return powercap().get()


def get_statistics() -> dict:
    return statistics().get()


def jobid_to_pids(jobid: int) -> DataFrame:
    cmd = subprocess.run(('scontrol', 'listpids', str(jobid)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    return f'SLURM job' \
           f'{state if is_not_running else ""}' \
           f' {fmt_info}#{job_id}{FMT_RST}:' \
           f' "{fmt_info}{sjob["name"]}{FMT_RST}"' \
           f' by {fmt_info}{sjob["user"]}{FMT_RST}' \
           f' ({f"submitted {strtdelta(td_since_submit)}" if is_not_running else f"started {strtdelta(td_since_started)}"} ago):' \
           f' {fmt_info}{sjob["command"]}{FMT_RST}'


def suggest_n_gpu_srun_cmd(n_gpu: int = 1, job_name: str = None, command: str = 'bash',
                           vram: int = MAX_MEM_PER_GPU) -> str:
    gpu_factor = max(1, n_gpu)
    # CPU
    cpu_sugg_per_gpu = int(CPU_FREE / gpu_factor) - CPU_SPARE_PER_GPU
    cpu_sugg = round_down(min(cpu_sugg_per_gpu, MAX_CPU_PER_GPU) * gpu_factor, 0b111)
    # RAM
    mem_sugg_per_gpu = MEM_FREE / 1024 / gpu_factor
    mem_sugg = round_down(int(min(mem_sugg_per_gpu, MAX_MEM_PER_GPU, vram * 1.5) * gpu_factor), 0b1111)
    return res_to_srun_cmd(cpu_sugg, mem_sugg, n_gpu, job_name, command)


def res_to_str(fmt_cpu: str, n_cpu: int, fmt_mem: str, mem: float, fmt_gpu: str, n_gpu: int,
               total: bool = False) -> str:
    if int(mem) == mem:
        mem = int(mem)
    s_mem = f'{mem:.1f}' if isinstance(mem, float) else str(mem)
    return f'{fmt_cpu}{n_cpu}{FMT_RST}{f"/{CPU_TOTAL}" if total else ""} CPU{"s" if n_cpu != 1 else ""}, ' \
           f'{fmt_mem}{s_mem}{"" if total else "G"}{FMT_RST}{f"/{MEM_TOTAL}G" if total else ""} RAM, ' \
           f'{fmt_gpu}{n_gpu}{FMT_RST}{f"/{GPU_TOTAL}" if total else ""} GPU{"s" if n_gpu != 1 else ""}'


def res_to_srun_cmd(n_cpu: int, mem: int, n_gpu: int, job_name: str = None, command: str = 'bash') -> str:
    if int(mem) == mem:
        mem = int(mem)
    s_mem = f'{mem:.1f}' if isinstance(mem, float) else str(mem)
    if job_name is None:
        job_name = f'{FMT_INFO1}<jobname>{FMT_RST}'
    return f'srun --pty --ntasks=1 --cpus-per-task={n_cpu} --mem={s_mem}G --gres=gpu:{n_gpu} --job-name={job_name} {command}'
