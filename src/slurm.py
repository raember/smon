import math
import re
import subprocess
from datetime import timedelta, datetime
from pathlib import Path
from typing import Tuple, List

from pandas import DataFrame, Series
from psutil import Process

from src.logging import FMT_RST, FMT_INFO1

# https://slurm.schedmd.com/scontrol.html#SECTION_JOBS---SPECIFICATIONS-FOR-UPDATE-COMMAND
# https://slurm.schedmd.com/scontrol.html#SECTION_JOBS---SPECIFICATIONS-FOR-SHOW-COMMAND
SCONTROL_HEADERS = [
    'JobId', 'JobName', 'UserId', 'GroupId', 'MCS_label', 'Priority', 'Nice', 'Account', 'QOS', 'JobState', 'Reason',
    'Dependency', 'Requeue', 'Restarts', 'BatchFlag', 'Reboot', 'ExitCode', 'DerivedExitCode', 'RunTime', 'TimeLimit',
    'TimeMin', 'SubmitTime', 'EligibleTime', 'AccrueTime', 'StartTime', 'EndTime', 'Deadline', 'SuspendTime',
    'SecsPreSuspend', 'LastSchedEval', 'Partition', 'AllocNode:Sid', 'ReqNodeList', 'ExcNodeList', 'NodeList',
    'BatchHost', 'NumNodes', 'NumCPUs', 'NumTasks', 'CPUs/Task', 'ReqB:S:C:T', 'TRES', 'Socks/Node', 'NtasksPerN:B:S:C',
    'CoreSpec', 'Nodes', 'CPU_IDs', 'Mem', 'GRES', 'MinCPUsNode', 'MinMemoryNode', 'MinTmpDiskNode', 'Features',
    'DelayBoot', 'OverSubscribe', 'Contiguous', 'Licenses', 'Network', 'Command', 'WorkDir', 'Power', 'TresPerNode'
]


def scontrol_show_job() -> DataFrame:
    """
    Queries information about current slurm jobs from scontrol.

    May contain jobs that are COMPLETED.
    May lack entries for GRES and TresPerNode if no GPU was reserved.
    Is susceptible to injection attacks like putting valid key-value pairs into the job name.
    :return: A table of the accumulated slurm job information.
    :rtype: DataFrame
    """
    sproc = subprocess.Popen(['scontrol', 'show', 'job', '-do'], stdout=subprocess.PIPE)
    sjobs = []
    altered_header = rf"{SCONTROL_HEADERS[-1]}|END"
    current_headers = SCONTROL_HEADERS[:-1]
    current_headers.append(altered_header)
    next_headers = SCONTROL_HEADERS[1:-1]
    next_headers.append(altered_header)
    next_headers.append(altered_header)
    END = 'END='
    for sjob_line in sproc.stdout.readlines():
        data = {}
        rest_s = sjob_line.decode().strip('\n') + f' {END}'
        for header, next_header in zip(current_headers, next_headers):
            if rest_s == END:
                break
            regex = rf'^({header})=(|\S.*)(\s+)({next_header})='
            m = re.search(regex, rest_s)
            assert m is not None
            key = m.group(1)
            val = m.group(2)
            data[key] = val
            rest_s = rest_s[m.regs[3][1]:]
        sjobs.append(data)
    df = DataFrame(sjobs)
    return df


def time_s_to_timedelta(time_s: str) -> timedelta:
    if time_s == 'UNLIMITED':
        return timedelta(days=-1)
    days = 0
    if '-' in time_s:
        time_parts = time_s.split('-')
        days = int(time_parts[0])
        time_s = time_parts[1]
    time_parts = time_s.split(':')
    return timedelta(days=days, hours=int(time_parts[0]), minutes=int(time_parts[1]), seconds=int(time_parts[2]))


def scontrol_show_job_pretty() -> DataFrame:
    """
    Queries scontrol and formats the received data into more useful datatypes.

    Note that some conversions may be unstable.
    :return: The processed list of slurm job information.
    :rtype: DataFrame
    """
    df = scontrol_show_job()
    df.JobId = df.JobId.astype(int)
    df['User'] = df.UserId.map(lambda s: s.split('(')[0])
    df.UserId = df.UserId.map(lambda s: int(s.strip(')').split('(')[1]))
    df['Group'] = df.GroupId.map(lambda s: s.split('(')[0])
    df.GroupId = df.GroupId.map(lambda s: int(s.strip(')').split('(')[1]))
    df.Priority = df.Priority.astype(int)
    df.Nice = df.Nice.astype(int)
    df.Account = df.Account.map(lambda s: None if s == '(null)' else s)
    df.QOS = df.QOS.map(lambda s: None if s == '(null)' else s)
    df.Reason = df.Reason.map(lambda s: None if s == 'None' else s)
    df.Dependency = df.Dependency.map(lambda s: None if s == '(null)' else s)
    df.Requeue = df.Requeue.astype(bool)
    df.Restarts = df.Restarts.astype(bool)
    df.BatchFlag = df.BatchFlag.astype(bool)
    df.Reboot = df.Reboot.astype(bool)
    df.RunTime = df.RunTime.map(time_s_to_timedelta)
    df.TimeLimit = df.TimeLimit.map(time_s_to_timedelta)
    df.SubmitTime = df.SubmitTime.map(datetime.fromisoformat)
    df.EligibleTime = df.EligibleTime.map(datetime.fromisoformat)
    df.StartTime = df.StartTime.map(datetime.fromisoformat)
    df.SuspendTime = df.SuspendTime.map(lambda s: None if s == 'None' else s)
    df.SecsPreSuspend = df.SecsPreSuspend.astype(int)
    df.LastSchedEval = df.LastSchedEval.map(datetime.fromisoformat)
    df['Sid'] = df['AllocNode:Sid'].map(lambda s: int(s.split(':')[1]))
    df['AllocNode:Sid'] = df['AllocNode:Sid'].map(lambda s: s.split(':')[0])
    df.ReqNodeList = df.ReqNodeList.map(lambda s: None if s == '(null)' else s)
    df.ExcNodeList = df.ExcNodeList.map(lambda s: None if s == '(null)' else s)
    df.NumNodes = df.NumNodes.astype(int)
    df.NumCPUs = df.NumCPUs.astype(int)
    df.NumTasks = df.NumTasks.astype(int)
    df['CPUs/Task'] = df['CPUs/Task'].astype(int)

    def id_to_list(s: str) -> List[int]:
        l = []
        if s == '':
            return l
        for ran in s.split(','):
            if '-' in ran:
                a, b = ran.split('-')
                l += list(range(int(a), int(b) + 1))
            else:
                l.append(int(ran))
        return l

    df.CPU_IDs = df.CPU_IDs.map(id_to_list)
    df.GRES = df.GRES.map(lambda s: id_to_list(s.strip(')').split(':')[-1]))
    df.MinCPUsNode = df.MinCPUsNode.astype(int)
    df.MinMemoryNode = df.MinMemoryNode.map(lambda s: int(s[:-1]))
    df.MinTmpDiskNode = df.MinTmpDiskNode.astype(int)
    df.Features = df.Features.map(lambda s: None if s == '(null)' else s)
    df.DelayBoot = df.DelayBoot.map(time_s_to_timedelta)
    df.Contiguous = df.Contiguous.astype(int)
    df.Licenses = df.Licenses.map(lambda s: None if s == '(null)' else s)
    df.Network = df.Network.map(lambda s: None if s == '(null)' else s)
    df.WorkDir = df.WorkDir.map(lambda s: Path(s))
    df.TresPerNode = df.TresPerNode.map(
        lambda s: 0 if s is None or isinstance(s, float) and math.isnan(s) else int(s[4:]))
    return df


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


def slurm_job_to_string(sjob: Series, fmt_info: str = FMT_INFO1) -> str:
    return f'SLURM job' \
           f' {fmt_info}#{sjob["JobId"]}{FMT_RST}:' \
           f' "{fmt_info}{sjob["JobName"]}{FMT_RST}"' \
           f' by {fmt_info}{sjob["User"]}{FMT_RST}' \
           f' (started {sjob["RunTime"]} ago): {fmt_info}{sjob["Command"]}{FMT_RST}'
