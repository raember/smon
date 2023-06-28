# smon - slurm info script for GPU-HPC users
# Copyright © 2023  Raphael Emberger

from datetime import datetime, timedelta
from typing import List, Callable, Dict

from psutil import Process
from smon.log import FMT_INFO1, FMT_RST


def find_parent(process: Process, branches: Dict[Callable[[Process], bool], Callable[[Process], None]]):
    pproc = process.parent()
    while (pproc := pproc.parent()) is not None:
        for condition, func in branches.items():
            if condition(pproc):
                func(pproc)
                pproc = None
                break
        if pproc is None:
            break


def is_slurm_session(process: Process, pid_list: List[int]) -> bool:
    return process.pid in pid_list


def is_docker_container(process: Process) -> bool:
    return process.name() == 'containerd-shim-runc-v2'


def get_container_id_from(process: Process) -> str:
    cmd_args = process.cmdline()
    while len(cmd_args) > 0:
        cmd_arg = cmd_args.pop(0)
        if cmd_arg == '-id':
            break
    assert len(cmd_args) > 0
    return cmd_args.pop(0)


def process_to_string(process: Process, fmt_info: str = FMT_INFO1) -> str:
    user = ''
    if process.username() != 'root':
        user = f' by {fmt_info}{process.username()}{FMT_RST}'
    start_time = datetime.fromtimestamp(process.create_time())
    return f'"{process.name()}" (started {datetime.now() - start_time} ago)' + user


def round_down(n: int, threshold: int = 0b11) -> int:
    """
    Round down as to not exceed n.
    :param n: The number to round down.
    :param threshold: Threshold to stop right-shifting
    :return: The rounded number
    """
    power = 0
    while n > threshold:
        power += 1
        n = n >> 1
    return n << power


def strtdelta(delta: timedelta) -> str:
    days = delta.days
    hours, rem = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    s_days = f"{days} day{'s' if days > 1 else ''}, " if days > 0 else ''
    return f"{s_days}{hours:02}:{minutes:02}:{seconds:02}"


def strmbytes(n: float, i: bool = True) -> str:
    if int(n) == n:
        n = int(n)
    n_g = n / (1024 if i else 1000)
    if n_g >= 1.0:
        return strgbytes(n_g, i)
    if int(n) == n:
        n_s = f"{int(n)}"
    else:
        n_s = f"{n:.1f}"
    return f"{n_s}M{'i' if i else ''}B"


def strgbytes(n: float, i: bool = True) -> str:
    if int(n) == n:
        n_s = f"{int(n)}"
    else:
        n_s = f"{n:.1f}"
    return f"{n_s}G{'i' if i else ''}B"
