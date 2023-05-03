from datetime import datetime
from typing import List, Callable, Dict

from psutil import Process

from src.logging import FMT_INFO1, FMT_RST


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
    start_time = datetime.utcfromtimestamp(process.create_time())
    return f'"{process.name()}" (started {datetime.utcnow() - start_time} ago)' + user


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
