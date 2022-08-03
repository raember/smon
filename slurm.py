#!/usr/bin/env python3

import argparse
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Callable, Tuple

import yaml
from dateutil.parser import parse, ParserError
from docker import client
from psutil import Process

TRAINING_RUNTIME_WARN_THRESHOLD = timedelta(days=7.0)
TRAINING_RUNTIME_ERROR_THRESHOLD = timedelta(days=14.0)
TRAINING_RUNTIME_THRESHOLD = (TRAINING_RUNTIME_WARN_THRESHOLD, TRAINING_RUNTIME_ERROR_THRESHOLD)
CONTAINER_RUNTIME_WARN_THRESHOLD = timedelta(days=30.0)
CONTAINER_RUNTIME_ERROR_THRESHOLD = timedelta(days=60.0)
CONTAINER_RUNTIME_THRESHOLD = (CONTAINER_RUNTIME_WARN_THRESHOLD, CONTAINER_RUNTIME_ERROR_THRESHOLD)
SJOB_RUNTIME_WARN_THRESHOLD = timedelta(days=30.0)
SJOB_RUNTIME_ERROR_THRESHOLD = timedelta(days=60.0)
SJOB_RUNTIME_THRESHOLD = (SJOB_RUNTIME_WARN_THRESHOLD, SJOB_RUNTIME_ERROR_THRESHOLD)

GREEN = 32
YELLOW = 33
BLUE = 34
RED = 31


def msg(s: str, level: int, color: int):
    if level == 0:
        print(f"\033[1;{color}m=>\033[m {s}\033[m")
    else:
        print(f"  \033[1;{color}m{'-' * level}>\033[m {s}\033[m")


def msg1(s: str):
    msg(s, 0, GREEN)


def msg2(s: str):
    msg(s, 1, BLUE)


def msg3(s: str):
    msg(s, 2, BLUE)


def msg4(s: str):
    msg(s, 3, BLUE)


WARN = f"\033[1;{YELLOW}m"


def warn(s: str, level: int):
    msg(f"{WARN}{s}\033[m", level, YELLOW)


def warn1(s: str):
    warn(s, 0)


def warn2(s: str):
    warn(s, 1)


def warn3(s: str):
    warn(s, 2)


def warn4(s: str):
    warn(s, 3)


ERR = f"\033[1;{RED}m"


def err(s: str, level: int):
    msg(f"{ERR}{s}\033[m", level, RED)


def err1(s: str):
    err(s, 0)


def err2(s: str):
    err(s, 1)


def err3(s: str):
    err(s, 2)


def err4(s: str):
    err(s, 3)


def parse_value(val: str):
    if val.isdigit():
        val = int(val)
    else:
        try:
            val = float(val)
        except ValueError:
            if len(val) > 0 and val[0].isdigit() and not re.match(r'\d:\d', val):
                try:
                    return parse(val)
                except ParserError:
                    pass
            elif val == '(null)':
                val = None
            elif val.startswith('/'):
                val = Path(val)
    return val


def get_sjobs() -> list:
    sjobs = []
    for sjob_str in subprocess.check_output(('scontrol', 'show', 'job')).decode().strip('\n ').split('\n\n'):
        sjob = {}
        for field in sjob_str.replace('\n', '').split(' '):
            if field == '':
                continue
            key, val = field.split('=', maxsplit=1)
            val = parse_value(val)
            sjob[key] = val
        sjobs.append(sjob)
    return sjobs


def jobid_to_pids(jobid: int) -> List[Process]:
    pids = []
    cmd = subprocess.run(('scontrol', 'listpids', str(jobid)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if cmd.returncode > 0:
        return pids
    for line in cmd.stdout.decode().splitlines()[1:]:
        pids.append(Process(int(line.split(' ', maxsplit=1)[0])))
    return pids


def pid_to_docker_pid(proc: Process) -> Process:
    print(f"{proc.cpu_num()}({proc.cpu_percent()}%), {proc.memory_info()}({proc.memory_percent()}%)")
    p = None
    for subproc in proc.children(recursive=True):
        if subproc.name() == 'docker':
            p = subproc
    return p


def nvidia_smi() -> dict:
    out_str = subprocess.run(('nvidia-smi', '-q'), stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode()
    blocks = out_str.split('\nGPU ')
    head = blocks[0]
    gpu_strs = blocks[1:]
    out = {}
    # parse head
    for line in head.splitlines():
        if ':' in line:
            kvp = line.split(':')
            out[kvp[0].strip()] = parse_value(kvp[1].strip())
    out['CUDA Version'] = str(out['CUDA Version'])

    # parse gpus
    gpus = []
    for gpu_str in gpu_strs:
        text = '\n'.join(gpu_str.splitlines()[1:]).replace('  ', ' ')
        text = re.sub(r'^([^:]*?)$', r'\1:', text, flags=re.MULTILINE)
        text = re.sub(r'\s+:', ':', text, flags=re.MULTILINE)
        text = re.sub(r'(HW Slowdown)\s*:.+$', r'\1:', text, flags=re.MULTILINE)
        text = re.sub(r'(Process ID)\s*: (\d+)$', r'\2:', text, flags=re.MULTILINE)
        text = re.sub(r': (\d+([\D\s]\d+)+|0x[0-9a-fA-F]+)$', r': "\1"', text, flags=re.MULTILINE)
        text = text.strip(':')
        yml = yaml.safe_load(text)
        gpus.append(yml)
    out['GPUs'] = gpus
    return out


def print_proc(proc: Process, level: int = 1):
    threshold = (timedelta(days=356), timedelta(days=356))
    if is_docker(proc) or is_docker_container(proc):
        threshold = CONTAINER_RUNTIME_THRESHOLD
    elif proc.name() == 'python':
        threshold = TRAINING_RUNTIME_THRESHOLD
    msg(f"{proc.name()}({proc.pid}, {judge_runtime(datetime.fromtimestamp(proc.create_time()), threshold)}): $ {' '.join(proc.cmdline())}",
        level, BLUE)


def proc_tree(proc: Process, level: int, func: Callable = lambda p, l: None) -> bool:
    result = bool(func(proc, level))
    for child in proc.children(recursive=False):
        result |= proc_tree(child, level + 1, func)
    return result


def is_docker(proc: Process) -> bool:
    return proc.cmdline()[0] == 'docker'


def is_docker_container(proc: Process) -> bool:
    return proc.cmdline()[0] == '/usr/bin/containerd-shim-runc-v2'


def get_container_from(proc: Process) -> Process:
    while proc is not None and not is_docker_container(proc):
        proc = proc.parent()
    if proc is None:
        err2("COULD NOT FIND CONTAINER PROCESS")
    else:
        return proc


def to_container_id(proc: Process) -> str:
    return re.search(r"-id ([\da-f]{64})", ' '.join(proc.cmdline())).group(1)


def is_tmux(proc: Process) -> bool:
    return proc.cmdline()[0] == 'tmux'


def is_screen(proc: Process) -> bool:
    return proc.cmdline()[0] == 'screen'


def is_conda(proc: Process) -> bool:
    out = False
    for cpid in proc.children():
        out |= 'conda' in cpid.cmdline()[0]
    return out


UNITS = {
    'B': 1, 'b': 1,
    'kB': 1000, 'KB': 1000, 'KiB': 1024, 'Kb': 1024,
    'MB': 1000 ** 2, 'MiB': 1024 ** 2, 'Mb': 1024 ** 2,
    'GB': 1000 ** 3, 'GiB': 1024 ** 3, 'Gb': 1024 ** 3,
    'TB': 1000 ** 3, 'TiB': 1024 ** 3, 'Tb': 1024 ** 3,
}


def parse_data_size(str_bytes: str) -> int:
    num, unit = str_bytes.strip().split()[:2]
    return int(float(num) * UNITS[unit])


def judge_runtime(timestamp: datetime, thresholds: Tuple[timedelta, timedelta]) -> str:
    runtime = datetime.now() - timestamp
    COLOR = ''
    RUNTIME_WARN_THRESHOLD, RUNTIME_ERROR_THRESHOLD = thresholds
    if runtime > RUNTIME_WARN_THRESHOLD:
        COLOR = f';{YELLOW}'
    if runtime > RUNTIME_ERROR_THRESHOLD:
        COLOR = f';{RED}'
    return f"\033[1{COLOR}m{str(runtime)}\033[m"


def find_gpu(proc: Process, level):
    # print_proc(proc, level)
    uses_gpu = False
    if is_docker(proc):
        for arg in proc.cmdline():
            if arg.startswith('--cpuset-cpus='):
                cpuset = arg.split('=', maxsplit=1)[1]
                cont_info = cpusets_to_container.get(cpuset)
                if cont_info is not None:
                    del cpusets_to_container[cpuset]
            else:
                arg = short_container_id_to_id.get(arg, arg)
                if arg in cont_id_to_info.keys():
                    arg = cont_id_to_info[arg]['Name']
                if arg in name_to_containers.keys():
                    cont_infos = name_to_containers[arg]
                    if len(cont_infos) > 1:
                        if args.verbose:
                            warn(f"Links to one of {len(cont_infos)} containers", level + 1)
                        continue
                    else:
                        cont_info = cont_infos[0]
                else:
                    continue
            if args.verbose:
                msg(f"Linked to \033[{BLUE}m{cont_info['Name']}\033[m(\033[{BLUE}m{cont_info['Config']['Image']}\033[m, \033[1m{cont_info['State']['Pid']}\033[m) [{cont_info['Path']}]: {judge_runtime(parse(cont_info['State']['StartedAt']).replace(tzinfo=None), CONTAINER_RUNTIME_THRESHOLD)}{WARN}",
                    level + 1, BLUE)
            for nvidia_proc in container_id_to_pids.get(cont_info['Id'], []):
                for gpu in pid_to_gpus.get(nvidia_proc.pid, []):
                    if args.verbose:
                        msg(f"\033[{GREEN}mUses GPU gpu:{gpu['Minor Number']}\033[m", level + 2, GREEN)
                    uses_gpu = True
            return uses_gpu
        err("FAILED TO LINK TO CORRECT DOCKER CONTAINER", level + 1)
    elif is_tmux(proc) or is_screen(proc):
        err(f"ILLEGAL TMUX/SCREEN SESSION WITHIN SLURM JOB", level + 1)
    elif proc.pid in pid_to_gpus.keys():
        if args.verbose:
            warn("Training is running directly inside SLURM job", level + 1)
            if is_conda(proc):
                msg(f"\033[{GREEN}mConda environment detected", level + 2, GREEN)
            else:
                warn("No conda environment detected", level + 2)
            for gpu in pid_to_gpus.get(proc.pid, []):
                msg(f"\033[{GREEN}mUses GPU gpu:{gpu['Minor Number']}\033[m", level + 1, GREEN)
        uses_gpu = True
    return uses_gpu


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true', help="Display full output", default=False)
    args = parser.parse_args()
    nvidia_info = nvidia_smi()
    container_id_to_pids = {}
    short_container_id_to_id = {}
    pid_to_gpus = {}
    n_free_gpus = 0
    total_mem = 0
    total_mem_used = 0
    if args.verbose:
        print(f"NUM GPUS: {nvidia_info['Attached GPUs']}")
    for i, gpu in enumerate(nvidia_info['GPUs']):
        if args.verbose:
            msg1(f"gpu:{i}: {gpu['PCI']['Bus Id']} - {gpu['Product Name']}")
        mem = parse_data_size(gpu['FB Memory Usage']['Total'])
        mem_used = parse_data_size(gpu['FB Memory Usage']['Used'])
        total_mem += mem
        total_mem_used += mem_used
        mem_ratio = mem_used / mem
        color = RED if mem_ratio > 0.8 else YELLOW if mem_ratio > 0.5 else GREEN
        if args.verbose:
            msg2(
                f"Memory: \033[{color}m{mem_used / 1024 ** 3:.1f}GiB\033[m/\033[{BLUE}m{mem / 1024 ** 3:.1f}GiB\033[m (\033[{color}m{mem_ratio * 100:.2f}%\033[m)")
        procs = gpu['Processes']
        if procs == 'None':
            n_free_gpus += 1
            continue
        procs['Entries'] = {}
        for key in list(procs.keys()):
            if isinstance(key, int):
                proc_data = procs[key]
                del procs[key]
                procs['Entries'][key] = proc_data
                proc = Process(key)
                if args.verbose:
                    print_proc(proc, 1)
                gpus = pid_to_gpus.get(key, [])
                gpus.append(gpu)
                pid_to_gpus[key] = gpus
                while proc is not None and not is_docker_container(proc):
                    proc = proc.parent()
                if proc is None:
                    warn3("Process is not running inside a container")
                    if is_conda(Process(key)):
                        msg(f"\033[{GREEN}mConda environment detected", 3, GREEN)
                    else:
                        warn4("No conda environment detected")
                else:
                    if args.verbose:
                        print_proc(proc, 2)
                    cont_id = to_container_id(proc)
                    short_container_id_to_id[cont_id[:12]] = cont_id
                    nvidia_procs = container_id_to_pids.get(cont_id, set())
                    nvidia_procs.add(Process(key))
                    container_id_to_pids[cont_id] = nvidia_procs

    if args.verbose:
        print("=" * 100)
    docker = client.Client(base_url='unix://var/run/docker.sock')
    cont_id_to_info = {}
    cpusets_to_container = {}
    name_to_containers = {}
    # container_to_gpu = {}
    containers = docker.containers()
    if args.verbose:
        print(f"NUM CONTAINERS: {len(containers)}")
    for container in containers:
        cont_info = docker.inspect_container(container['Id'])
        cont_id_to_info[container['Id']] = cont_info
        if args.verbose:
            msg1(
                f"\033[{BLUE}m{cont_info['Name']}\033[m \033[1m{cont_info['Id'][:12]}\033[m (\033[1m{cont_info['Config']['Image']}\033[m) [{cont_info['Path']}]: {judge_runtime(parse(cont_info['State']['StartedAt']).replace(tzinfo=None), CONTAINER_RUNTIME_THRESHOLD)}")
        # print_proc(get_container_from(Process(cont_info['State']['Pid'])), 1)
        is_using_gpu = container['Id'] in container_id_to_pids.keys()
        if args.verbose:
            if not is_using_gpu:
                warn3("not using GPU")
            else:
                msg(f"\033[{GREEN}mUses GPU", 3, GREEN)
        cpusets_to_container[cont_info['HostConfig']['CpusetCpus']] = cont_info
        # Sometimes people attach or exec bash on running containers instead of keeping
        # the run open, so we associate the names with the container info
        conts = name_to_containers.get(container['Names'][0].strip('/'), [])
        conts.append(cont_info)
        name_to_containers[container['Names'][0].strip('/')] = conts

    if args.verbose:
        print("=" * 100)
    sjobs = get_sjobs()
    user_stats = defaultdict(list)
    offenders = defaultdict(list)
    if args.verbose:
        print(f"NUM SJOBS: {len(sjobs)}")
    for sjob in sjobs:
        username, uid = tuple(sjob["UserId"].strip(')').split('('))
        if args.verbose:
            msg1(
                f'\033[{BLUE}m{sjob["JobName"]}\033[m({sjob["JobId"]}) by \033[{BLUE}m{username}\033[m({uid}):{sjob["GroupId"]} [\033[1m{sjob["JobState"]}\033[m, run time {judge_runtime(sjob["StartTime"], SJOB_RUNTIME_THRESHOLD)}]')
            if sjob['Command'] == 'bash':
                warn2("Uses interactive bash session")
        found = False
        for pid in jobid_to_pids(sjob['JobId']):
            found |= proc_tree(pid, 1, find_gpu)
        if not found:
            offenders[username].append(sjob)
            if args.verbose:
                err2("Not using GPU")
        else:
            user_stats[username].append(sjob)

    if args.verbose:
        print("=" * 100)
        print("Statistics")
    color = YELLOW if n_free_gpus > 0 else RED if n_free_gpus == 0 else GREEN
    s_free_gpus = f"\033[{color}m{n_free_gpus}\033[m"
    msg1(
        f"Free GPUs: {s_free_gpus}/\033[{BLUE}m{len(nvidia_info['GPUs'])}\033[m (\033[{color}m{n_free_gpus / len(nvidia_info['GPUs']) * 100:.2f}%\033[m)")
    mem_ratio = total_mem_used / total_mem
    color = RED if mem_ratio > 0.8 else YELLOW if mem_ratio > 0.5 else GREEN
    msg1(
        f"Memory: \033[{color}m{total_mem_used / 1024 ** 3:.1f}GiB\033[m/\033[{BLUE}m{total_mem / 1024 ** 3:.1f}GiB\033[m (\033[{color}m{mem_ratio * 100:.2f}%\033[m)")
    if len(user_stats) > 0:
        msg1("Users running GPU processes:")
        for user, jobs in user_stats.items():
            s_jobs = [f"{job['JobName']}({job['JobId']})" for job in jobs]
            msg(f"\033[{GREEN}m{user} ({len(s_jobs)}): [{', '.join(s_jobs)}]", 2, GREEN)
    if len(offenders) > 0:
        msg1("Users reserving a slurm job but not actually using it:")
        for offender, jobs in offenders.items():
            s_jobs = [f"{job['JobName']}({job['JobId']})" for job in jobs]
            err2(f"{offender} ({len(s_jobs)}): [{', '.join(s_jobs)}]")
    if args.verbose:
        print("Done")
