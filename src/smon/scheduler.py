# smon - slurm info script for GPU-HPC users
# Copyright © 2023  Raphael Emberger, Pascal Sager

import datetime
import json
import logging
import sched
import socket
import sys
import time
import traceback
from collections import defaultdict

from dateutil.parser import parse
from smon.main import main

from db import ClusterResources, User, Message, get_cluster_by_name, get_users_abbreviation_dict, \
    get_active_messages, add_objects, update_messages_valid_date, add_cluster_resources, SlurmProcess, \
    get_active_slurm_jobs, update_slurm_jobs

logging.basicConfig(filename=f'slurm_info_{socket.gethostname()}_{datetime.datetime.utcnow()}.log', level=logging.INFO,
                    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')


def exception_hook(type, value, tb):
    logging.exception(''.join(traceback.format_exception(type, value, tb)), exc_info=True)


sys.excepthook = exception_hook


def get_hostname():
    hosts = {
        "dgx": "DGX-1",
        "dgx2": "DGX-2",
        "dgx3": "DGX-3",
        "dgx-a100": "DGX-A100",
    }
    return hosts[socket.gethostname()]


QUERY_INTERVAL = 60
CLUSTER_NAME = get_hostname()


class NvidiaInfo:

    def __init__(self):
        self.num_gpus = None
        self.total_memory_gpus = None
        self.total_memory_used_gpus = None
        self.n_unused_gpus = None
        self.gpus = []

    def __repr__(self):
        return f"Nvidia Infos:\n\tnum_gpus='{self.num_gpus}'\n\ttotal_memory_gpus='{self.total_memory_gpus}'\n\ttotal_memory_used_gpus='{self.total_memory_used_gpus}'\n\tn_unused_gpus='{self.n_unused_gpus}'\n\tgpus='{self.gpus}'\n\n"


class GPU:
    def __init__(self, id, product_name, memory, memory_used, memory_used_ratio):
        self.id = id
        self.product_name = product_name
        self.memory = memory
        self.memory_used = memory_used
        self.memory_used_ratio = memory_used_ratio
        self.processes = []
        self.is_idle = len(self.processes) == 0

    def __repr__(self):
        return f"\n\tGPU\n\t\tid='{self.id}'\n\t\tproduct_name='{self.product_name}'\n\t\tmemory='{self.memory}'\n\t\tmemory_used='{self.memory_used}'\n\t\tmemory_used_ratio='{self.memory_used_ratio}'\n\t\tprocesses='{self.processes}'\n\t\tis_idle='{self.is_idle}'"


class GpuProcess:

    def __init__(self, is_docker_container_, is_conda_):
        self.is_docker_container = is_docker_container_
        self.is_conda = is_conda_

    def __repr__(self):
        return f"\n\t\tGpuProcess\n\t\t\tis_docker_container='{self.is_docker_container}'\n\t\t\tis_conda='{self.is_conda}'"


class DockerInfos:

    def __init__(self, num_containers):
        self.num_containers = num_containers
        self.containers = []

    def __repr__(self):
        return f"DockerInfos\n\tnum_containers='{self.num_containers}'\n\tcontainers='{self.containers}'\n\n"


class DockerContainer:

    def __init__(self, name, id, image, path, runtime, is_using_gpu, pids):
        self.name = name
        self.id = id
        self.image = image
        self.path = path
        self.runtime = runtime
        self.is_using_gpu = is_using_gpu
        self.pids = pids

    def __repr__(self):
        return f"\n\tDockerContainer\n\t\tname='{self.name}'\n\t\tid='{self.id}'\n\t\timage='{self.image}'\n\t\tpath='{self.path}'\n\t\truntime='{self.runtime}'\n\t\tis_using_gpu='{self.is_using_gpu}'\n\t\tpids='{self.pids}'"


class SlurmInfo:

    def __init__(self, num_jobs):
        self.num_jobs = num_jobs
        self.jobs = []

    def __repr__(self):
        return f"SlurmInfo\n\tnum_jobs='{self.num_jobs}'\n\tjobs='{self.jobs}'\n\n"


class SlurmJob:

    def __init__(self, name, id, username, user_group_id, state, runtime, is_interactive_bash, num_cpus, num_tasks,
                 cpus_per_task, n_gpus, min_memory, not_using_gpu=False):
        self.name = name
        self.id = id
        self.username = username
        self.user_group_id = user_group_id
        self.state = state
        self.runtime = runtime
        self.is_interactive_bash = is_interactive_bash
        self.not_using_gpu = not_using_gpu
        self.num_cpus = num_cpus
        self.num_tasks = num_tasks
        self.cpus_per_task = cpus_per_task
        self.n_gpus = n_gpus
        self.min_memory = min_memory
        self.pid = []

    def __repr__(self):
        return f"\n\tSlurmJob\n\t\tname='{self.name}'\n\t\tid='{self.id}'\n\t\tusername='{self.username}'\n\t\tuser_group_id='{self.user_group_id}'\n\t\tstate='{self.state}'\n\t\truntime='{self.runtime}'\n\t\tis_interactive_bash='{self.is_interactive_bash}'\n\t\tnot_using_gpu='{self.not_using_gpu}'\n\t\tnum_cpus='{self.num_cpus}'\n\t\tnum_tasks='{self.num_tasks}'\n\t\tcpus_per_task='{self.cpus_per_task}'\n\t\tn_gpus='{self.n_gpus}'\n\t\tmin_memory='{self.min_memory}'\n\t\tpid='{self.pid}'"


class MessageUser:
    LEVEL_INFO = 0
    LEVEL_WARN = 1
    LEVEL_DANGER = 2

    def __init__(self, severity, user, message, metadata):
        self.severity = severity
        self.user = user
        self.message = message
        self.metadata = json.loads(metadata) if isinstance(metadata, str) else metadata

    def __repr__(self):
        return f"Message\n\tseverity='{self.severity}'\n\tuser='{self.user}'\n\tmessage='{self.message}\n\tmetadata='{self.metadata}'\n\t"


def query_server():
    messages = []
    sjobs, node, partition, stats, stats_type, stats_user, gpu_info, gpu_processes, containers = main(show_all=True,
                                                                                                      extended=True)

    #### NVIDIA INFOS
    nvidia_infos = NvidiaInfo()
    nvidia_infos.num_gpus = len(gpu_info)
    container2pids = {}
    for i, gpu in gpu_info.iterrows():
        mem = gpu['memory.total [MiB]'] / 1024
        mem_used = gpu['memory.used [MiB]'] / 1024
        mem_ratio = mem_used / mem
        g = GPU(id=gpu['pci.bus_id'], product_name=gpu['name'], memory=mem, memory_used=mem_used,
                memory_used_ratio=mem_ratio)
        for j, gpu_proc in gpu_processes[gpu_processes['gpu_uuid'] == gpu['uuid']].iterrows():
            cont_id = gpu_proc['container']
            g.processes.append(GpuProcess(is_docker_container_=cont_id is not None,
                                          is_conda_=gpu_proc['conda'] is not None))
            if cont_id is not None:
                container2pids[cont_id] = [*container2pids.get(cont_id, []), gpu_proc['pid']]
        nvidia_infos.gpus.append(g)
    nvidia_infos.total_memory_gpus = round(sum(gpu_info['memory.total [MiB]']) / 1024)
    nvidia_infos.total_memory_used_gpus = round(sum(gpu_info['memory.used [MiB]']) / 1024)
    nvidia_infos.n_unused_gpus = node['gres']['gpu'] - node['gres_used']['gpu']

    ### DOCKER INFOS

    running_containers = containers[containers['State.Running']]
    docker_infos = DockerInfos(num_containers=len(running_containers))
    for container_id, container in running_containers.iterrows():
        docker_infos.containers.append(DockerContainer(
            name=container['Name'][1:], id=container['IdShort'], image=container['Image'], path=container['Path'],
            runtime=datetime.datetime.utcnow() - parse(container['State.StartedAt']).replace(tzinfo=None),
            is_using_gpu=container_id in container2pids.keys(), pids=container2pids.get(container_id)
        ))

    ### SLURM INFOS

    user_stats = defaultdict(list)
    offenders = defaultdict(list)
    slurm_infos = SlurmInfo(num_jobs=len(sjobs))
    for job_id, sjob in sjobs.iterrows():
        slurm_job = SlurmJob(
            name=sjob['name'], id=job_id, username=sjob['user'], user_group_id=sjob['group_id'],
            state=sjob['job_state'],
            runtime=datetime.datetime.utcnow() - datetime.datetime.utcfromtimestamp(sjob['start_time']),
            is_interactive_bash=sjob['is_interactive_bash_session'],
            num_cpus=int(sjob['tres_req'].get('cpu', len(sjob['CPU_IDs']))), num_tasks=sjob['num_tasks'],
            cpus_per_task=int(sjob['cpus_per_task']), n_gpus=sjob['gpus'], min_memory=sjob['pn_min_memory'] / 1024,
            not_using_gpu=not sjob['is_using_gpu']
        )
        slurm_job.pid.append(sjob['alloc_sid'])
        if slurm_job.not_using_gpu:
            offenders[sjob['user']].append(sjob)
        else:
            user_stats[sjob['user']].append(sjob)
        slurm_infos.jobs.append(slurm_job)

    if len(user_stats) > 0:
        for user, jobs in user_stats.items():
            s_jobs = [f"{job['name']}({job['job_id']})" for job in jobs]
            messages.append(MessageUser(user=user, message="is running GPU process", metadata={'jobs': s_jobs},
                                        severity=MessageUser.LEVEL_INFO))

    if len(offenders) > 0:
        for offender, jobs in offenders.items():
            s_jobs = [f"{job['name']}({job['job_id']})" for job in jobs]
            messages.append(MessageUser(user=offender, message="not using active slurm job", metadata={'jobs': s_jobs},
                                        severity=MessageUser.LEVEL_WARN))

    # print(cpusets_to_container, short_container_id_to_id, cont_id_to_info, name_to_containers, container_id_to_pids, pid_to_gpus)

    # print(short_container_id_to_id)  # dict DockerContainer.id -> full length Docker Container id
    # print(container_id_to_pids)  # Docker ID -> Nvidia Process ID
    # print(pid_to_gpus)  # Nvidia Process ID -> GPU
    # print(jobid_to_pids)  # this is a function slurm job id -> process id

    return nvidia_infos, slurm_infos, docker_infos, messages


def store_results(nvidia_infos, slurm_infos, docker_infos, messages):
    cluster = get_cluster_by_name(CLUSTER_NAME)
    users = add_all_users(slurm_infos, messages)
    add_all_users(slurm_infos, messages)
    store_messages(messages, users, cluster.id)
    cluster_resources = store_cluster_resources(cluster, nvidia_infos, slurm_infos, docker_infos)
    store_slurm_processes(slurm_infos, users, cluster.id)


def store_slurm_processes(slurm_infos, users, cluster_id):
    active_slurm_jobs = get_active_slurm_jobs(cluster_id)
    new_job_state = {active_slurm_job: {"runtime": None, "found": False} for active_slurm_job in active_slurm_jobs}
    new_jobs = []

    for slurm_job in slurm_infos.jobs:
        user = users[slurm_job.username]

        found = False
        for active_slurm_job in active_slurm_jobs:
            if active_slurm_job.job_id == slurm_job.id and active_slurm_job.user_id == user.id and active_slurm_job.not_using_gpu == slurm_job.not_using_gpu and active_slurm_job.status == slurm_job.state:
                # assume this is the same job...
                new_job_state[active_slurm_job]["runtime"] = slurm_job.runtime
                new_job_state[active_slurm_job]["found"] = True
                found = True
                break

        if not found:
            new_jobs.append(SlurmProcess(cluster_id=cluster_id,
                                         job_id=slurm_job.id,
                                         user_id=user.id,
                                         status=slurm_job.state,
                                         runtime=slurm_job.runtime,
                                         n_gpus=slurm_job.n_gpus,
                                         n_cpus=slurm_job.num_cpus,
                                         min_memory=slurm_job.min_memory,
                                         is_interactive_bash=slurm_job.is_interactive_bash,
                                         not_using_gpu=slurm_job.not_using_gpu,
                                         valid_from=datetime.datetime.utcnow()))

    update_jobs_runtime = {}
    update_jobs_validity = {}
    for job, state in new_job_state.items():
        if state["runtime"] is not None:
            update_jobs_runtime[job] = state["runtime"]
        elif not state["found"]:
            update_jobs_validity[job] = datetime.datetime.utcnow()

    add_objects(new_jobs)
    update_slurm_jobs(update_jobs_runtime, update_jobs_validity)


def store_cluster_resources(cluster, nvidia_infos, slurm_infos, docker_infos):
    mem_reserved = 0
    cpu_reserved = 0
    gpus_reserved = 0

    for slurm_job in slurm_infos.jobs:
        mem_reserved += slurm_job.min_memory
        cpu_reserved += slurm_job.num_cpus
        gpus_reserved += slurm_job.n_gpus

    cluster_res = ClusterResources(cluster_id=cluster.id,
                                   is_active=True,
                                   timestamp=datetime.datetime.utcnow(),
                                   total_memory_used_gpus=nvidia_infos.total_memory_used_gpus,
                                   total_memory_reserved=mem_reserved,
                                   total_cpu_cores_reserved=cpu_reserved,
                                   n_busy_gpus=cluster.number_of_gpus - nvidia_infos.n_unused_gpus,
                                   n_reserved_gpus=gpus_reserved,
                                   n_containers=docker_infos.num_containers,
                                   n_slurm_jobs=slurm_infos.num_jobs
                                   )

    return add_cluster_resources(cluster_res)


def add_all_users(slurm_infos, messages):
    users = get_users_abbreviation_dict()

    new_users_abbr = set()
    for slurm_job in slurm_infos.jobs:
        if slurm_job.username not in users.keys():
            new_users_abbr.add(slurm_job.username)

    for message in messages:
        if message.user not in users.keys():
            new_users_abbr.add(message.user)

    if len(new_users_abbr) > 0:
        new_users = [User(abbreviation=abbreviation, email=f"{abbreviation}@zhaw.ch") for abbreviation in
                     new_users_abbr]
        add_objects(new_users)
        users = get_users_abbreviation_dict()

    return users


def store_messages(messages, users, cluster_id):
    active_messages = get_active_messages(cluster_id)  # get all active messages
    still_active_messages = []
    new_messages = []

    for message in messages:
        user = users[message.user]

        found = False
        for active_message in active_messages:
            if user.id == active_message.user_id and message.severity == active_message.severity and message.message == active_message.message:
                still_active_messages.append(active_message)
                found = True
                break

        if not found:
            new_messages.append(
                Message(user_id=user.id, severity=message.severity, message=message.message, data=message.metadata,
                        valid_from=datetime.datetime.utcnow(), cluster_id=cluster_id))

    update_messages_valid_until = {}
    for active_message in active_messages:
        if active_message not in still_active_messages:
            update_messages_valid_until[active_message] = datetime.datetime.utcnow()

    add_objects(new_messages)
    update_messages_valid_date(update_messages_valid_until)


def debug():
    nvidia_infos, slurm_infos, docker_infos, messages = query_server()
    store_results(nvidia_infos, slurm_infos, docker_infos, messages)


class PeriodicScheduler:

    def __init__(self):
        self.scheduler = sched.scheduler(time.time, time.sleep)
        self.job_start_time = None
        self.job_end_time = None

    def start(self):
        self.scheduler.enter(0, 1, self.query_all, ())
        self.scheduler.run()

    def query_all(self):
        self.job_start_time = datetime.datetime.now()
        nvidia_infos, slurm_infos, docker_infos, messages = query_server()
        try:
            store_results(nvidia_infos, slurm_infos, docker_infos, messages)
            logging.debug("cluster info stored in database")
        except Exception as e:
            logging.exception("message")
            logging.error(e, exc_info=True)
            logging.info("Could not save results...")

        self.job_end_time = datetime.datetime.now()

        if self.job_start_time is not None and self.job_end_time is not None:
            self.scheduler.enter(QUERY_INTERVAL - (self.job_end_time - self.job_start_time).seconds, 1,
                                 self.query_all, ())
        else:
            self.scheduler.enter(QUERY_INTERVAL, 1, self.query_all, ())


if __name__ == '__main__':
    # debug()
    PeriodicScheduler().start()
