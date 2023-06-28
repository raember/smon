# smon - slurm info script for GPU-HPC users
# Copyright © 2023  Raphael Emberger

import docker
from docker.models.containers import Container
from pandas import DataFrame, Series

from smon.log import FMT_INFO1, FMT_RST

CONTAINER_COLUMNS = [
    'Id', 'Created', 'Path', 'Args', 'Image', 'ResolvConfPath', 'HostnamePath', 'HostsPath', 'LogPath', 'Name',
    'RestartCount', 'Driver', 'Platform', 'MountLabel', 'ProcessLabel', 'AppArmorProfile', 'ExecIDs', 'Mounts',
    'IdShort', 'State.Status', 'State.Running', 'State.Paused', 'State.Restarting', 'State.OOMKilled', 'State.Dead',
    'State.Pid', 'State.ExitCode', 'State.Error', 'State.StartedAt', 'State.FinishedAt', 'HostConfig.Binds',
    'HostConfig.ContainerIDFile', 'HostConfig.LogConfig', 'HostConfig.NetworkMode', 'HostConfig.PortBindings',
    'HostConfig.RestartPolicy', 'HostConfig.AutoRemove', 'HostConfig.VolumeDriver', 'HostConfig.VolumesFrom',
    'HostConfig.CapAdd', 'HostConfig.CapDrop', 'HostConfig.CgroupnsMode', 'HostConfig.Dns', 'HostConfig.DnsOptions',
    'HostConfig.DnsSearch', 'HostConfig.ExtraHosts', 'HostConfig.GroupAdd', 'HostConfig.IpcMode', 'HostConfig.Cgroup',
    'HostConfig.Links', 'HostConfig.OomScoreAdj', 'HostConfig.PidMode', 'HostConfig.Privileged',
    'HostConfig.PublishAllPorts', 'HostConfig.ReadonlyRootfs', 'HostConfig.SecurityOpt', 'HostConfig.UTSMode',
    'HostConfig.UsernsMode', 'HostConfig.ShmSize', 'HostConfig.Runtime', 'HostConfig.ConsoleSize',
    'HostConfig.Isolation', 'HostConfig.CpuShares', 'HostConfig.Memory', 'HostConfig.NanoCpus',
    'HostConfig.CgroupParent', 'HostConfig.BlkioWeight', 'HostConfig.BlkioWeightDevice',
    'HostConfig.BlkioDeviceReadBps', 'HostConfig.BlkioDeviceWriteBps', 'HostConfig.BlkioDeviceReadIOps',
    'HostConfig.BlkioDeviceWriteIOps', 'HostConfig.CpuPeriod', 'HostConfig.CpuQuota', 'HostConfig.CpuRealtimePeriod',
    'HostConfig.CpuRealtimeRuntime', 'HostConfig.CpusetCpus', 'HostConfig.CpusetMems', 'HostConfig.Devices',
    'HostConfig.DeviceCgroupRules', 'HostConfig.DeviceRequests', 'HostConfig.KernelMemory',
    'HostConfig.KernelMemoryTCP', 'HostConfig.MemoryReservation', 'HostConfig.MemorySwap',
    'HostConfig.MemorySwappiness', 'HostConfig.OomKillDisable', 'HostConfig.PidsLimit', 'HostConfig.Ulimits',
    'HostConfig.CpuCount', 'HostConfig.CpuPercent', 'HostConfig.IOMaximumIOps', 'HostConfig.IOMaximumBandwidth',
    'HostConfig.MaskedPaths', 'HostConfig.ReadonlyPaths', 'GraphDriver.Data', 'GraphDriver.Name', 'Config.Hostname',
    'Config.Domainname', 'Config.User', 'Config.AttachStdin', 'Config.AttachStdout', 'Config.AttachStderr',
    'Config.ExposedPorts', 'Config.Tty', 'Config.OpenStdin', 'Config.StdinOnce', 'Config.Env', 'Config.Cmd',
    'Config.Image', 'Config.Volumes', 'Config.WorkingDir', 'Config.Entrypoint', 'Config.OnBuild', 'Config.Labels',
    'NetworkSettings.Bridge', 'NetworkSettings.SandboxID', 'NetworkSettings.HairpinMode',
    'NetworkSettings.LinkLocalIPv6Address', 'NetworkSettings.LinkLocalIPv6PrefixLen', 'NetworkSettings.Ports',
    'NetworkSettings.SandboxKey', 'NetworkSettings.SecondaryIPAddresses', 'NetworkSettings.SecondaryIPv6Addresses',
    'NetworkSettings.EndpointID', 'NetworkSettings.Gateway', 'NetworkSettings.GlobalIPv6Address',
    'NetworkSettings.GlobalIPv6PrefixLen', 'NetworkSettings.IPAddress', 'NetworkSettings.IPPrefixLen',
    'NetworkSettings.IPv6Gateway', 'NetworkSettings.MacAddress', 'NetworkSettings.Networks', 'GPUs'
]

# client = Client(base_url='unix://var/run/docker.sock')
client = docker.from_env()


def get_running_containers(gpu_info: DataFrame) -> DataFrame:
    containers = []

    def flatten_dict(d: dict, key: str):
        for k, v in d[key].items():
            d[f'{key}.{k}'] = v
        del d[key]

    for container in client.containers.list():
        container: Container
        # print('=' * 100)
        # print(container.name)
        container_info = container.attrs
        container_info['IdShort'] = container_info['Id'][:12]
        flatten_dict(container_info, 'State')
        flatten_dict(container_info, 'HostConfig')
        flatten_dict(container_info, 'GraphDriver')
        flatten_dict(container_info, 'Config')
        flatten_dict(container_info, 'NetworkSettings')
        gpu_uuids = []
        for env in container_info['Config.Env']:
            if env.startswith('NVIDIA_VISIBLE_DEVICES='):  # TODO: Does this really determine which GPUs are visible?
                gpu_uuids_str = env.split('=')[-1]
                if gpu_uuids_str == 'all':
                    gpu_uuids = list(range(8))
                elif gpu_uuids_str == '':
                    gpu_uuids = []
                else:
                    for gpu_uuid in gpu_uuids_str.split(','):
                        gpu = gpu_info[gpu_info['uuid'] == gpu_uuid]
                        if len(gpu) != 1:
                            # If inside a slurm session, we can't see all resources!
                            continue
                        gpu_uuids.append(gpu.index.item())
                break
        container_info['GPUs'] = gpu_uuids
        containers.append(container_info)

        # TODO: Check which method gives certainty about visibility of GPUs:
        # print(f'ENV: {len(gpu_uuids)}')
        # ec, s = container.exec_run('python -c "import torch; print(torch.cuda.device_count());"', tty=True)
        # print(f"[{ec}]Torch: {s.decode()}")
        # ec, s = container.exec_run('bash -c "nvidia-smi --query-gpu=gpu_name,gpu_bus_id --format=csv,noheader | wc -l"', tty=True)
        # print(f"[{ec}]Nvidia-smi: {s.decode()}")
        # ec, s = container.exec_run('bash -c "ls -l /proc/driver/nvidia/gpus | tail -n +2 | wc -l"', tty=True)
        # print(f"[{ec}]ls /proc/driver/nvidia/gpus: {s.decode()}")
    df = DataFrame(containers, columns=CONTAINER_COLUMNS)
    return df.set_index('Id')


def container_to_string(container_info: Series, container_id: str, fmt_info: str = FMT_INFO1) -> str:
    running = '' if container_info['State.Running'] else f' ({fmt_info}not running{FMT_RST})'
    return f'"{fmt_info}{container_info["Name"][1:]}{FMT_RST}"' \
           f' {fmt_info}{container_id[:12]}{FMT_RST} ({fmt_info}{container_info["Config.Image"]}{FMT_RST})' + running
