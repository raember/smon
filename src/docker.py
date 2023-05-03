# from docker.client import Client
import docker
from docker.models.containers import Container
from pandas import DataFrame, Series

from src.logging import FMT_INFO1, FMT_RST

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
                        gpu_uuids.append(gpu_info[gpu_info['uuid'] == gpu_uuid].index.item())
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
    df = DataFrame(containers)
    return df.set_index('Id')


def container_to_string(container_info: Series, container_id: str, fmt_info: str = FMT_INFO1) -> str:
    return f'"{fmt_info}{container_info["Name"]}{FMT_RST}"' \
           f' {fmt_info}{container_id[:12]}{FMT_RST} ({fmt_info}{container_info["Config.Image"]}{FMT_RST})'
