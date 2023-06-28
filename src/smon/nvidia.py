import subprocess
from typing import List

from pandas import DataFrame, read_csv, Series
from smon.log import FMT_RST, FMT_INFO1
from smon.util import strmbytes

# nvidia-smi --help-query-gpu
NVIDIA_GPU_QUERY2COLUMN = {
    "timestamp": "timestamp",
    "driver_version": "driver_version",
    "count": "count",
    "name": "name",
    "serial": "serial",
    "uuid": "uuid",
    "pci.bus_id": "pci.bus_id",
    "pci.domain": "pci.domain",
    "pci.bus": "pci.bus",
    "pci.device": "pci.device",
    "pci.device_id": "pci.device_id",
    "pci.sub_device_id": "pci.sub_device_id",
    "pcie.link.gen.current": "pcie.link.gen.current",
    "pcie.link.gen.max": "pcie.link.gen.max",
    "pcie.link.width.current": "pcie.link.width.current",
    "pcie.link.width.max": "pcie.link.width.max",
    "index": "index",
    "display_mode": "display_mode",
    "display_active": "display_active",
    "persistence_mode": "persistence_mode",
    "accounting.mode": "accounting.mode",
    "accounting.buffer_size": "accounting.buffer_size",
    "driver_model.current": "driver_model.current",
    "driver_model.pending": "driver_model.pending",
    "vbios_version": "vbios_version",
    "inforom.img": "inforom.img",
    "inforom.oem": "inforom.oem",
    "inforom.ecc": "inforom.ecc",
    "inforom.pwr": "inforom.pwr",
    "gom.current": "gom.current",
    "gom.pending": "gom.pending",
    "fan.speed": "fan.speed [%]",
    "pstate": "pstate",
    # https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html
    "clocks_throttle_reasons.supported": "clocks_throttle_reasons.supported",
    "clocks_throttle_reasons.active": "clocks_throttle_reasons.active",
    "clocks_throttle_reasons.gpu_idle": "clocks_throttle_reasons.gpu_idle",
    "clocks_throttle_reasons.applications_clocks_setting": "clocks_throttle_reasons.applications_clocks_setting",
    "clocks_throttle_reasons.sw_power_cap": "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown": "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown": "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown": "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown": "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.sync_boost": "clocks_throttle_reasons.sync_boost",
    "memory.total": "memory.total [MiB]",
    "memory.used": "memory.used [MiB]",
    "memory.free": "memory.free [MiB]",
    "compute_mode": "compute_mode",
    "utilization.gpu": "utilization.gpu [%]",
    "utilization.memory": "utilization.memory [%]",
    "encoder.stats.sessionCount": "encoder.stats.sessionCount",
    "encoder.stats.averageFps": "encoder.stats.averageFps",
    "encoder.stats.averageLatency": "encoder.stats.averageLatency",
    "ecc.mode.current": "ecc.mode.current",
    "ecc.mode.pending": "ecc.mode.pending",
    "ecc.errors.corrected.volatile.device_memory": "ecc.errors.corrected.volatile.device_memory",
    "ecc.errors.corrected.volatile.dram": "ecc.errors.corrected.volatile.dram",
    "ecc.errors.corrected.volatile.register_file": "ecc.errors.corrected.volatile.register_file",
    "ecc.errors.corrected.volatile.l1_cache": "ecc.errors.corrected.volatile.l1_cache",
    "ecc.errors.corrected.volatile.l2_cache": "ecc.errors.corrected.volatile.l2_cache",
    "ecc.errors.corrected.volatile.texture_memory": "ecc.errors.corrected.volatile.texture_memory",
    "ecc.errors.corrected.volatile.cbu": "ecc.errors.corrected.volatile.cbu",
    "ecc.errors.corrected.volatile.sram": "ecc.errors.corrected.volatile.sram",
    "ecc.errors.corrected.volatile.total": "ecc.errors.corrected.volatile.total",
    "ecc.errors.corrected.aggregate.device_memory": "ecc.errors.corrected.aggregate.device_memory",
    "ecc.errors.corrected.aggregate.dram": "ecc.errors.corrected.aggregate.dram",
    "ecc.errors.corrected.aggregate.register_file": "ecc.errors.corrected.aggregate.register_file",
    "ecc.errors.corrected.aggregate.l1_cache": "ecc.errors.corrected.aggregate.l1_cache",
    "ecc.errors.corrected.aggregate.l2_cache": "ecc.errors.corrected.aggregate.l2_cache",
    "ecc.errors.corrected.aggregate.texture_memory": "ecc.errors.corrected.aggregate.texture_memory",
    "ecc.errors.corrected.aggregate.cbu": "ecc.errors.corrected.aggregate.cbu",
    "ecc.errors.corrected.aggregate.sram": "ecc.errors.corrected.aggregate.sram",
    "ecc.errors.corrected.aggregate.total": "ecc.errors.corrected.aggregate.total",
    "ecc.errors.uncorrected.volatile.device_memory": "ecc.errors.uncorrected.volatile.device_memory",
    "ecc.errors.uncorrected.volatile.dram": "ecc.errors.uncorrected.volatile.dram",
    "ecc.errors.uncorrected.volatile.register_file": "ecc.errors.uncorrected.volatile.register_file",
    "ecc.errors.uncorrected.volatile.l1_cache": "ecc.errors.uncorrected.volatile.l1_cache",
    "ecc.errors.uncorrected.volatile.l2_cache": "ecc.errors.uncorrected.volatile.l2_cache",
    "ecc.errors.uncorrected.volatile.texture_memory": "ecc.errors.uncorrected.volatile.texture_memory",
    "ecc.errors.uncorrected.volatile.cbu": "ecc.errors.uncorrected.volatile.cbu",
    "ecc.errors.uncorrected.volatile.sram": "ecc.errors.uncorrected.volatile.sram",
    "ecc.errors.uncorrected.volatile.total": "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.device_memory": "ecc.errors.uncorrected.aggregate.device_memory",
    "ecc.errors.uncorrected.aggregate.dram": "ecc.errors.uncorrected.aggregate.dram",
    "ecc.errors.uncorrected.aggregate.register_file": "ecc.errors.uncorrected.aggregate.register_file",
    "ecc.errors.uncorrected.aggregate.l1_cache": "ecc.errors.uncorrected.aggregate.l1_cache",
    "ecc.errors.uncorrected.aggregate.l2_cache": "ecc.errors.uncorrected.aggregate.l2_cache",
    "ecc.errors.uncorrected.aggregate.texture_memory": "ecc.errors.uncorrected.aggregate.texture_memory",
    "ecc.errors.uncorrected.aggregate.cbu": "ecc.errors.uncorrected.aggregate.cbu",
    "ecc.errors.uncorrected.aggregate.sram": "ecc.errors.uncorrected.aggregate.sram",
    "ecc.errors.uncorrected.aggregate.total": "ecc.errors.uncorrected.aggregate.total",
    "retired_pages.single_bit_ecc.count": "retired_pages.single_bit_ecc.count",
    "retired_pages.double_bit.count": "retired_pages.double_bit.count",
    "retired_pages.pending": "retired_pages.pending",
    "temperature.gpu": "temperature.gpu",
    "temperature.memory": "temperature.memory",
    "power.management": "power.management",
    "power.draw": "power.draw [W]",
    "power.limit": "power.limit [W]",
    "enforced.power.limit": "enforced.power.limit [W]",
    "power.default_limit": "power.default_limit [W]",
    "power.min_limit": "power.min_limit [W]",
    "power.max_limit": "power.max_limit [W]",
    "clocks.current.graphics": "clocks.current.graphics [MHz]",
    "clocks.current.sm": "clocks.current.sm [MHz]",
    "clocks.current.memory": "clocks.current.memory [MHz]",
    "clocks.current.video": "clocks.current.video [MHz]",
    "clocks.applications.graphics": "clocks.applications.graphics [MHz]",
    "clocks.applications.memory": "clocks.applications.memory [MHz]",
    "clocks.default_applications.graphics": "clocks.default_applications.graphics [MHz]",
    "clocks.default_applications.memory": "clocks.default_applications.memory [MHz]",
    "clocks.max.graphics": "clocks.max.graphics [MHz]",
    "clocks.max.sm": "clocks.max.sm [MHz]",
    "clocks.max.memory": "clocks.max.memory [MHz]",
    "mig.mode.current": "mig.mode.current",
    "mig.mode.pending": "mig.mode.pending"
}
NVIDIA_CLOCK_SPEED_THROTTLE_REASONS = [
    "clocks_throttle_reasons.gpu_idle",
    "clocks_throttle_reasons.applications_clocks_setting",
    "clocks_throttle_reasons.sw_power_cap",
    "clocks_throttle_reasons.hw_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "clocks_throttle_reasons.sync_boost",
]
NVIDIA_GPU_QUERIES = [
    'name', 'uuid', 'pci.bus_id', 'index', *NVIDIA_CLOCK_SPEED_THROTTLE_REASONS, 'memory.total', 'memory.used',
    'memory.free',
    'utilization.gpu', 'utilization.memory', 'temperature.gpu', 'temperature.memory', 'power.draw',
]


def nvidia_smi_gpu(fields: List[str] = None, units=False) -> DataFrame:
    if fields is None:
        fields = NVIDIA_GPU_QUERIES  # list(NVIDIA_GPU_QUERY2COLUMN.keys())
    opts = ['csv']
    if not units:
        opts.append('nounits')
    sproc = subprocess.Popen(
        ['/usr/bin/nvidia-smi', f'--query-gpu={",".join(fields)}', f'--format={",".join(opts)}'],
        stdout=subprocess.PIPE
    )
    df = read_csv(sproc.stdout, sep=', ', engine='python')
    if len(df.columns) == 1:
        # "No devices were found"
        df = DataFrame(columns=[NVIDIA_GPU_QUERY2COLUMN[k] for k in fields])
    return df


NVIDIA_COMPUTE_QUERY2COLUMN = {
    "timestamp": "timestamp",
    "gpu_name": "gpu_name",
    "gpu_bus_id": "gpu_bus_id",
    "gpu_serial": "gpu_serial",
    "gpu_uuid": "gpu_uuid",
    "pid": "pid",
    "process_name": "process_name",
    "used_gpu_memory": "used_gpu_memory [MiB]"
}
NVIDIA_COMPUTE_QUERIES = [
    "gpu_uuid", "pid", "process_name", "used_gpu_memory",
]


def nvidia_smi_compute(fields: List[str] = None, units=False) -> DataFrame:
    if fields is None:
        fields = NVIDIA_COMPUTE_QUERIES  # list(NVIDIA_COMPUTE_QUERY2COLUMN.keys())
    opts = ['csv']
    if not units:
        opts.append('nounits')
    sproc = subprocess.Popen(
        ['/usr/bin/nvidia-smi', f'--query-compute-apps={",".join(fields)}', f'--format={",".join(opts)}'],
        stdout=subprocess.PIPE
    )
    df = read_csv(sproc.stdout, sep=', ', engine='python')
    if len(df.columns) == 1:
        # "No devices were found"
        df = DataFrame(columns=[NVIDIA_COMPUTE_QUERY2COLUMN[k] for k in fields])
    return df


def gpu_to_string(gpu_info: Series, gpu_id: int, fmt_info: str = FMT_INFO1) -> str:
    temp_gpu = f'{fmt_info}{gpu_info["temperature.gpu"]}°C{FMT_RST}'
    temp_mem = f'{fmt_info}{gpu_info["temperature.memory"]}°C{FMT_RST}'
    load_util = f'{fmt_info}{gpu_info["utilization.gpu [%]"]}%{FMT_RST}'
    mem_used = f'{fmt_info}{strmbytes(gpu_info["memory.used [MiB]"])}{FMT_RST}'
    mem_total = f'{fmt_info}{strmbytes(gpu_info["memory.total [MiB]"])}'
    mem_util = f'{fmt_info}{gpu_info["utilization.gpu [%]"]}%{FMT_RST}'
    return f'GPU:{fmt_info}{gpu_id}{FMT_RST}' \
           f' - {gpu_info["name"]}' \
           f', load: {load_util} ({temp_gpu})' \
           f', mem: {mem_util} ({mem_used}/{mem_total}, {temp_mem})' \
           f', power: {fmt_info}{gpu_info["power.draw [W]"]}W{FMT_RST}'
