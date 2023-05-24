import subprocess
from typing import List

from pandas import DataFrame, read_csv, Series
from smon.log import FMT_RST, FMT_INFO1
from smon.util import strmbytes

# https://docs.nvidia.com/deploy/nvml-api/group__nvmlClocksThrottleReasons.html
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
# nvidia-smi --help-query-gpu
NVIDIA_GPU_QUERIES = [
    "timestamp",
    "driver_version",
    "count",
    "name",
    "serial",
    "uuid",
    "pci.bus_id",
    "pci.domain",
    "pci.bus",
    "pci.device",
    "pci.device_id",
    "pci.sub_device_id",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "index",
    "display_mode",
    "display_active",
    "persistence_mode",
    "accounting.mode",
    "accounting.buffer_size",
    "driver_model.current",
    "driver_model.pending",
    "vbios_version",
    "inforom.img",
    "inforom.oem",
    "inforom.ecc",
    "inforom.pwr",
    "gom.current",
    "gom.pending",
    "fan.speed",  # "fan.speed [%]"
    "pstate",
    "clocks_throttle_reasons.supported",
    "clocks_throttle_reasons.active",
    *NVIDIA_CLOCK_SPEED_THROTTLE_REASONS,
    "memory.total",  # "memory.total [MHz]"
    "memory.used",  # "memory.used [MHz]"
    "memory.free",  # "memory.free [MHz]"
    "compute_mode",
    "utilization.gpu",  # "utilization.gpu [%]"
    "utilization.memory",  # "utilization.memory [%]"
    "encoder.stats.sessionCount",
    "encoder.stats.averageFps",
    "encoder.stats.averageLatency",
    "ecc.mode.current",
    "ecc.mode.pending",
    "ecc.errors.corrected.volatile.device_memory",
    "ecc.errors.corrected.volatile.dram",
    "ecc.errors.corrected.volatile.register_file",
    "ecc.errors.corrected.volatile.l1_cache",
    "ecc.errors.corrected.volatile.l2_cache",
    "ecc.errors.corrected.volatile.texture_memory",
    "ecc.errors.corrected.volatile.cbu",
    "ecc.errors.corrected.volatile.sram",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.corrected.aggregate.device_memory",
    "ecc.errors.corrected.aggregate.dram",
    "ecc.errors.corrected.aggregate.register_file",
    "ecc.errors.corrected.aggregate.l1_cache",
    "ecc.errors.corrected.aggregate.l2_cache",
    "ecc.errors.corrected.aggregate.texture_memory",
    "ecc.errors.corrected.aggregate.cbu",
    "ecc.errors.corrected.aggregate.sram",
    "ecc.errors.corrected.aggregate.total",
    "ecc.errors.uncorrected.volatile.device_memory",
    "ecc.errors.uncorrected.volatile.dram",
    "ecc.errors.uncorrected.volatile.register_file",
    "ecc.errors.uncorrected.volatile.l1_cache",
    "ecc.errors.uncorrected.volatile.l2_cache",
    "ecc.errors.uncorrected.volatile.texture_memory",
    "ecc.errors.uncorrected.volatile.cbu",
    "ecc.errors.uncorrected.volatile.sram",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.uncorrected.aggregate.device_memory",
    "ecc.errors.uncorrected.aggregate.dram",
    "ecc.errors.uncorrected.aggregate.register_file",
    "ecc.errors.uncorrected.aggregate.l1_cache",
    "ecc.errors.uncorrected.aggregate.l2_cache",
    "ecc.errors.uncorrected.aggregate.texture_memory",
    "ecc.errors.uncorrected.aggregate.cbu",
    "ecc.errors.uncorrected.aggregate.sram",
    "ecc.errors.uncorrected.aggregate.total",
    "retired_pages.single_bit_ecc.count",
    "retired_pages.double_bit.count",
    "retired_pages.pending",
    "temperature.gpu",
    "temperature.memory",
    "power.management",  # "power.management [W]"
    "power.draw",  # "power.draw [W]"
    "power.limit",  # "power.limit [W]"
    "enforced.power.limit",  # "enforced.power.limit [W]"
    "power.default_limit",  # "power.default_limit [W]"
    "power.min_limit",  # "power.min_limit [W]"
    "power.max_limit",  # "power.max_limit [W]"
    "clocks.current.graphics",  # "clocks.current.graphics [MHz]"
    "clocks.current.sm",  # "clocks.current.sm [MHz]"
    "clocks.current.memory",  # "clocks.current.memory [MHz]"
    "clocks.current.video",  # "clocks.current.video [MHz]"
    "clocks.applications.graphics",  # "clocks.applications.graphics [MHz]"
    "clocks.applications.memory",  # "clocks.applications.memory [MHz]"
    "clocks.default_applications.graphics",  # "clocks.default_applications.graphics [MHz]"
    "clocks.default_applications.memory",  # "clocks.default_applications.memory [MHz]"
    "clocks.max.graphics",  # "clocks.max.graphics [MHz]"
    "clocks.max.sm",  # "clocks.max.sm [MHz]"
    "clocks.max.memory",  # "clocks.max.memory [MHz]"
    "mig.mode.current",
    "mig.mode.pending",
]


def nvidia_smi_gpu(fields: List[str] = None, units=False) -> DataFrame:
    if fields is None:
        fields = NVIDIA_GPU_QUERIES
    opts = ['csv']
    if not units:
        opts.append('nounits')
    sproc = subprocess.Popen(
        ['/usr/bin/nvidia-smi', f'--query-gpu={",".join(fields)}', f'--format={",".join(opts)}'],
        stdout=subprocess.PIPE
    )
    df = read_csv(sproc.stdout, sep=', ', engine='python')
    return df


NVIDIA_COMPUTE_QUERIES = [
    "timestamp",
    "gpu_name",
    "gpu_bus_id",
    "gpu_serial",
    "gpu_uuid",
    "pid",
    "process_name",
    "used_gpu_memory",
]


def nvidia_smi_compute(fields: List[str] = None, units=False) -> DataFrame:
    if fields is None:
        fields = NVIDIA_COMPUTE_QUERIES
    opts = ['csv']
    if not units:
        opts.append('nounits')
    sproc = subprocess.Popen(
        ['/usr/bin/nvidia-smi', f'--query-compute-apps={",".join(fields)}', f'--format={",".join(opts)}'],
        stdout=subprocess.PIPE
    )
    df = read_csv(sproc.stdout, sep=', ', engine='python')
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
