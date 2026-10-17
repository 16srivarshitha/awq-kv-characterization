import re, socket, subprocess, torch


def gpu_tag():
    name = torch.cuda.get_device_name(0).lower()
    name = re.sub(r"nvidia|geforce", "", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    return f"{name}_{socket.gethostname().split('.')[0]}"


def check_arch():
    cc = torch.cuda.get_device_capability(0)
    arch = f"sm_{cc[0]}{cc[1]}"
    if arch not in torch.cuda.get_arch_list():
        raise SystemExit(
            f"This torch build ({torch.__version__}, CUDA {torch.version.cuda}) has no kernels for {arch}.\n"
            f"Compiled for: {torch.cuda.get_arch_list()}\n"
            f"Blackwell (sm_120) needs torch >= 2.7 built for CUDA 12.8.")
    return arch


def _f(x):
    try:
        return float(x)
    except Exception:
        return None


def smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
        v = [x.strip() for x in out.strip().splitlines()[0].split(",")]
        return dict(sm_mhz=_f(v[0]), mem_mhz=_f(v[1]), power_w=_f(v[2]), temp_c=_f(v[3]))
    except Exception:
        return {}


def driver_version():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True, timeout=10)
        return out.strip().splitlines()[0]
    except Exception:
        return None


def time_cuda(fn, warmup=5, iters=20):
    """Median GPU time in ms of fn() measured with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2]