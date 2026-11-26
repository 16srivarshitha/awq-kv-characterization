import argparse, csv, math
import torch
from cold_loadonly import time_cold, l2_bytes
from src.simt_attn import _simt_split_kv_stage2_kernel
import src.scale_free_int8_attn as sf

H_Q, H_KV, D, G = 24, 8, 128, 3
KERNELS = {"base": sf._int32_packed_fused_sk_vtok_stage1_kernel}
if hasattr(sf, "_int32_packed_fused_sk_vtok_hoist_stage1_kernel"):
    KERNELS["hoist"] = sf._int32_packed_fused_sk_vtok_hoist_stage1_kernel


def make_sets(B, S, nbuf, dev):
    sets = []
    for _ in range(nbuf):
        kb = torch.randint(-127, 128, (B, S, H_KV, D), dtype=torch.int8, device=dev)
        vb = torch.randint(-127, 128, (B, S, H_KV, D), dtype=torch.int8, device=dev)
        svt = (torch.rand(B, S, H_KV, device=dev) * 0.02 + 0.001).to(torch.float16)
        sets.append((kb, vb, svt, kb.view(torch.int32), vb.view(torch.int32)))
    return sets


def make_ws(B, sp, dev):
    mo = torch.empty((B, H_KV, sp, G, D), device=dev, dtype=torch.float32)
    ml = torch.empty((B, H_KV, sp, G), device=dev, dtype=torch.float32)
    ms = torch.empty_like(ml)
    fo = torch.empty((B, H_Q, D), device=dev, dtype=torch.float16)
    return mo, ml, ms, fo


def launch_triton(kern, q_rot, s_k, st, ws, S, sp, run_stage2):
    _, _, svt, k32, v32 = st
    mo, ml, ms, fo = ws
    B = q_rot.shape[0]
    kern[(B * H_KV * sp,)](
        q_rot, s_k, k32, v32, svt, mo, ml, ms,
        *q_rot.stride(), s_k.stride(0), s_k.stride(1),
        *k32.stride(), *v32.stride(), *svt.stride(),
        *mo.stride(), *ml.stride(),
        S, 1.0 / math.sqrt(D),
        NUM_SPLITS=sp, SPLIT_TOKENS=math.ceil(S / sp), BLOCK_K=16, BLOCK_D32=32,
        num_warps=1, num_stages=1)
    if run_stage2:
        _simt_split_kv_stage2_kernel[(B * H_Q,)](
            mo, ml, ms, fo,
            *mo.stride(), *ml.stride(), fo.stride(0), fo.stride(1), fo.stride(2),
            NUM_SPLITS=sp, BLOCK_D=128, num_warps=2)


def cuda_launch(i, sets, B, sp, mode):
    """TODO: call your existing launch_dp4a_raw on buffer set sets[i] = (k_int8, v_int8, s_v_tok, ...).
    Preallocate q_int8_buf, q_scale_buf and the mid_* buffers OUTSIDE this function (sized for sp).
    mode bit 1 = skip stage 2, bit 2 = skip the Q prescale kernel (needs the small .cu patch)."""
    raise NotImplementedError("fill in your launch_dp4a_raw call")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seqlen", type=int, default=8192)
    ap.add_argument("--peak-gbs", type=float, required=True)
    ap.add_argument("--l2-mb", type=float, default=0)
    ap.add_argument("--splits", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--cuda-splits", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default="attn_cold_runs.csv")
    a = ap.parse_args()

    dev, S = "cuda", a.seqlen
    target = max(2 * l2_bytes(a.l2_mb), 256 * 2**20)
    print(f"GPU {torch.cuda.get_device_name(0)} | cycling >= {target/2**20:.0f} MB per timing")
    rows = []
    for B in a.batches:
        kv_bytes = 2 * B * S * H_KV * D
        set_bytes = kv_bytes + B * S * H_KV * 2
        nbuf = max(2, math.ceil(target / set_bytes))
        sets = make_sets(B, S, nbuf, dev)
        q = torch.randn(B, H_Q, D, device=dev, dtype=torch.float16)
        q_rot = q.view(B, H_KV, G, D).contiguous()
        s_k = torch.rand(H_KV, D, device=dev, dtype=torch.float32) * 0.05 + 0.01
        floor = kv_bytes / (a.peak_gbs * 1e9) * 1e6
        print(f"\n=== B={B}: {kv_bytes/2**20:.1f} MiB KV, {nbuf} buffer sets, "
              f"load-only floor ~{floor:.1f} us ===")

        if a.check and len(KERNELS) > 1:
            sp = 64
            outs = {}
            for name, kern in KERNELS.items():
                ws = make_ws(B, sp, dev)
                launch_triton(kern, q_rot, s_k, sets[0], ws, S, sp, True)
                torch.cuda.synchronize()
                outs[name] = ws[3].clone()
            print("bit-identical output (base vs hoist):",
                  torch.equal(outs["base"], outs["hoist"]))

        for name, kern in KERNELS.items():
            print(f"-- triton int8 [{name}]  (split: full / stage1-only, us)")
            for sp in a.splits:
                ws = make_ws(B, sp, dev)
                res = {}
                for run2 in (True, False):
                    res[run2] = time_cold(
                        lambda i, kern=kern, ws=ws, sp=sp, run2=run2: launch_triton(
                            kern, q_rot, s_k, sets[i], ws, S, sp, run2), nbuf)
                    med, mn = res[run2]
                    rows.append([B, f"triton_{name}", sp, "full" if run2 else "stage1", med, mn,
                                 kv_bytes / (med * 1e-6) / 1e9, floor])
                print(f"   sp={sp:4d}: {res[True][0]:7.2f} / {res[False][0]:7.2f}   "
                      f"(stage2+gap ~{res[True][0]-res[False][0]:.2f})")

        if a.cuda:
            print("-- cuda dp4a  (split: full / no-stage2 / stage1-only, us)")
            for sp in a.cuda_splits:
                t = {}
                for mode in (0, 1, 3):
                    t[mode] = time_cold(
                        lambda i, mode=mode, sp=sp: cuda_launch(i, sets, B, sp, mode), nbuf)[0]
                    rows.append([B, "cuda_dp4a", sp, f"mode{mode}", t[mode], t[mode],
                                 kv_bytes / (t[mode] * 1e-6) / 1e9, floor])
                print(f"   sp={sp:4d}: {t[0]:7.2f} / {t[1]:7.2f} / {t[3]:7.2f}")
        del sets
        torch.cuda.empty_cache()

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B", "kernel", "splits", "variant", "median_us", "min_us", "kv_GBps", "floor_us"])
        w.writerows(rows)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
