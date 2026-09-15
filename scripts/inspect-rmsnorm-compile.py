"""Preserve selected launch configurations and PTX for the generated modules."""

import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torch._inductor.runtime.runtime_utils import cache_dir

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "public/measurements/rmsnorm-compile"
record = json.loads((ROOT / "content/data/rmsnorm-compile.json").read_text())
torch.set_num_threads(1)
details = []
with torch.no_grad():
    for row in record["measurements"]:
        M, N = row["M"], row["N"]
        x = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        gamma = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        g = torch.randn_like(x)
        r = torch.ones(M, 1, device="cuda", dtype=torch.float32)
        for direction in ("forward", "backward"):
            path = ROOT / row[direction]["code"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row[direction]["code_sha256"]
            spec = importlib.util.spec_from_file_location("generated_rmsnorm", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.call([x, gamma] if direction == "forward" else [x, gamma, g, r])
            torch.cuda.synchronize()
            kernels = []
            for name, kernel in vars(module).items():
                if not name.startswith("triton_") or not hasattr(kernel, "launchers"):
                    continue
                assert len(kernel.launchers) == 1
                launcher = kernel.launchers[0]
                cache = Path(cache_dir()) / "triton/0" / launcher.cache_hash
                ptx = cache / f"{name}.ptx"
                target = path.parent / f"{name}.ptx"
                target.write_bytes(ptx.read_bytes())
                kernels.append({
                    "name": name, "tile": launcher.config.kwargs,
                    "num_warps": launcher.config.num_warps,
                    "num_stages": launcher.config.num_stages,
                    "registers_per_thread": launcher.n_regs,
                    "spills": launcher.n_spills, "shared_bytes": launcher.shared,
                    "cache_hash": launcher.cache_hash,
                    "ptx": str(target.relative_to(ROOT)),
                    "ptx_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                })
            details.append({"M": M, "N": N, "direction": direction, "kernels": kernels})
            print(M, N, direction, [(k["tile"], k["num_warps"]) for k in kernels], flush=True)
(ARTIFACTS / "kernel-details.json").write_text(json.dumps(details, indent=2) + "\n")
