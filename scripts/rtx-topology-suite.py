#!/usr/bin/env python3
"""One bounded, single-node RTX PRO 6000 experiment; no provisioning or retries.

Run inside the prepared CUDA image. Tool versions and CLI semantics are pinned to
nvbandwidth v0.9 (1aa9e818d0728c25c87e121d692d829db2239720) and nccl-tests
v2.17.9 (2535da805b34e96d1dc08be66289be1a6d57f5ad). --self-check uses no GPU.
"""

import argparse
import csv
import ctypes
import hashlib
import itertools
import json
import mmap
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import uuid
import xml.etree.ElementTree as ET


PINS = {
    "nvbandwidth": "1aa9e818d0728c25c87e121d692d829db2239720",
    "nccl-tests": "2535da805b34e96d1dc08be66289be1a6d57f5ad",
}
EXPECTED_NAME = "NVIDIA RTX PRO 6000 Blackwell Server Edition"
MESSAGE_BYTES = [1024 * 4**i for i in range(9)]  # 1 KiB through 64 MiB.
PCI_ID = re.compile(r"^[0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$")
PATH_COST = {"PIX": 1, "PXB": 2, "PHB": 3, "NODE": 4, "SYS": 5}


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def expand_ranges(value):
    result = set()
    for token in (value or "").split(","):
        if not token:
            continue
        bounds = token.split("-")
        result.update(range(int(bounds[0]), int(bounds[-1]) + 1))
    return result


def status_fields(pid="self"):
    text = read_text(f"/proc/{pid}/status") or ""
    wanted = {"Cpus_allowed_list", "Mems_allowed_list", "VmRSS", "VmLck"}
    return {key: value.strip() for line in text.splitlines() if ":" in line
            for key, value in [line.split(":", 1)] if key in wanted}


def normalize_busid(value):
    domain, bus, devfn = value.strip().lower().split(":")
    return f"{int(domain, 16):04x}:{bus}:{devfn}"


def numa_pages(line):
    return {key: int(value) for key, value in re.findall(r"\b(N\d+)=(\d+)", line)}


def numa_probe():
    """Verify an actual first-touched anonymous mapping under numactl membind."""
    size = 64 * 1024**2
    buf = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    for offset in range(0, size, mmap.PAGESIZE):
        buf[offset] = 1
    address = ctypes.addressof(ctypes.c_char.from_buffer(buf))
    start = None
    for line in (read_text("/proc/self/maps") or "").splitlines():
        lower, upper = [int(x, 16) for x in line.split()[0].split("-")]
        if lower <= address < upper:
            start = lower
            break
    rows = [line for line in (read_text("/proc/self/numa_maps") or "").splitlines()
            if start is not None and int(line.split()[0], 16) == start]
    print(json.dumps({"bytes": size, "status": status_fields(),
                      "cpu_affinity": sorted(os.sched_getaffinity(0)),
                      "mapping": rows, "pages": numa_pages(rows[0]) if rows else {}}))
    buf.close()


def cuda_probe():
    """Use the CUDA driver API to verify visible devices and directional P2P."""
    lib = ctypes.CDLL("libcuda.so.1")

    def check(result):
        if result:
            raise RuntimeError(f"CUDA driver API returned {result}")

    check(lib.cuInit(0))
    count = ctypes.c_int()
    check(lib.cuDeviceGetCount(ctypes.byref(count)))
    devices = []
    for ordinal in range(count.value):
        dev = ctypes.c_int()
        check(lib.cuDeviceGet(ctypes.byref(dev), ordinal))
        name, bus, uid = ctypes.create_string_buffer(256), ctypes.create_string_buffer(64), ctypes.create_string_buffer(16)
        check(lib.cuDeviceGetName(name, len(name), dev))
        check(lib.cuDeviceGetPCIBusId(bus, len(bus), dev))
        get_uuid = getattr(lib, "cuDeviceGetUuid_v2", lib.cuDeviceGetUuid)
        check(get_uuid(uid, dev))
        devices.append({"ordinal": ordinal, "device": dev.value, "name": name.value.decode(),
                        "bus_id": normalize_busid(bus.value.decode()),
                        "uuid": "GPU-" + str(uuid.UUID(bytes=uid.raw))})
    peers = []
    for source in devices:
        row = []
        for target in devices:
            if source == target:
                row.append(None)
            else:
                access = ctypes.c_int()
                check(lib.cuDeviceCanAccessPeer(ctypes.byref(access), source["device"], target["device"]))
                row.append(bool(access.value))
        peers.append(row)
    print(json.dumps({"devices": devices, "can_access_peer": peers,
                      "matrix_semantics": "row CUDA device may access column CUDA device"}))


def parse_topology(text):
    # Newer nvidia-smi decorates the GPU header with ANSI underline codes.
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    rows = {}
    columns = []
    for line in text.splitlines():
        tokens = line.split()
        if len(tokens) > 1 and re.fullmatch(r"GPU\d+", tokens[0]) and re.fullmatch(r"GPU\d+", tokens[1]):
            columns = [int(token[3:]) for token in tokens if re.fullmatch(r"GPU\d+", token)]
        elif tokens and columns and re.fullmatch(r"GPU\d+", tokens[0]):
            rows[int(tokens[0][3:])] = dict(zip(columns, tokens[1:]))
    return rows


def select_groups(devices, topology):
    """Choose representative near/far subsets from observed topology, never IDs."""
    def label(a, b):
        row = topology.get(devices[a]["smi_index"], {})
        index = devices[b]["smi_index"]
        return row.get(index, "UNKNOWN")

    def cost(a, b):
        value = label(a, b)
        return 0 if value.startswith("NV") else PATH_COST.get(value)

    groups = [{"id": "all8", "indices": list(range(len(devices))), "reason": "all verified visible GPUs"}]
    skipped = []
    for n in (2, 4):
        candidates = []
        for indices in itertools.combinations(range(len(devices)), n):
            values = [cost(a, b) for a, b in itertools.combinations(indices, 2)]
            if any(value is None for value in values):
                continue
            candidates.append((sum(values), max(values), indices))
        if not candidates:
            skipped.extend({"id": f"{kind}{n}", "reason": "topology labels unavailable"} for kind in ("near", "far"))
            continue
        near = min(candidates)
        far = max(candidates, key=lambda row: (row[0], row[1], tuple(-i for i in row[2])))
        for kind, chosen in (("near", near), ("far", far)):
            if kind == "far" and chosen[:2] == near[:2]:
                skipped.append({"id": f"far{n}", "reason": "no distinct near/far path-score class exists"})
                continue
            groups.append({"id": f"{kind}{n}", "indices": list(chosen[2]),
                           "path_score_sum": chosen[0], "path_score_max": chosen[1],
                           "pair_paths": [{"a": a, "b": b, "path": label(a, b)}
                                          for a, b in itertools.combinations(chosen[2], 2)],
                           "reason": "observed path-label score; not an assertion of measured speed or physical switch identity"})
    return groups, skipped


def parse_nccl(text):
    rows = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 13 or not parts[0].isdigit() or not parts[1].isdigit():
            continue
        try:
            rows.append({"size_bytes": int(parts[0]), "count": int(parts[1]), "dtype": parts[2],
                         "op": parts[3], "root": int(parts[4]),
                         "out_of_place": {"time_us": float(parts[5]), "algbw_gbs": float(parts[6]),
                                          "busbw_gbs": float(parts[7]), "wrong": int(parts[8])},
                         "in_place": {"time_us": float(parts[9]), "algbw_gbs": float(parts[10]),
                                      "busbw_gbs": float(parts[11]), "wrong": int(parts[12])}})
        except ValueError:
            continue
    footer = re.search(r"Out of bounds values\s*:\s*(\d+)\s+(\w+)", text)
    return {"rows": rows, "reported_errors": int(footer[1]) if footer else None,
            "check_passed": bool(footer and footer[1] == "0" and footer[2] == "OK" and rows
                                 and all(r[mode]["wrong"] == 0 for r in rows for mode in ("in_place", "out_of_place"))),
            "size_semantics": "nccl-tests size column; AllGather size is total receive bytes per rank, not each rank's contribution"}


class Suite:
    def __init__(self, args):
        self.args = args
        self.output = Path(args.output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "metadata.json").exists():
            raise RuntimeError("Output already has metadata.json; refusing to overwrite an earlier run")
        self.started = time.monotonic()
        self.deadline = self.started + args.budget_seconds
        self.stop_reason = None
        self.active = None
        self.data = {"schema_version": 1, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                     "budget_seconds": args.budget_seconds, "expected_gpu_count": 8,
                     "expected_gpu_name": EXPECTED_NAME, "tool_commits": PINS,
                     "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "status": "running", "steps": [], "gpu_order": "ascending normalized PCI bus ID",
                     "protocol": {"nvbandwidth_buffer_mib": 256, "nvbandwidth_samples": 3,
                                  "nvbandwidth_loop_count": 16, "nccl_messages_bytes": MESSAGE_BYTES,
                                  "latency_buffer_mib": 2, "latency_accesses_per_pair": 1000000,
                                  "latency_samples": "one sweep per ordered pair, -i ignored by pinned implementation",
                                  "nccl_iterations": 20, "nccl_warmup_iterations": 5, "nccl_check_iterations": 1},
                     "inherited_communication_env": {k: v for k, v in os.environ.items()
                                                     if k.startswith(("CUDA_", "NCCL_", "NVIDIA_"))},
                     "process_status": status_fields(), "cpu_affinity": sorted(os.sched_getaffinity(0)),
                     "notes": ["No automatic full-suite or failed-step retries.",
                               "Missing topology/NUMA access is recorded, never synthesized.",
                               "No NCCL -J output: that option serializes the full process environment."]}
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, self.interrupted)
        self.save()

    def interrupted(self, signum, _frame):
        self.stop_reason = f"signal {signum}"
        if self.active is not None:
            self.kill(self.active)
        raise InterruptedError(self.stop_reason)

    @staticmethod
    def kill(process):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    def save(self):
        self.data["elapsed_seconds"] = round(time.monotonic() - self.started, 3)
        self.data["stop_reason"] = self.stop_reason
        atomic_json(self.output / "metadata.json", self.data)

    def event(self, name, status, files):
        self.save()
        print(json.dumps({"event": "step_complete", "step": name, "status": status,
                          "files": list(dict.fromkeys([*files, "metadata.json"])),
                          "elapsed_seconds": self.data["elapsed_seconds"]}), flush=True)

    def skip(self, name, reason, context=None):
        record = {"name": name, "status": "skipped", "reason": reason, "context": context or {}}
        self.data["steps"].append(record)
        self.event(name, "skipped", [])
        return record

    def run(self, name, command, timeout=30, env=None, context=None, monitor_numa=False, benchmark=False):
        if self.stop_reason:
            return self.skip(name, self.stop_reason, context)
        remaining = self.deadline - time.monotonic() - 8
        if remaining < 3:
            self.stop_reason = "suite deadline reached"
            return self.skip(name, self.stop_reason, context)
        stdout_name, stderr_name = name + ".stdout.txt", name + ".stderr.txt"
        record = {"name": name, "command": command, "context": context or {},
                  "environment_overrides": env or {}, "stdout": stdout_name, "stderr": stderr_name,
                  "timeout_seconds": min(timeout, remaining), "status": "running"}
        self.data["steps"].append(record)
        self.save()
        began = time.monotonic()
        files = [stdout_name, stderr_name]
        snapshots = []
        link_samples = []
        next_sample = began
        try:
            with (self.output / stdout_name).open("w") as out, (self.output / stderr_name).open("w") as err:
                process = subprocess.Popen(command, stdout=out, stderr=err, env={**os.environ, **(env or {})}, start_new_session=True)
                self.active = process
                while process.poll() is None:
                    if time.monotonic() - began >= record["timeout_seconds"]:
                        self.kill(process)
                        process.wait(timeout=3)
                        record["status"] = "timeout"
                        self.stop_reason = f"subprocess timeout: {name}"
                        break
                    sample_now = time.monotonic() >= next_sample
                    if monitor_numa and sample_now:
                        rows = read_text(f"/proc/{process.pid}/numa_maps")
                        if rows:
                            # Preserve raw maps: CUDA host allocations need not follow the requested policy.
                            snapshots.append({"elapsed_seconds": round(time.monotonic() - began, 3),
                                              "status": status_fields(process.pid), "numa_maps": rows})
                    if benchmark and sample_now:
                        link_samples.append({"elapsed_seconds": round(time.monotonic() - began, 3),
                                             "links": {busid: {key: read_text(path / key) for key in
                                                       ("current_link_speed", "current_link_width")}
                                                       for busid, path in getattr(self, "pci_paths", {}).items()}})
                    if sample_now:
                        next_sample = time.monotonic() + 0.5
                    time.sleep(0.1 if monitor_numa else 0.15)
                record["returncode"] = process.returncode
                if record["status"] == "running":
                    record["status"] = "ok" if process.returncode == 0 else "failed"
                if benchmark and record["status"] == "failed":
                    self.stop_reason = f"benchmark failed: {name}; no subsequent GPU tests"
        except OSError as exc:
            record.update(status="failed", error=str(exc))
            if benchmark:
                self.stop_reason = f"benchmark could not start: {name}"
        finally:
            self.active = None
            record["elapsed_seconds"] = round(time.monotonic() - began, 3)
            if snapshots:
                fname = name + ".numa-samples.json"
                atomic_json(self.output / fname, snapshots)
                record["numa_samples"] = fname
                files.append(fname)
            if link_samples:
                fname = name + ".pcie-link-samples.json"
                atomic_json(self.output / fname, link_samples)
                record["pcie_link_samples"] = fname
                files.append(fname)
            self.event(name, record["status"], files)
        return record

    def contents(self, record):
        return read_text(self.output / record["stdout"]) or "" if "stdout" in record else ""

    def parsed(self, name, result, record):
        path = name + ".json"
        atomic_json(self.output / path, result)
        record["parsed"] = path
        self.event(name + "-parsed", record["status"], [path])

    def inventory(self):
        fields = "index,name,pci.bus_id,uuid,driver_version,memory.total,memory.free"
        raw = self.run("inventory", ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if raw["status"] != "ok":
            raise RuntimeError("nvidia-smi inventory failed")
        devices = []
        for row in csv.reader(self.contents(raw).splitlines(), skipinitialspace=True):
            if len(row) != 7:
                raise RuntimeError("Unexpected GPU inventory columns")
            devices.append({"smi_index": int(row[0]), "name": row[1].strip(), "bus_id": normalize_busid(row[2]),
                            "uuid": row[3].strip(), "driver_version": row[4].strip(),
                            "memory_total_mib": int(row[5]), "memory_free_mib": int(row[6])})
        devices.sort(key=lambda item: item["bus_id"])
        self.data["devices"] = devices
        if len(devices) != 8 or any(item["name"] != EXPECTED_NAME for item in devices):
            raise RuntimeError("Allocation does not match exactly eight NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs")
        if len({item["bus_id"] for item in devices}) != 8 or len({item["uuid"] for item in devices}) != 8:
            raise RuntimeError("Device IDs are not unique")
        if any(item["memory_free_mib"] < 8192 for item in devices):
            raise RuntimeError("Less than 8 GiB free on a target GPU; stopping without resizing tests")
        self.gpus = devices
        self.gpu_env = {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(d["uuid"] for d in devices)}
        probe = self.run("cuda-driver-probe", [sys.executable, __file__, "--cuda-probe"], env=self.gpu_env)
        if probe["status"] != "ok":
            raise RuntimeError("CUDA driver probe failed")
        self.cuda = json.loads(self.contents(probe))
        if [(d["bus_id"], d["uuid"], d["name"]) for d in self.cuda["devices"]] != [(d["bus_id"], d["uuid"], d["name"]) for d in devices]:
            raise RuntimeError("CUDA and nvidia-smi device mappings disagree")
        self.parsed("cuda-driver-probe", self.cuda, probe)
        commands = {
            "gpu-list": ["nvidia-smi", "-L"],
            "gpu-state": ["nvidia-smi", "-q"],
            "topology": ["nvidia-smi", "topo", "-m"],
            "p2p-read": ["nvidia-smi", "topo", "-p2p", "r"],
            "p2p-write": ["nvidia-smi", "topo", "-p2p", "w"],
            "pci-tree": ["lspci", "-D", "-t"],
            "pci-list": ["lspci", "-D", "-nn"],
            "numa-hardware": ["numactl", "--hardware"],
            "cpu-topology": ["lscpu", "--json"],
            "cuda-toolkit": ["nvcc", "--version"],
            "nvbandwidth-tests": [self.args.nvbandwidth, "-l"],
        }
        for name, command in commands.items():
            record = self.run(name, command, env=self.gpu_env)
            if name == "topology":
                self.topology = parse_topology(self.contents(record))
            if name == "nvbandwidth-tests":
                self.test_list = self.contents(record)
        self.sysfs()
        inherited = os.environ.get("NCCL_TOPO_FILE")
        if inherited:
            value = read_text(inherited)
            self.data["inherited_topology_file_readable"] = value is not None
            if value is not None:
                (self.output / "inherited-nccl-topology.xml").write_text(value)
                self.event("inherited-nccl-topology", "observed", ["inherited-nccl-topology.xml"])
        for path in ("/etc/nccl.conf", "/var/run/nvidia-topologyd/virtualTopology.xml"):
            value = read_text(path)
            if value is not None:
                filename = "observed-" + Path(path).name
                (self.output / filename).write_text(value)
                self.event(filename, "observed", [filename])

    def sysfs(self):
        nodes = {}
        for node in Path("/sys/devices/system/node").glob("node[0-9]*"):
            nodes[int(node.name[4:])] = {"cpulist": read_text(node / "cpulist"), "distance": read_text(node / "distance"),
                                         "meminfo": read_text(node / "meminfo")}
        self.nodes = nodes
        paths = {}
        for device in self.gpus:
            root = Path("/sys/bus/pci/devices") / device["bus_id"]
            device["sysfs_visible"] = root.exists()
            device["sysfs_path"] = str(root.resolve()) if root.exists() else None
            device["numa_node"] = int(read_text(root / "numa_node") or -1)
            device["local_cpulist"] = read_text(root / "local_cpulist")
            device["pci_ancestors"] = [part for part in root.resolve().parts if PCI_ID.fullmatch(part)] if root.exists() else []
            for busid in device["pci_ancestors"]:
                paths[busid] = Path("/sys/bus/pci/devices") / busid
        for device in Path("/sys/bus/pci/devices").glob("*"):
            if (read_text(device / "class") or "").startswith("0x02"):
                paths[device.name] = device
        self.pci_paths = paths
        info = {"numa_nodes": nodes, "pci_devices": {}, "iommu_groups_visible": Path("/sys/kernel/iommu_groups").exists()}
        for busid, device in paths.items():
            info["pci_devices"][busid] = {key: read_text(device / key) for key in
                ("vendor", "device", "class", "numa_node", "local_cpulist", "current_link_speed", "current_link_width", "max_link_speed", "max_link_width")}
            info["pci_devices"][busid]["resolved_path"] = str(device.resolve())
        atomic_json(self.output / "sysfs.json", info)
        self.event("sysfs", "observed", ["sysfs.json"])
        # One process captures every path device's LnkCap/LnkSta and ACS without changing config.
        self.run("pci-link-details", ["lspci", "-D", "-vv", "-nn"], timeout=20)

    def visible(self, indices):
        return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": ",".join(self.gpus[i]["uuid"] for i in indices)}

    def bandwidth(self, name, tests, indices=None, prefix=None, monitor=False, context=None, timeout=65):
        indices = list(range(8)) if indices is None else indices
        context = {**(context or {}), "gpu_indices": indices, "bus_ids": [self.gpus[i]["bus_id"] for i in indices],
                   "verification_enabled": True, "tests": tests}
        if any(test not in self.test_list for test in tests):
            return self.skip(name, "test not exposed by pinned nvbandwidth binary", context)
        command = [*(prefix or []), self.args.nvbandwidth, "-j", "-d", "-b", "256", "-i", "3", "-t", *tests]
        record = self.run(name, command, timeout=timeout, env=self.visible(indices), context=context, monitor_numa=monitor, benchmark=True)
        if record["status"] != "ok":
            return record
        try:
            parsed = json.loads(self.contents(record))
            root = parsed["nvbandwidth"]
            results = root.get("testcases", [])
            passed = {row["name"] for row in results if row.get("status") == "Passed"}
            waived = {row["name"] for row in results if row.get("status") == "Waived"}
            if root.get("error") or set(tests) - passed - waived:
                raise ValueError("missing or failed nvbandwidth test result")
            parsed["interpretation"] = {
                "gpu_indices_in_tool_order": indices,
                "write_ce_direction": "row=source,column=destination, per v0.9 DeviceToDeviceWriteCE::run; its printed arrow is reversed",
                "latency_direction": "row=requesting GPU,column=remote memory owner; pointer-chase ns, not transfer latency",
                "host_to_all_semantics": "per measured GPU throughput with traffic on other visible GPUs; not aggregate throughput",
                "diagonal_and_na": "not measured; never convert to zero bandwidth",
                "waived_tests": sorted(waived),
                "latency_buffer_mib": 2 if "device_to_device_latency_sm" in tests else None,
            }
            self.parsed(name, parsed, record)
        except (ValueError, KeyError) as exc:
            record.update(status="invalid_result", error=str(exc))
            self.stop_reason = f"invalid benchmark output: {name}"
            self.event(name, record["status"], [])
        return record

    def nccl(self, group, operation):
        name = f"nccl-{group['id']}-{operation}"
        indices = group["indices"]
        xmlname, logname = name + ".topology.xml", name + ".nccl.log"
        env = {**self.visible(indices), "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,P2P,SHM,NET,ENV",
               "NCCL_DEBUG_FILE": str(self.output / logname), "NCCL_TOPO_DUMP_FILE": str(self.output / xmlname)}
        if self.args.container_compat:
            env["NCCL_CUMEM_HOST_ENABLE"] = "0"
        env.update(getattr(self, "transport_env", {}))
        command = [str(Path(self.args.nccl_tests) / f"{operation}_perf"), "-g", str(len(indices)),
                   "-b", "1K", "-e", "64M", "-f", "4", "-d", "float", "-n", "20", "-w", "5", "-c", "1", "-T", "25"]
        record = self.run(name, command, timeout=40, env=env, context=group, benchmark=True)
        files = [filename for filename in (xmlname, logname) if (self.output / filename).exists()]
        record["artifacts"] = files
        if record["status"] == "ok":
            result = parse_nccl(self.contents(record))
            if not result["check_passed"] or [row["size_bytes"] for row in result["rows"]] != MESSAGE_BYTES:
                record["status"] = "invalid_result"
                self.stop_reason = f"NCCL correctness or message matrix incomplete: {name}"
            result["topology_xml"] = xmlname if xmlname in files else None
            if xmlname in files:
                try:
                    root = ET.parse(self.output / xmlname).getroot()
                    result["xml_root"] = root.tag
                    result["xml_gpu_count"] = len(root.findall(".//gpu"))
                except ET.ParseError as exc:
                    result["xml_error"] = str(exc)
                    record["incomplete_artifact"] = True
            else:
                record["missing_topology_xml"] = True
                record["incomplete_artifact"] = True
            self.parsed(name, result, record)
        self.event(name + "-artifacts", record["status"], files)
        return record

    def first_touch_tests(self):
        allowed = set(self.data["cpu_affinity"])
        nodes = {n for n, info in self.nodes.items() if expand_ranges(info["cpulist"]) & allowed}
        target = next((i for i, d in enumerate(self.gpus) if d["numa_node"] in nodes), None)
        if target is None or len(nodes) < 2:
            self.skip("numa-first-touch", "No comparable NUMA nodes visible")
            return
        local = self.gpus[target]["numa_node"]
        remote = next(n for n in sorted(nodes) if n != local)
        for label, node in (("local", local), ("remote", remote)):
            record = self.run("numa-first-touch-" + label,
                [sys.executable, str(Path(__file__).with_name("numa-first-touch.py")),
                 "--node", str(node), "--submit-node", str(local)], timeout=45,
                env=self.visible([target]), context={"memory_node": node, "gpu": target,
                "submission_node": local, "method": "verified first-touch, not strict membind"})
            if record["status"] == "ok":
                result = json.loads(self.contents(record))
                if not result.get("correctness_passed") or result.get("status") != "complete":
                    record["status"] = "invalid_result"
                self.parsed(record["name"], result, record)

    def recover_collective(self):
        reason = self.stop_reason
        if not reason or self.deadline - time.monotonic() < 30:
            return
        self.stop_reason = None
        probe = self.run("post-failure-health-" + str(len(self.data["steps"])),
            [sys.executable, __file__, "--cuda-health-probe"], timeout=15, env=self.gpu_env,
            context={"preceding_failure": reason})
        if probe["status"] != "ok":
            self.stop_reason = "CUDA health check failed after " + reason
        else:
            self.data.setdefault("recovered_failures", []).append(reason)
        self.save()

    def numa_tests(self):
        allowed_cpus = set(self.data["cpu_affinity"])
        allowed_mems = expand_ranges(self.data["process_status"].get("Mems_allowed_list"))
        candidates = {node: sorted(expand_ranges(info["cpulist"]) & allowed_cpus)
                      for node, info in self.nodes.items() if node in allowed_mems}
        candidates = {node: cpus for node, cpus in candidates.items() if cpus}
        # Hold the GPU fixed and vary CPU+memory placement; compare only actual accessible NUMA nodes.
        target = next((i for i, gpu in enumerate(self.gpus) if gpu["numa_node"] in candidates), None)
        if target is None or len(candidates) < 2:
            self.skip("numa-local-remote", "need a GPU's known local NUMA node and at least one other CPU+memory-bindable node",
                      {"candidate_nodes": candidates, "allowed_mems": sorted(allowed_mems)})
            return
        local = self.gpus[target]["numa_node"]
        remote = next(node for node in sorted(candidates) if node != local)
        for label, node in (("local", local), ("remote", remote)):
            prefix = ["numactl", "--physcpubind=" + ",".join(str(cpu) for cpu in candidates[node]), f"--membind={node}"]
            context = {"target_gpu": target, "target_bus_id": self.gpus[target]["bus_id"],
                       "gpu_numa_node": local, "bound_cpu_memory_node": node, "bound_cpus": candidates[node]}
            probe = self.run(f"numa-{label}-binding-probe", [*prefix, sys.executable, __file__, "--numa-probe"], context=context, timeout=10)
            if probe["status"] != "ok":
                self.skip(f"numa-{label}-transfers", "strict CPU/memory binding failed; no unbound fallback", context)
                continue
            try:
                observed = json.loads(self.contents(probe))
                pages = observed["pages"]
                if not pages or set(pages) != {f"N{node}"} or pages[f"N{node}"] < 8192:
                    raise ValueError("first-touch allocation not verified exclusively on requested node")
            except (ValueError, KeyError) as exc:
                self.skip(f"numa-{label}-transfers", str(exc), context)
                continue
            self.bandwidth(f"numa-{label}-transfers", ["host_to_device_memcpy_ce", "device_to_host_memcpy_ce"],
                           indices=[target], prefix=prefix, monitor=True, context=context, timeout=25)
        self.data["numa_interpretation"] = "numactl probe verifies first-touch; benchmark numa_maps must independently support host-buffer placement before attributing differences to NUMA"

    def same_node_comparison(self, groups, skipped):
        self.data["scope"] = "same-node default NCCL versus disabled P2P; full topology and transfer controls"
        expected = {"near2", "far2", "near4", "far4", "all8"}
        if skipped or {g["id"] for g in groups} != expected:
            raise RuntimeError("Complete near/far grouping unavailable; refusing an incomplete comparison")
        if not all(value for row in self.cuda["can_access_peer"] for value in row if value is not None):
            raise RuntimeError("Not all 56 directed GPU pairs support CUDA peer access")
        pair = self.bandwidth("gpu-pair-bandwidth", ["device_to_device_memcpy_write_ce"], timeout=80)
        if pair["status"] != "ok":
            raise RuntimeError("P2P transfer preflight failed; do not continue collectives")
        ordered = sorted(groups, key=lambda g: ["near2", "far2", "near4", "far4", "all8"].index(g["id"]))
        # Every case runs once. A failed default case stops the comparison rather
        # than silently replacing the requested native matrix with a workaround.
        for mode, environment in (("default", {}), ("no_p2p", {"NCCL_P2P_DISABLE": "1"})):
            self.transport_env = environment
            for group in ordered:
                for operation in ("all_reduce", "all_gather"):
                    case = {**group, "id": mode + "_" + group["id"],
                            "base_group": group["id"], "transport_mode": mode}
                    record = self.nccl(case, operation)
                    if record["status"] != "ok" or record.get("incomplete_artifact"):
                        raise RuntimeError("Same-node matrix failed at " + record["name"])
        steps = [s for s in self.data["steps"] if s["name"].startswith("nccl-")]
        self.data["collective_matrix_complete"] = len(steps) == 20 and all(s["status"] == "ok" for s in steps)
        self.bandwidth("host-serial-transfers", ["host_to_device_memcpy_ce", "device_to_host_memcpy_ce"], timeout=40)
        self.first_touch_tests()
        self.bandwidth("host-concurrent-transfers", ["host_to_all_memcpy_ce", "all_to_host_memcpy_ce"], timeout=55)
        self.bandwidth("gpu-concurrent-fan-in", ["all_to_one_write_ce"], timeout=50)
        self.bandwidth("gpu-remote-memory-latency", ["device_to_device_latency_sm"], timeout=105)
        self.run("gpu-state-after", ["nvidia-smi", "-q"], timeout=15)
        self.data["status"] = "partial" if self.stop_reason or any(s["status"] not in ("ok", "observed") or s.get("incomplete_artifact") for s in self.data["steps"]) else "complete"
        return 0 if self.data["status"] == "complete" else 2

    def execute(self):
        try:
            self.inventory()
            groups, skipped = select_groups(self.gpus, self.topology)
            self.data["groups"], self.data["unavailable_groups"] = groups, skipped
            if self.args.same_node_comparison:
                return self.same_node_comparison(groups, skipped)
            for missing in skipped:
                for operation in ("all_reduce", "all_gather"):
                    self.skip(f"nccl-{missing['id']}-{operation}", missing["reason"])
            if self.args.collectives_only:
                self.data["scope"] = "NUMA first-touch and near/far collectives supplement"
                self.data["nccl_cumem_host_enable"] = "0" if self.args.container_compat else "default"
                if self.args.transport_diagnosis:
                    target = next(g for g in groups if g["id"] == "near2")
                    selected = None
                    for label, env in [("default", {}), ("no_direct_buffer", {"NCCL_P2P_DIRECT_DISABLE": "1"}),
                                       ("no_p2p", {"NCCL_P2P_DISABLE": "1"})]:
                        self.transport_env = env
                        case = {**target, "id": "diagnostic_" + label + "_near2", "transport_environment": env}
                        record = self.nccl(case, "all_reduce")
                        if record["status"] == "ok" and not record.get("incomplete_artifact"):
                            selected = label
                            break
                        self.recover_collective()
                        if self.stop_reason:
                            break
                    self.data["selected_transport"] = selected
                    self.data["selected_transport_environment"] = getattr(self, "transport_env", {})
                    if selected is None:
                        self.stop_reason = self.stop_reason or "No tested NCCL transport configuration passed"
                else:
                    self.first_touch_tests()
                # Complete smaller groups before the potentially problematic all8 control.
                for group in [*groups[1:], groups[0]]:
                    for operation in ("all_reduce", "all_gather"):
                        self.nccl(group, operation)
                        if self.stop_reason and self.args.container_compat:
                            self.recover_collective()
                main_steps = [s for s in self.data["steps"] if s["name"].startswith("nccl-") and not s["name"].startswith("nccl-diagnostic_")]
                self.data["collective_matrix_complete"] = len(main_steps) == 2 * len(groups) and all(s["status"] == "ok" and not s.get("incomplete_artifact") for s in main_steps)
                self.run("gpu-state-after", ["nvidia-smi", "-q"], timeout=15)
                self.data["status"] = "partial" if self.stop_reason or any(s["status"] in ("failed", "timeout", "invalid_result") or s.get("incomplete_artifact") for s in self.data["steps"]) else "complete"
                return 0 if self.data["status"] == "complete" else 2
            for operation in ("all_reduce", "all_gather"):
                self.nccl(groups[0], operation)
            p2p = any(value for row in self.cuda["can_access_peer"] for value in row)
            if p2p:
                self.bandwidth("gpu-pair-bandwidth", ["device_to_device_memcpy_write_ce"], timeout=80)
            else:
                for name in ("gpu-pair-bandwidth", "gpu-remote-memory-latency"):
                    self.skip(name, "CUDA driver reports no accessible peer pairs; NCCL may still use other transports")
            self.bandwidth("host-serial-transfers", ["host_to_device_memcpy_ce", "device_to_host_memcpy_ce"], timeout=40)
            self.numa_tests()
            self.bandwidth("host-concurrent-transfers", ["host_to_all_memcpy_ce", "all_to_host_memcpy_ce"], timeout=55,
                           context={"comparison": "host-serial-transfers", "cpu_memory_binding": "container default; nvbandwidth affinity disabled consistently"})
            for group in groups[1:]:
                for operation in ("all_reduce", "all_gather"):
                    self.nccl(group, operation)
            if p2p:
                self.bandwidth("gpu-concurrent-fan-in", ["all_to_one_write_ce"], timeout=50,
                               context={"semantics": "aggregate fan-in to each target over accessible peer paths; not per-link bandwidth"})
            else:
                self.skip("gpu-concurrent-fan-in", "CUDA peer access unavailable")
            # Pointer chasing has a fixed 2 MiB buffer / 1M accesses in v0.9.
            # It is last so a slow remote-memory path cannot discard the core matrix.
            if p2p:
                self.bandwidth("gpu-remote-memory-latency", ["device_to_device_latency_sm"], timeout=105,
                               context={"buffer_mib_actual": 2, "pointer_accesses_per_pair": 1000000,
                                        "samples_note": "v0.9 latency uses one 1M-access sweep per accessible ordered pair; -i is not used"})
            self.run("gpu-state-after", ["nvidia-smi", "-q"], timeout=15)
            self.data["status"] = "partial" if self.stop_reason or any(s["status"] in ("failed", "timeout", "invalid_result") or s.get("incomplete_artifact") for s in self.data["steps"]) else "complete"
        except (Exception, KeyboardInterrupt) as exc:
            self.data.update(status="aborted", error=str(exc), traceback=traceback.format_exc())
            self.stop_reason = self.stop_reason or str(exc)
            for name in ("gpu-pair-bandwidth", "gpu-remote-memory-latency", "host-serial-transfers", "numa-local-remote",
                         "host-concurrent-transfers", "nccl-all8-all_reduce", "nccl-all8-all_gather", "gpu-concurrent-fan-in"):
                if not any(row["name"] == name for row in self.data["steps"]):
                    self.skip(name, "allocation/preflight aborted: " + self.stop_reason)
        finally:
            self.data["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            self.event("suite", self.data["status"], [str(p.relative_to(self.output)) for p in self.output.iterdir() if p.is_file()])
        return 0 if self.data["status"] == "complete" else 2


def self_check():
    """Pure parsing/grouping checks; no nvidia-smi, CUDA, subprocesses or network."""
    assert expand_ranges("0-2,7,9-10") == {0, 1, 2, 7, 9, 10}
    assert normalize_busid("00000000:AB:00.0") == "0000:ab:00.0"
    devices = [{"smi_index": i} for i in range(8)]
    topo = {i: {j: "X" if i == j else "PIX" if i // 4 == j // 4 else "SYS" for j in range(8)} for i in range(8)}
    groups, skipped = select_groups(devices, topo)
    assert not skipped
    assert next(g for g in groups if g["id"] == "near4")["indices"] == [0, 1, 2, 3]
    assert next(g for g in groups if g["id"] == "far4")["indices"] == [0, 1, 4, 5]
    assert next(g for g in groups if g["id"] == "far2")["indices"] == [0, 4]
    homogeneous = {i: {j: "X" if i == j else "PHB" for j in range(8)} for i in range(8)}
    assert {row["id"] for row in select_groups(devices, homogeneous)[1]} == {"far2", "far4"}
    parsed = parse_nccl("1024 256 float sum -1 12.3 0.1 0.2 0 11.2 0.1 0.2 0\n# Out of bounds values : 0 OK")
    assert parsed["check_passed"] and parsed["rows"][0]["size_bytes"] == 1024
    assert not parse_nccl("# Out of bounds values : 0 OK")["check_passed"]
    assert numa_pages("0000 bind:2 anon=10 N2=10 kernelpagesize_kB=4") == {"N2": 10}
    assert parse_topology(" GPU1 GPU3 CPU Affinity\nGPU1 X SYS 0-3\nGPU3 SYS X 4-7")[1][3] == "SYS"
    assert parse_topology("\t\x1b[4mGPU0\tGPU1\tCPU Affinity\x1b[0m\nGPU0 X PIX 0-3\nGPU1 PIX X 0-3")[0][1] == "PIX"
    assert len(MESSAGE_BYTES) == 9 and MESSAGE_BYTES[-1] == 64 * 1024**2
    print(json.dumps({"self_check": "passed", "target_gpu_used": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output")
    parser.add_argument("--budget-seconds", type=int, default=660)
    parser.add_argument("--nvbandwidth", default="/opt/nvbandwidth/build/nvbandwidth")
    parser.add_argument("--nccl-tests", default="/opt/nccl-tests/build")
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--collectives-only", action="store_true", help="Supplement near/far 2/4-GPU collectives with a fresh all8 control")
    parser.add_argument("--container-compat", action="store_true")
    parser.add_argument("--transport-diagnosis", action="store_true")
    parser.add_argument("--same-node-comparison", action="store_true")
    parser.add_argument("--cuda-health-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cuda-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--numa-probe", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    if args.cuda_health_probe:
        cuda_probe()
        lib = ctypes.CDLL("libcudart.so.12")
        for i in range(8):
            pointer = ctypes.c_void_p()
            for code in (lib.cudaSetDevice(i), lib.cudaMalloc(ctypes.byref(pointer), ctypes.c_size_t(4096))):
                if code: raise RuntimeError(f"CUDA health failed: {code}")
            try:
                for code in (lib.cudaMemset(pointer, 0, ctypes.c_size_t(4096)), lib.cudaDeviceSynchronize()):
                    if code: raise RuntimeError(f"CUDA health failed: {code}")
            finally:
                lib.cudaFree(pointer)
        return 0
    if args.cuda_probe:
        cuda_probe()
        return 0
    if args.numa_probe:
        numa_probe()
        return 0
    if not args.output or not 60 <= args.budget_seconds <= 660:
        parser.error("--output is required; --budget-seconds must be between 60 and the planned 660 second limit")
    return Suite(args).execute()


if __name__ == "__main__":
    sys.exit(main())
