"""Convert saved experimental results to the article's data; never invent missing values."""

import argparse
import json
from pathlib import Path
import re


def summarize(folder):
    def read(name):
        path = folder / name
        return json.loads(path.read_text()) if path.exists() else None

    metadata = read("metadata.json")
    if not metadata or not metadata.get("devices"):
        raise ValueError("No verified GPU inventory in this run")
    devices = metadata["devices"]
    topology = {}
    columns = []
    topo_path = folder / "topology.stdout.txt"
    for line in topo_path.read_text().splitlines() if topo_path.exists() else []:
        parts = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).split()
        if len(parts) > 1 and re.fullmatch(r"GPU\d+", parts[0]) and re.fullmatch(r"GPU\d+", parts[1]):
            columns = [int(part[3:]) for part in parts if re.fullmatch(r"GPU\d+", part)]
        elif parts and columns and re.fullmatch(r"GPU\d+", parts[0]):
            topology[int(parts[0][3:])] = dict(zip(columns, parts[1:]))
    paths = [[topology.get(a["smi_index"], {}).get(b["smi_index"])
              for b in devices] for a in devices]

    def matrix(filename, test):
        result = read(filename)
        for row in result.get("nvbandwidth", {}).get("testcases", []) if result else []:
            if row["name"] == test and row["status"] == "Passed":
                return [[None if value == "N/A" else float(value) for value in values]
                        for values in row["bandwidth_matrix"]]
        return None

    cuda = read("cuda-driver-probe.json") or {}
    sysfs = read("sysfs.json") or {}
    collectives = []
    limitations = []
    for step in metadata["steps"]:
        result = read(step["parsed"]) if step["name"].startswith("nccl-") and step.get("parsed") else None
        if step["status"] == "ok" and result and result.get("check_passed"):
            collectives.append({"name": step["name"], "group": step["context"]["id"],
                                "operation": "all_reduce" if step["name"].endswith("all_reduce") else "all_gather",
                                "transport_env": {k: v for k, v in step.get("environment_overrides", {}).items() if k in ("NCCL_CUMEM_HOST_ENABLE", "NCCL_P2P_DIRECT_DISABLE", "NCCL_P2P_DISABLE")},
                                "gpu_indices": step["context"]["indices"], "rows": result["rows"],
                                "topology_xml": result.get("topology_xml"), "xml_root": result.get("xml_root"),
                                "xml_gpu_count": result.get("xml_gpu_count"), "xml_error": result.get("xml_error")})
        if step["status"] not in ("ok", "observed") or step.get("incomplete_artifact"):
            reason = step.get("reason") or step.get("error")
            if not reason and step.get("incomplete_artifact"):
                if step.get("missing_topology_xml"):
                    reason = "NCCL topology XML was not produced"
                elif result and result.get("xml_error"):
                    reason = "Invalid NCCL topology XML: " + result["xml_error"]
                else:
                    reason = "Incomplete experiment artifact"
            limitations.append({"step": step["name"], "status": step["status"],
                                "reason": reason or metadata.get("stop_reason"),
                                "incomplete_artifact": bool(step.get("incomplete_artifact")),
                                "artifacts": step.get("artifacts", [])})

    return {
        "schema_version": 1, "status": metadata["status"], "scope": metadata.get("scope", "full suite"), "gpu_name": devices[0]["name"],
        "devices": [{"id": index, "smi_index": gpu["smi_index"], "bus_id": gpu["bus_id"],
                     "numa_node": gpu.get("numa_node"), "pci_ancestors": gpu.get("pci_ancestors", [])}
                    for index, gpu in enumerate(devices)],
        "path_labels": paths, "can_access_peer": cuda.get("can_access_peer"),
        "pci_devices": sysfs.get("pci_devices", {}),
        "numa_nodes": {key: {"cpulist": value["cpulist"], "distance": value["distance"]}
                       for key, value in sysfs.get("numa_nodes", {}).items()},
        "bandwidth_gbs": matrix("gpu-pair-bandwidth.json", "device_to_device_memcpy_write_ce"),
        "remote_latency_ns": matrix("gpu-remote-memory-latency.json", "device_to_device_latency_sm"),
        "host_to_device_gbs": matrix("host-serial-transfers.json", "host_to_device_memcpy_ce"),
        "device_to_host_gbs": matrix("host-serial-transfers.json", "device_to_host_memcpy_ce"),
        "host_to_all_gbs": matrix("host-concurrent-transfers.json", "host_to_all_memcpy_ce"),
        "all_to_host_gbs": matrix("host-concurrent-transfers.json", "all_to_host_memcpy_ce"),
        "fan_in_gbs": matrix("gpu-concurrent-fan-in.json", "all_to_one_write_ce"),
        "groups": metadata.get("groups", []), "collectives": collectives,
        "numa_first_touch": [{"label": step["name"], "context": step["context"], "result": read(step["parsed"])}
                             for step in metadata["steps"] if step["name"].startswith("numa-first-touch-") and step["status"] == "ok" and step.get("parsed")],
        "selected_transport": metadata.get("selected_transport"),
        "selected_transport_environment": metadata.get("selected_transport_environment"),
        "collective_matrix_complete": metadata.get("collective_matrix_complete"),
        "limitations": limitations,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = summarize(args.folder)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": data["status"], "gpu_count": len(data["devices"]),
                      "collectives": len(data["collectives"]), "bandwidth_available": data["bandwidth_gbs"] is not None}))
