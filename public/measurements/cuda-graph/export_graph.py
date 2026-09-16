"""Export raw CUDA DOT and node records; also called by profile.py."""
import hashlib
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import torch

from example import prepare, workload


def node_card(node):
    width = 96
    lines = [f"NODE {node['node_id']}: KERNEL — {node['label']}",
             f"Operation:   {node['operation']}",
             f"ID:          {node['node_id']} (topoId: {node['topo_id']})"]
    wrapped = textwrap.wrap(node["kernel"], width=width - 15, break_on_hyphens=False)
    lines += ["Kernel:      " + wrapped[0]] + ["             " + s for s in wrapped[1:]]
    lines += [f"Launch:      {node['launch']}", f"Node handle: {node['node_handle']}",
              f"Func handle: {node['func_handle']}"]
    return "┌" + "─" * width + "┐\n" + "\n".join(
        "│ " + line.ljust(width - 2) + " │" for line in lines) + "\n└" + "─" * width + "┘"


def export_structure(graph, buffers, provenance):
    output = Path(__file__).with_name("structure")
    output.mkdir(exist_ok=True)
    graph.debug_dump(str(output / "graph.dot"))
    dot = (output / "graph.dot").read_text()
    nodes = []
    operations = ["x = 2 * static_input", "b = x + 1", "c = x + 2", "y = b + c"]
    for name, attributes in re.findall(r'"(graph_\d+_node_\d+)"\[([^;]+)\];', dot):
        identity = re.search(r'\{ID \| (\d+) \(topoId: (\d+)\)', attributes)
        kernel = re.search(r'\| (\S+?)\\<\\<\\<(\d+),(\d+),(\d+)\\>\\>\\>', attributes)
        handles = re.search(r'\{node handle \| func handle\} \| \{(0x[0-9A-Fa-f]+) \| (0x[0-9A-Fa-f]+)\}', attributes)
        assert identity and kernel and handles and 'label="{KERNEL' in attributes
        index = int(identity[1])
        nodes.append({"id": name, "node_id": index, "topo_id": int(identity[2]),
                      "label": "ABCD"[index], "operation": operations[index], "kernel": kernel[1],
                      "grid_x": int(kernel[2]), "block_x": int(kernel[3]), "shared_bytes": int(kernel[4]),
                      "launch": f"<<<{kernel[2]},{kernel[3]},{kernel[4]}>>>",
                      "node_handle": handles[1], "func_handle": handles[2]})
    nodes.sort(key=lambda node: node["node_id"])
    edges = [{"from": a, "to": b} for a, b in re.findall(
        r'"(graph_\d+_node_\d+)" -> "(graph_\d+_node_\d+)"', dot)]
    assert len(nodes) == 4 and len(edges) == 4
    ids = {node["id"]: node["node_id"] for node in nodes}
    assert {(ids[e["from"]], ids[e["to"]]) for e in edges} == {(0, 1), (0, 2), (1, 3), (2, 3)}
    assert "MulFunctor" in nodes[0]["kernel"] and "CUDAFunctor_add" in nodes[3]["kernel"]
    assert nodes[1]["kernel"] == nodes[2]["kernel"] and "CUDAFunctorOnSelf_add" in nodes[1]["kernel"]
    assert len({node["node_handle"] for node in nodes}) == 4
    data = {"source": "torch.cuda.CUDAGraph.debug_dump / cudaGraphDebugDotPrint",
            "provenance": provenance, "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "shape": [4096], "dtype": "float32", "nodes": nodes, "edges": edges,
            "buffer_addresses": {key: hex(value.data_ptr()) for key, value in
                                 zip(("static_input", "x", "b", "c", "y"), buffers)},
            "correctness_passed": True, "dot_sha256": hashlib.sha256(dot.encode()).hexdigest()}
    (output / "graph.json").write_text(json.dumps(data, indent=2) + "\n")
    text = "Dependencies: A -> B, A -> C, B -> D, C -> D\n\n"
    (output / "nodes.txt").write_text(text + "\n\n".join(node_card(n) for n in nodes) + "\n")
    print("Exported 4 kernel nodes and 4 diamond edges, including raw IDs, symbols and handles.")


@torch.no_grad()
def main():
    buffers, capture_stream, side = prepare()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    graph.enable_debug_mode()
    with torch.cuda.graph(graph, stream=capture_stream):
        workload(*buffers, side)
    assert torch.count_nonzero(buffers[-1]).item() == 0
    graph.instantiate()
    graph.replay()
    assert torch.equal(buffers[-1], torch.full_like(buffers[-1], 31))
    export_structure(graph, buffers, "Independent export_graph.py run; rerun profile.py for same-report provenance")


if __name__ == "__main__":
    main()
