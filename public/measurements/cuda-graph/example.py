"""Four elementwise kernels with a real cross-stream fork and join."""
import torch


def workload(static_input, x, b, c, y, side):
    origin = torch.cuda.current_stream()
    torch.mul(static_input, 2, out=x)   # A
    side.wait_stream(origin)           # A -> C; record before submitting B
    torch.add(x, 1, out=b)             # B, ordered after A on origin
    with torch.cuda.stream(side):
        torch.add(x, 2, out=c)         # C, independent of B
    origin.wait_stream(side)           # C -> D; join back to capture origin
    torch.add(b, c, out=y)             # D, also ordered after B on origin


@torch.no_grad()
def prepare():
    static_input = torch.full((4096,), 7., device="cuda")
    x, b, c, y = [torch.zeros_like(static_input) for _ in range(4)]
    buffers = (static_input, x, b, c, y)
    capture_stream, side = torch.cuda.Stream(), torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        for _ in range(20):
            workload(*buffers, side)
    torch.cuda.current_stream().wait_stream(capture_stream)
    y.zero_()
    torch.cuda.synchronize()
    return buffers, capture_stream, side


@torch.no_grad()
def main():
    buffers, capture_stream, side = prepare()
    static_input, x, b, c, y = buffers
    addresses = [tensor.data_ptr() for tensor in buffers]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        workload(*buffers, side)
    assert torch.count_nonzero(y).item() == 0
    graph.replay()
    assert torch.equal(y, torch.full_like(y, 31))
    saved = y.clone()
    graph.replay()
    assert torch.equal(y, torch.full_like(y, 31))

    static_input.copy_(torch.full_like(static_input, 9))
    graph.replay()
    assert torch.equal(y, torch.full_like(y, 39))
    assert torch.equal(saved, torch.full_like(saved, 31))

    new_input = torch.arange(4096, dtype=torch.float32, device="cuda") / 16
    static_input.copy_(new_input)
    graph.replay()
    torch.testing.assert_close(y, 4 * new_input + 3, rtol=0, atol=0)
    assert addresses == [tensor.data_ptr() for tensor in buffers]
    print("PASS: capture leaves y=0; replay gives 31 twice; input 9 gives 39; "
          "saved output stays 31; vector reference and fixed addresses verified.")


if __name__ == "__main__":
    main()
