"""Measure transfers from verified first-touched pages, without set_mempolicy."""
import argparse
import ctypes as C
import hashlib
import json
import mmap
import os
from pathlib import Path
import re
import statistics


def ranges(text):
    out = set()
    for part in text.strip().split(','):
        ends = part.split('-'); out.update(range(int(ends[0]), int(ends[-1]) + 1))
    return out


def residence(address, size, node):
    for line in Path('/proc/self/maps').read_text().splitlines():
        low, high = [int(x, 16) for x in line.split()[0].split('-')]
        if low == address and high == address + size:
            break
    else:
        raise RuntimeError('Guarded host mapping boundaries are not exact')
    row = next(x for x in Path('/proc/self/numa_maps').read_text().splitlines() if int(x.split()[0], 16) == address)
    pages = {int(k): int(v) for k, v in re.findall(r'\bN(\d+)=(\d+)', row)}
    page_kb = int(re.search(r'kernelpagesize_kB=(\d+)', row)[1])
    if set(pages) != {node} or pages[node] * page_kb * 1024 != size:
        raise RuntimeError('Host pages are not entirely on the requested NUMA node: ' + row)
    return {'node': node, 'pages': pages, 'kernel_page_kib': page_kb, 'numa_maps': row}


def measure(node, submit_node):
    size = 256 * 1024**2; page = mmap.PAGESIZE
    allowed = os.sched_getaffinity(0)
    def cpus(n):
        available = sorted(allowed & ranges(Path(f'/sys/devices/system/node/node{n}/cpulist').read_text()))
        if not available: raise RuntimeError('No allowed CPU on requested node')
        return {available[0]}
    touch_cpu, submit_cpu = cpus(node), cpus(submit_node)
    libc = C.CDLL(None, use_errno=True)
    libc.mprotect.argtypes = [C.c_void_p, C.c_size_t, C.c_int]
    host = mmap.mmap(-1, size + 2 * page, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    base = C.addressof(C.c_char.from_buffer(host)); address = base + page
    for guard in (base, address + size):
        if libc.mprotect(guard, page, 0): raise OSError(C.get_errno(), 'mprotect guard')
    host.madvise(mmap.MADV_NOHUGEPAGE)
    os.sched_setaffinity(0, touch_cpu)
    C.memset(address, 0xA5, size)
    before = residence(address, size, node)
    # Submission CPU is identical for local/remote; only host page location changes.
    os.sched_setaffinity(0, submit_cpu)
    cuda = C.CDLL('libcudart.so.12')
    def bind(name, args):
        f = getattr(cuda, name); f.argtypes = args; f.restype = C.c_int
        def call(*values):
            code = f(*values)
            if code: raise RuntimeError(f'{name} failed with CUDA error {code}')
        return call
    ptr, size_t, integer = C.c_void_p, C.c_size_t, C.c_int
    setdev = bind('cudaSetDevice', [integer]); malloc = bind('cudaMalloc', [C.POINTER(ptr), size_t])
    reg = bind('cudaHostRegister', [ptr, size_t, C.c_uint]); unreg = bind('cudaHostUnregister', [ptr])
    free = bind('cudaFree', [ptr]); sync = bind('cudaDeviceSynchronize', [])
    copy = bind('cudaMemcpyAsync', [ptr, ptr, size_t, integer, ptr])
    create = bind('cudaEventCreate', [C.POINTER(ptr)]); record = bind('cudaEventRecord', [ptr, ptr])
    event_sync = bind('cudaEventSynchronize', [ptr]); elapsed = bind('cudaEventElapsedTime', [C.POINTER(C.c_float), ptr, ptr])
    destroy = bind('cudaEventDestroy', [ptr])
    dev = ptr(); events = []; registered = False
    try:
        setdev(0); malloc(C.byref(dev), size); reg(address, size, 0); registered = True
        registered_pages = residence(address, size, node)
        for _ in range(2):
            e = ptr(); create(C.byref(e)); events.append(e)
        results = {}
        for name, destination, source, kind in [('host_to_device', dev, address, 1), ('device_to_host', address, dev, 2)]:
            copy(destination, source, size, kind, None); sync()
            samples = []
            for _ in range(3):
                record(events[0], None)
                for _ in range(16): copy(destination, source, size, kind, None)
                record(events[1], None); event_sync(events[1])
                ms = C.c_float(); elapsed(C.byref(ms), *events)
                if ms.value <= 0: raise RuntimeError('Invalid CUDA event duration')
                samples.append(size * 16 / (ms.value / 1000) / 1e9)
            results[name + '_gbs'] = statistics.median(samples)
            results[name + '_samples_gbs'] = samples
        C.memset(address, 0, size)
        copy(address, dev, size, 2, None); sync()
        with memoryview(host)[page:page+size] as view: digest = hashlib.sha256(view).hexdigest()
        correct = digest == hashlib.sha256(b'\xa5' * size).hexdigest()
        if not correct: raise RuntimeError('Full-buffer GPU round-trip verification failed')
        after = residence(address, size, node)
        return {'status': 'complete', 'method': 'CPU first-touch + observed page residency + cudaHostRegister; not strict membind',
                'bytes': size, 'iterations': 16, 'samples': 3, 'correctness_passed': correct,
                'memory_node': node, 'submission_node': submit_node, 'touch_cpu': sorted(touch_cpu),
                'submission_cpu': sorted(submit_cpu), 'before_register': before,
                'after_register': registered_pages, 'after_transfers': after, **results}
    finally:
        for event in events: destroy(event)
        if registered: unreg(address)
        if dev.value: free(dev)
        host.close(); os.sched_setaffinity(0, allowed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--node', type=int, required=True); ap.add_argument('--submit-node', type=int, required=True)
    args = ap.parse_args()
    print(json.dumps(measure(args.node, args.submit_node)))
