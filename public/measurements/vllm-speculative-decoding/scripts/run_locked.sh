#!/bin/sh
# usage: run_locked.sh <out-root> <name:args> ...   (one flock window, configs run sequentially, resumable)
# Holds the shared GPU lock for the whole window.
BASE=/root/blog-vllm-20260926
exec flock -n $BASE/gpu.lock timeout --signal=TERM --kill-after=30s 1800s "$0.inner" "$@"
