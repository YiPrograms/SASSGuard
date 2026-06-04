# CUDA Mining Kernel Dataset

CUDA implementations of various cryptocurrency mining algorithms, designed for benchmarking and dataset generation. Some of the algorithms have a "mono" variant (single-kernel) and a "split" variant (multi-kernel with separate dataset generation and mining phases).

The programs are offline benchmarks only: there is no pool mining, no wallet handling, no network code, and no real mining submission behavior.

Every generated executable source accepts a mandatory first positional argument:

```bash
./<program> <runtime_seconds> [optional args...]
```

Examples:

```bash
./ethash 60
./kawpow_split 120
./verthash 300 --dataset-mb 128
```

## Common options

All programs share the same parser and support:

* `<runtime_seconds>` (mandatory; must be a positive integer)
* `--blocks <N>`
* `--threads <N>`
* `--nonces-per-thread <N>`
* `--dataset-mb <N>`
* `--scratchpad-mb <N>`
* `--seed <N>`
* `--sync-every <N>`

## Standard output

Each benchmark prints:

```text
algorithm=<name>
variant=<mono|split>
runtime_seconds=<N>
threads_per_block=<N>
total_launches=<N>
total_nonces=<N>
result_count=<N>
checksum=0x...
status=ok
```

## Safety and scope

These are representative synthetic or reduced-reference kernels for dataset generation. They are not complete cryptocurrency miners and are not designed for profitability or protocol compatibility.
