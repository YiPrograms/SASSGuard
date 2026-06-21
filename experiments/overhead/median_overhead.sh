#!/usr/bin/env bash
set -euo pipefail

runs="${1:-100}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binary="${script_dir}/drivertest"
progress_every="${PROGRESS_EVERY:-10}"
run_timeout_seconds="${RUN_TIMEOUT_SECONDS:-120}"

if ! [[ "${runs}" =~ ^[0-9]+$ ]] || (( runs < 1 )); then
  echo "usage: $0 [positive-run-count]" >&2
  exit 2
fi

if ! [[ "${progress_every}" =~ ^[0-9]+$ ]] || (( progress_every < 1 )); then
  echo "PROGRESS_EVERY must be a positive integer" >&2
  exit 2
fi

if ! [[ "${run_timeout_seconds}" =~ ^[0-9]+$ ]] || (( run_timeout_seconds < 1 )); then
  echo "RUN_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 2
fi

if [[ ! -x "${binary}" ]]; then
  echo "missing executable: ${binary}" >&2
  echo "build it with: make -C ${script_dir} drivertest" >&2
  exit 2
fi

work_log="$(mktemp)"
device_get_count_samples="$(mktemp)"
module_samples="$(mktemp)"
mem_alloc_da_samples="$(mktemp)"
launch_samples="$(mktemp)"
mem_free_da_samples="$(mktemp)"
cleanup() {
  rm -f "${work_log}" \
        "${device_get_count_samples}" \
        "${module_samples}" \
        "${mem_alloc_da_samples}" \
        "${launch_samples}" \
        "${mem_free_da_samples}"
}
trap cleanup EXIT

cd "${script_dir}"
started_at="$(date +%s)"
printf 'running %d drivertest executions; progress every %d completed runs\n' "${runs}" "${progress_every}" >&2

for ((i = 1; i <= runs; i++)); do
  if ! timeout "${run_timeout_seconds}" "${binary}" >"${work_log}"; then
    echo "drivertest failed or exceeded ${run_timeout_seconds}s on run ${i}" >&2
    cat "${work_log}" >&2
    exit 1
  fi

  device_get_count_ns="$(
    awk -F': ' 'index($0, "Time (ns) of cuDeviceGetCount(&deviceCount):") == 1 { print $2 }' "${work_log}"
  )"
  module_load_ns="$(
    awk -F': ' 'index($0, "Time (ns) of cuModuleLoad(&module, module_file):") == 1 { print $2 }' "${work_log}"
  )"
  mem_alloc_da_ns="$(
    awk -F': ' 'index($0, "Time (ns) of cuMemAlloc(d_a, sizeof(int) * N):") == 1 { print $2 }' "${work_log}"
  )"
  launch_kernel_ns="$(
    awk -F': ' 'index($0, "Time (ns) of cuLaunchKernel(function,") == 1 { print $2 }' "${work_log}"
  )"
  mem_free_da_ns="$(
    awk -F': ' 'index($0, "Time (ns) of cuMemFree(d_a):") == 1 { print $2 }' "${work_log}"
  )"

  if ! [[ "${device_get_count_ns}" =~ ^[0-9]+$ ]]; then
    echo "failed to parse cuDeviceGetCount time on run ${i}" >&2
    cat "${work_log}" >&2
    exit 1
  fi

  if ! [[ "${module_load_ns}" =~ ^[0-9]+$ ]]; then
    echo "failed to parse cuModuleLoad time on run ${i}" >&2
    cat "${work_log}" >&2
    exit 1
  fi

  if ! [[ "${mem_alloc_da_ns}" =~ ^[0-9]+$ ]]; then
    echo "failed to parse cuMemAlloc(d_a) time on run ${i}" >&2
    cat "${work_log}" >&2
    exit 1
  fi

  if ! [[ "${launch_kernel_ns}" =~ ^[0-9]+$ ]]; then
    echo "failed to parse cuLaunchKernel time on run ${i}" >&2
    cat "${work_log}" >&2
    exit 1
  fi

  if ! [[ "${mem_free_da_ns}" =~ ^[0-9]+$ ]]; then
    echo "failed to parse cuMemFree(d_a) time on run ${i}" >&2
    cat "${work_log}" >&2
    exit 1
  fi

  printf '%s\n' "${device_get_count_ns}" >>"${device_get_count_samples}"
  printf '%s\n' "${module_load_ns}" >>"${module_samples}"
  printf '%s\n' "${mem_alloc_da_ns}" >>"${mem_alloc_da_samples}"
  printf '%s\n' "${launch_kernel_ns}" >>"${launch_samples}"
  printf '%s\n' "${mem_free_da_ns}" >>"${mem_free_da_samples}"

  if (( i == 1 || i % progress_every == 0 || i == runs )); then
    elapsed_seconds=$(($(date +%s) - started_at))
    printf 'completed %d/%d runs (%ds elapsed)\n' "${i}" "${runs}" "${elapsed_seconds}" >&2
  fi
done

median_ns() {
  sort -n "$1" | awk '
    { values[NR] = $1 }
    END {
      if (NR == 0) {
        exit 1
      }
      if (NR % 2 == 1) {
        printf "%.1f", values[(NR + 1) / 2]
      } else {
        printf "%.1f", (values[NR / 2] + values[NR / 2 + 1]) / 2.0
      }
    }
  '
}

device_get_count_median_ns="$(median_ns "${device_get_count_samples}")"
module_median_ns="$(median_ns "${module_samples}")"
mem_alloc_da_median_ns="$(median_ns "${mem_alloc_da_samples}")"
launch_median_ns="$(median_ns "${launch_samples}")"
mem_free_da_median_ns="$(median_ns "${mem_free_da_samples}")"

awk -v runs="${runs}" \
    -v device_get_count_ns="${device_get_count_median_ns}" \
    -v module_ns="${module_median_ns}" \
    -v mem_alloc_da_ns="${mem_alloc_da_median_ns}" \
    -v launch_ns="${launch_median_ns}" \
    -v mem_free_da_ns="${mem_free_da_median_ns}" '
  BEGIN {
    printf "runs: %d\n", runs
    printf "cuDeviceGetCount median: %.1f ns (%.3f us)\n", device_get_count_ns, device_get_count_ns / 1000.0
    printf "cuModuleLoad median: %.1f ns (%.3f us)\n", module_ns, module_ns / 1000.0
    printf "cuMemAlloc(d_a) median: %.1f ns (%.3f us)\n", mem_alloc_da_ns, mem_alloc_da_ns / 1000.0
    printf "cuLaunchKernel median: %.1f ns (%.3f us)\n", launch_ns, launch_ns / 1000.0
    printf "cuMemFree(d_a) median: %.1f ns (%.3f us)\n", mem_free_da_ns, mem_free_da_ns / 1000.0
  }
'
