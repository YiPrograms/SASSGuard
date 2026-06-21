#! /bin/bash

# t-rex
cd ~/SASSGuard/workloads/mining_samples/trex

./t-rex --algo ethash --benchmark


# GMiner
cd ~/SASSGuard/workloads/mining_samples/GMiner

./miner --algo ethash --server ethw.2miners.com:2020 \
	--user 0x00192Fb10dF37c9FB26829eb2CC623cd1BF599E8 \
	--pass x --devices 0 --cuda 1 --opencl 0 --watchdog 0
./miner --algo kheavyhash --server kas.2miners.com:2020 \
	--user kaspa:qrrzeucwfetuty3qserqydw4z4ax9unxd23zwp7tndvg7cs3ls8dvwldeayv5 \
	--pass x --devices 0 --cuda 1 --opencl 0 --watchdog 0


# vLLM benchmark
cd ~/SASSGuard/workloads/benign_samples/vllm_benchmark
source .venv/bin/activate

vllm bench throughput \
	--model /home/yi/models/Llama-3.2-3B-Instruct \
	--dataset-name sharegpt \
	--dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
	--num-prompts 100

# HPL
cd ~/SASSGuard/workloads/benign_samples/nv_hpl
./run.sh
