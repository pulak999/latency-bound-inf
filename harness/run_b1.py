"""Stack B run: a LeRobot policy (SmolVLA or pi0.5), in-process.

Synthetic observation in, one action chunk out.

    python -m harness.run_b1 --policy smolvla latency --period 1 --n-requests 100
    python -m harness.run_b1 --policy pi05 throughput --duration 300

Run with the LeRobot environment's Python (/data/projects/pulak/envs/lerobot).
Output layout matches Stack A: run.json, records.jsonl, power.csv,
tegrastats.log, summary.json, latency_cdf.png.
"""

import argparse
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from harness import runner
from harness.device import gpu_co_tenants, power_mode
from harness.records import RequestRecord
from harness.synthetic import synthetic_observation

# Policy name on the command line -> (LeRobot policy type, Hugging Face checkpoint).
POLICIES = {
    "smolvla": ("smolvla", "lerobot/smolvla_base"),
    "pi05": ("pi05", "lerobot/pi05_libero_finetuned_v044"),
}
# A LIBERO-style instruction. The policy sees the same text on every request.
TASK = "put the bowl on the plate"
OBSERVATION_POOL_SIZE = 64  # distinct observations, cycled; request i uses observation i % 64
N_WARMUP = 3  # first calls pay one-off CUDA setup and kernel selection; not recorded


def parse_args() -> argparse.Namespace:
    parser = runner.mode_arguments(__doc__.splitlines()[0], default_period_s=1.0)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="smolvla")
    return parser.parse_args()


async def measure(args: argparse.Namespace, policy, mode: str) -> list[RequestRecord]:
    """Warm up, then send observations on the chosen schedule. Returns one record per request."""
    observations = [
        synthetic_observation(seed, policy.camera_shapes(), policy.state_dim())
        for seed in range(OBSERVATION_POOL_SIZE)
    ]
    for observation in observations[:N_WARMUP]:
        policy.act(observation, TASK)

    # The policy call blocks, so it runs on one worker thread. One thread means
    # one request on the GPU at a time; a request that arrives while another is
    # running waits its turn, like a single-slot server.
    gpu_worker = ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_running_loop()

    async def send_one(req_id: int, t_arrival: float, deadline: float | None):
        co_tenants = asyncio.create_task(gpu_co_tenants(exclude_pids={os.getpid()}))
        observation = observations[req_id % OBSERVATION_POOL_SIZE]
        t_send = time.monotonic()
        reply = await loop.run_in_executor(gpu_worker, policy.act, observation, TASK)

        return RequestRecord(
            req_id=req_id,
            t_arrival=t_arrival,
            t_send=t_send,
            t_first_token=None,  # the chunk arrives all at once
            t_last_token=reply.t_ready,
            deadline=deadline,
            n_prompt_tokens=None,
            n_output_tokens=None,
            output_text="",
            engine_prompt_ms=None,
            engine_decode_ms=None,
            engine="lerobot (in-process)",
            model=policy.checkpoint,
            quant=policy.weight_dtype(),
            power_mode=mode,
            co_tenants=await co_tenants,
            n_actions=reply.n_actions,
            engine_stages_ms=reply.stages_ms,
        )

    try:
        return await runner.send_on_schedule(args, send_one)
    finally:
        gpu_worker.shutdown()


def main() -> None:
    args = parse_args()

    # Imported here, not at the top, so `--help` works without LeRobot installed (CI).
    import lerobot

    from harness.engines.lerobot_policy import LeRobotPolicy

    policy_type, checkpoint = POLICIES[args.policy]
    policy = LeRobotPolicy(checkpoint, policy_type)
    run_dir = runner.new_run_dir(f"b1-{args.policy}", args.mode)
    mode = power_mode()

    config = vars(args) | {
        "stack": "B1",
        "lerobot_version": lerobot.__version__,
        "policy_type": policy_type,
        "checkpoint": checkpoint,
        "weight_dtype": policy.weight_dtype(),
        "task": TASK,
        "camera_shapes": policy.camera_shapes(),
        "state_dim": policy.state_dim(),
        "observation_pool_size": OBSERVATION_POOL_SIZE,
        "n_warmup": N_WARMUP,
        "power_mode": mode,
    }
    runner.measure_and_report(run_dir, config, lambda: measure(args, policy, mode))


if __name__ == "__main__":
    main()
