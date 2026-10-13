"""A LeRobot policy (SmolVLA, pi0.5, ...) called in-process.

One observation in, one action chunk out.

This is the policy's own latency, with no network in between. It follows the
same steps as LeRobot's async policy server (lerobot.async_inference.policy_server):
preprocess, predict the chunk, postprocess.
"""

import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from lerobot.configs.types import FeatureType
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.policies.utils import prepare_observation_for_inference

# pi0.5 ships with torch.compile on. Triton's bundled ptxas predates the Thor's
# sm_110a target and fails; the system CUDA 13.2 ptxas supports it.
os.environ.setdefault("TRITON_PTXAS_BLACKWELL_PATH", "/usr/local/cuda/bin/ptxas")


@dataclass
class ChunkReply:
    t_ready: float  # monotonic time the postprocessed chunk was on the CPU
    n_actions: int
    stages_ms: dict[str, float]  # preprocess, inference, postprocess


class LeRobotPolicy:
    def __init__(self, checkpoint: str, policy_type: str, device: str = "cuda"):
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.policy = get_policy_class(policy_type).from_pretrained(checkpoint)
        self.policy.to(self.device).eval()
        self.policy.reset()  # clear any observation/action queues the policy keeps between calls

        on_device = {"device_processor": {"device": device}}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=checkpoint,
            preprocessor_overrides=on_device,
            postprocessor_overrides=on_device,
        )

    def camera_shapes(self) -> dict[str, tuple[int, int, int]]:
        """Camera observation keys and their (channels, height, width), per the policy config."""
        return {
            key: tuple(feature.shape)
            for key, feature in self.policy.config.input_features.items()
            if feature.type is FeatureType.VISUAL
        }

    def state_dim(self) -> int:
        return self.policy.config.robot_state_feature.shape[0]

    def weight_dtype(self) -> str:
        """The precision the weights were loaded in, e.g. 'bfloat16'."""
        return str(next(self.policy.parameters()).dtype).removeprefix("torch.")

    def _wait_for_gpu(self) -> None:
        """GPU work runs asynchronously; wait for it so stage times are real."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def act(self, observation: dict[str, np.ndarray], task: str) -> ChunkReply:
        """Run one observation through the policy and return when the action chunk is ready.

        `observation` maps feature keys to arrays as a robot produces them:
        images (height, width, channels) uint8, state a float32 vector.
        """
        t0 = time.monotonic()
        # prepare_observation_for_inference converts arrays in place, so give it a copy.
        batch = prepare_observation_for_inference(dict(observation), self.device, task=task)
        batch = self.preprocessor(batch)
        self._wait_for_gpu()
        t1 = time.monotonic()

        chunk = self.policy.predict_action_chunk(batch)  # (1, chunk_size, action_dim)
        self._wait_for_gpu()
        t2 = time.monotonic()

        # Unnormalizing is per action dimension, so the whole chunk goes through as
        # (chunk_size, action_dim) in one call instead of one call per action.
        actions = self.postprocessor(chunk.squeeze(0)).cpu()
        t3 = time.monotonic()

        return ChunkReply(
            t_ready=t3,
            n_actions=actions.shape[0],
            stages_ms={
                "preprocess": (t1 - t0) * 1000,
                "inference": (t2 - t1) * 1000,
                "postprocess": (t3 - t2) * 1000,
            },
        )
