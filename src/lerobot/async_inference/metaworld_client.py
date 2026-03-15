#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Run MetaWorld evaluation against a remote async inference policy server.

Start the server on the inference machine:

```shell
python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
```

Then run the simulation client on the evaluation machine:

```shell
python -m lerobot.async_inference.metaworld_client \
    --policy.path=your-policy-id \
    --env.task=medium \
    --server_address=127.0.0.1:8080 \
    --n_episodes=2
```
"""

import logging
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pprint import pformat
from queue import Queue
from typing import Any

import grpc
import numpy as np

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import (
    FPSTracker,
    RemotePolicyConfig,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)
from lerobot.async_inference.robot_client import RobotClient
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import MetaworldEnv as MetaworldEnvConfig
from lerobot.envs.metaworld import (
    DIFFICULTY_TO_TASKS,
    TASK_DESCRIPTIONS,
    MetaworldEnv as SingleMetaworldEnv,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options
from lerobot.utils.constants import OBS_IMAGE
from lerobot.utils.import_utils import register_third_party_plugins


def _expand_metaworld_tasks(task: str) -> list[str]:
    task_groups = [item.strip() for item in task.split(",") if item.strip()]
    if not task_groups:
        raise ValueError("`env.task` must contain at least one MetaWorld task or difficulty group.")

    expanded: list[str] = []
    for item in task_groups:
        expanded.extend(DIFFICULTY_TO_TASKS.get(item, [item]))
    return expanded


def _make_rename_map(policy_cfg: PreTrainedConfig) -> dict[str, str]:
    image_features = getattr(policy_cfg, "image_features", {})
    if OBS_IMAGE in image_features:
        return {"observation.images.top": OBS_IMAGE}
    return {}


@dataclass
class AsyncMetaWorldClientConfig:
    policy: PreTrainedConfig | None = None
    env: MetaworldEnvConfig = field(default_factory=MetaworldEnvConfig)

    server_address: str = field(default="localhost:8080", metadata={"help": "Policy server address"})
    n_episodes: int = field(default=1, metadata={"help": "Episodes per MetaWorld task"})
    seed: int = field(default=0, metadata={"help": "Base random seed"})
    actions_per_chunk: int = field(default=50, metadata={"help": "Actions requested per server response"})
    chunk_size_threshold: float = field(
        default=0.5, metadata={"help": "Observation resend threshold based on action queue fullness"}
    )
    aggregate_fn_name: str = field(
        default="weighted_average", metadata={"help": "How to merge overlapping action chunks"}
    )
    client_device: str = field(
        default="cpu",
        metadata={"help": "Device to move received actions to before stepping the simulator"},
    )
    debug_visualize_queue_size: bool = field(default=False, metadata={"help": "Plot queue size after run"})

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.policy is None:
            raise ValueError("Policy path is required. Pass `--policy.path=<repo-or-dir>`.")

        if self.n_episodes <= 0:
            raise ValueError(f"`n_episodes` must be positive, got {self.n_episodes}.")

        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError(
                f"`chunk_size_threshold` must be between 0 and 1, got {self.chunk_size_threshold}."
            )

        if self.actions_per_chunk <= 0:
            raise ValueError(f"`actions_per_chunk` must be positive, got {self.actions_per_chunk}.")

        self.aggregate_fn = get_aggregate_function(self.aggregate_fn_name)

    @property
    def fps(self) -> int:
        return self.env.fps

    @property
    def environment_dt(self) -> float:
        return 1 / self.env.fps

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class MetaWorldRobotAdapter:
    def __init__(self, env_cfg: MetaworldEnvConfig):
        self.env_cfg = env_cfg
        self._connected = False
        self._env: SingleMetaworldEnv | None = None
        self._task_description = ""
        self._observation: dict[str, Any] | None = None
        self._reward_sum = 0.0
        self._success = False
        self._steps = 0
        self._terminated = False
        self._truncated = False

        image_feature = env_cfg.features.get("pixels/top") or env_cfg.features.get("top")
        if image_feature is None:
            raise ValueError("MetaWorld async eval requires an image observation feature.")

        action_feature = env_cfg.features["action"]
        self._image_shape = tuple(image_feature.shape)
        self._action_dim = action_feature.shape[0]

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        features: dict[str, type | tuple] = {"top": self._image_shape}
        if self.env_cfg.obs_type == "pixels_agent_pos":
            for idx in range(4):
                features[f"agent_pos_{idx}"] = float
        return features

    @property
    def action_features(self) -> dict[str, type]:
        return {f"action_{idx}": float for idx in range(self._action_dim)}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def task_description(self) -> str:
        return self._task_description

    @property
    def episode_finished(self) -> bool:
        return self._terminated or self._truncated or self._steps >= self.env_cfg.episode_length

    @property
    def steps(self) -> int:
        return self._steps

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        self._connected = True

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        return None

    def configure(self) -> None:
        return None

    def disconnect(self) -> None:
        self._connected = False
        self._env = None
        self._observation = None

    def start_episode(
        self,
        env: SingleMetaworldEnv,
        *,
        task_description: str,
        seed: int,
    ) -> None:
        if not self.is_connected:
            raise RuntimeError("MetaWorld adapter is not connected.")

        self._env = env
        self._task_description = task_description
        self._reward_sum = 0.0
        self._success = False
        self._steps = 0
        self._terminated = False
        self._truncated = False
        self._observation, _ = env.reset(seed=seed)

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise RuntimeError("MetaWorld adapter is not connected.")
        if self._observation is None:
            raise RuntimeError("Episode has not been reset yet.")

        raw_observation: dict[str, Any] = {"top": self._observation["pixels"]}
        if "agent_pos" in self._observation:
            agent_pos = np.asarray(self._observation["agent_pos"], dtype=np.float32)
            for idx, value in enumerate(agent_pos):
                raw_observation[f"agent_pos_{idx}"] = float(value)

        return raw_observation

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        if not self.is_connected:
            raise RuntimeError("MetaWorld adapter is not connected.")
        if self._env is None:
            raise RuntimeError("Episode has not been started yet.")

        action_np = np.asarray([action[key] for key in self.action_features], dtype=np.float32)
        self._observation, reward, self._terminated, self._truncated, info = self._env.step(action_np)
        self._reward_sum += float(reward)
        self._success = self._success or bool(info.get("is_success", False))
        self._steps += 1
        return action

    def build_result(self, task_name: str, seed: int) -> dict[str, Any]:
        return {
            "task": task_name,
            "seed": seed,
            "steps": self._steps,
            "reward_sum": self._reward_sum,
            "success": self._success,
        }


class AsyncMetaWorldClient(RobotClient):
    prefix = "metaworld_client"
    logger = get_logger(prefix)

    def __init__(self, cfg: AsyncMetaWorldClientConfig):
        self.cfg = cfg
        self.config = cfg
        self.robot = MetaWorldRobotAdapter(cfg.env)
        self.robot.connect()

        self.server_address = cfg.server_address
        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.policy_config = RemotePolicyConfig(
            policy_type=cfg.policy.type,
            pretrained_name_or_path=str(cfg.policy.pretrained_path),
            lerobot_features=lerobot_features,
            actions_per_chunk=cfg.actions_per_chunk,
            device=cfg.policy.device or "cpu",
            rename_map=_make_rename_map(cfg.policy),
        )

        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{cfg.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = -1

        self._chunk_size_threshold = cfg.chunk_size_threshold
        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size: list[int] = []
        self.start_barrier = threading.Barrier(2)

        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.must_go = threading.Event()
        self.must_go.set()

        self.logger.info("MetaWorld adapter connected and ready")

    def _reset_episode_state(self):
        self.stub.Ready(services_pb2.Empty())

        with self.latest_action_lock:
            self.latest_action = -1
        self.action_chunk_size = -1

        with self.action_queue_lock:
            self.action_queue = Queue()

        self.must_go.set()
        self.fps_tracker.reset()

    def _run_episode(
        self,
        env: SingleMetaworldEnv,
        *,
        task_name: str,
        task_description: str,
        episode_seed: int,
    ) -> dict[str, Any]:
        self._reset_episode_state()
        self.robot.start_episode(env, task_description=task_description, seed=episode_seed)
        self.logger.info(
            f"Episode started | task={task_name} | seed={episode_seed} | "
            f"max_steps={self.cfg.env.episode_length}"
        )

        while self.running and not self.robot.episode_finished:
            loop_start = time.perf_counter()

            if self.actions_available():
                self.control_loop_action()

            if self._ready_to_send_observation():
                self.control_loop_observation(task=task_description)

            time.sleep(max(0.0, self.config.environment_dt - (time.perf_counter() - loop_start)))

        return self.robot.build_result(task_name, episode_seed)

    def evaluate(self) -> list[dict[str, Any]]:
        if not self.start():
            return []

        action_receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)
        action_receiver_thread.start()
        self.start_barrier.wait()

        results: list[dict[str, Any]] = []
        task_names = _expand_metaworld_tasks(self.cfg.env.task)

        try:
            for task_name in task_names:
                task_description = TASK_DESCRIPTIONS.get(task_name, task_name)
                env = SingleMetaworldEnv(task=task_name, **self.cfg.env.gym_kwargs)
                try:
                    for episode_idx in range(self.cfg.n_episodes):
                        episode_seed = self.cfg.seed + episode_idx
                        result = self._run_episode(
                            env,
                            task_name=task_name,
                            task_description=task_description,
                            episode_seed=episode_seed,
                        )
                        results.append(result)
                        self.logger.info(
                            f"Task={task_name} | episode={episode_idx} | "
                            f"success={result['success']} | steps={result['steps']} | "
                            f"reward_sum={result['reward_sum']:.3f}"
                        )
                finally:
                    env.close()
        finally:
            self.stop()
            action_receiver_thread.join(timeout=1)

        return results


def _summarize_results(results: Iterable[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["task"], []).append(result)

    for task, task_results in grouped.items():
        success_rate = sum(1 for item in task_results if item["success"]) / len(task_results)
        avg_reward = sum(item["reward_sum"] for item in task_results) / len(task_results)
        avg_steps = sum(item["steps"] for item in task_results) / len(task_results)
        summary[task] = {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_steps": avg_steps,
        }

    return summary


@parser.wrap()
def async_metaworld_client(cfg: AsyncMetaWorldClientConfig):
    logging.info(pformat(asdict(cfg)))
    client = AsyncMetaWorldClient(cfg)
    results = client.evaluate()
    summary = _summarize_results(results)

    client.logger.info("MetaWorld async evaluation summary:")
    for task, metrics in summary.items():
        client.logger.info(
            f"{task} | success_rate={metrics['success_rate']:.3f} | "
            f"avg_reward={metrics['avg_reward']:.3f} | avg_steps={metrics['avg_steps']:.1f}"
        )

    if cfg.debug_visualize_queue_size and client.action_queue_size:
        visualize_action_queue_size(client.action_queue_size)


if __name__ == "__main__":
    register_third_party_plugins()
    async_metaworld_client()
