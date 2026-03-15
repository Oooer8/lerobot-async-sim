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
import pickle  # nosec
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pprint import pformat
from queue import Queue
from typing import Any

import grpc
import numpy as np
import torch

from lerobot.async_inference.configs import get_aggregate_function
from lerobot.async_inference.helpers import (
    FPSTracker,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    visualize_action_queue_size,
)
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
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
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


def _build_lerobot_features(obs_type: str, image_shape: tuple[int, int, int]) -> dict[str, dict[str, Any]]:
    features: dict[str, dict[str, Any]] = {
        "observation.images.top": {
            "dtype": "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        }
    }

    if obs_type == "pixels_agent_pos":
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (4,),
            "names": [f"agent_pos_{idx}" for idx in range(4)],
        }

    return features


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

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]


class AsyncMetaWorldClient:
    logger = get_logger("metaworld_client")

    def __init__(self, cfg: AsyncMetaWorldClientConfig):
        self.cfg = cfg
        self.shutdown_event = threading.Event()
        self.channel = grpc.insecure_channel(
            cfg.server_address, grpc_channel_options(initial_backoff=f"{1 / cfg.env.fps:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)

        self.action_queue: Queue[TimedAction] = Queue()
        self.action_queue_lock = threading.Lock()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = cfg.actions_per_chunk
        self.must_go = threading.Event()
        self.must_go.set()

        self.action_queue_size: list[int] = []
        self.fps_tracker = FPSTracker(target_fps=cfg.env.fps)

        self.lerobot_features: dict[str, dict[str, Any]] | None = None
        self.policy_setup_sent = False
        self._current_task_name = ""
        self._current_task_description = ""

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def start(self):
        self.stub.Ready(services_pb2.Empty())
        self.logger.info(f"Connected to policy server at {self.cfg.server_address}")

    def stop(self):
        self.shutdown_event.set()
        self.channel.close()

    def _reset_episode_state(self):
        with self.latest_action_lock:
            self.latest_action = -1
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.must_go.set()

    def _ensure_policy_setup(self, first_observation: dict[str, Any]):
        if self.policy_setup_sent:
            return

        image = first_observation["pixels"]
        self.lerobot_features = _build_lerobot_features(self.cfg.env.obs_type, tuple(image.shape))

        policy_config = RemotePolicyConfig(
            policy_type=self.cfg.policy.type,
            pretrained_name_or_path=str(self.cfg.policy.pretrained_path),
            lerobot_features=self.lerobot_features,
            actions_per_chunk=self.cfg.actions_per_chunk,
            device=self.cfg.policy.device or "cpu",
            rename_map=_make_rename_map(self.cfg.policy),
        )
        self.stub.SendPolicyInstructions(services_pb2.PolicySetup(data=pickle.dumps(policy_config)))
        self.policy_setup_sent = True

        self.logger.info(
            "Sent policy setup | "
            f"policy_type={policy_config.policy_type} | "
            f"actions_per_chunk={policy_config.actions_per_chunk} | "
            f"device={policy_config.device}"
        )

    def _flatten_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        flattened: dict[str, Any] = {
            "top": observation["pixels"],
            "task": self._current_task_description,
        }
        if "agent_pos" in observation:
            agent_pos = np.asarray(observation["agent_pos"], dtype=np.float32)
            for idx, value in enumerate(agent_pos):
                flattened[f"agent_pos_{idx}"] = float(value)
        return flattened

    def _aggregate_action_queues(self, incoming_actions: list[TimedAction]):
        future_action_queue: Queue[TimedAction] = Queue()
        with self.action_queue_lock:
            current_queue = {action.get_timestep(): action for action in self.action_queue.queue}

        with self.latest_action_lock:
            latest_action = self.latest_action

        for incoming in incoming_actions:
            timestep = incoming.get_timestep()
            if timestep <= latest_action:
                continue

            if timestep not in current_queue:
                future_action_queue.put(incoming)
                continue

            existing = current_queue[timestep]
            future_action_queue.put(
                TimedAction(
                    timestamp=incoming.get_timestamp(),
                    timestep=timestep,
                    action=self.cfg.aggregate_fn(existing.get_action(), incoming.get_action()),
                )
            )

        incoming_timesteps = {action.get_timestep() for action in incoming_actions}
        for timestep, existing in sorted(current_queue.items()):
            if timestep <= latest_action or timestep in incoming_timesteps:
                continue
            future_action_queue.put(existing)

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self):
        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue

                timed_actions: list[TimedAction] = pickle.loads(actions_chunk.data)  # nosec
                if not timed_actions:
                    continue

                self._aggregate_action_queues(timed_actions)
                self.must_go.set()
            except grpc.RpcError as err:
                if self.running:
                    self.logger.error(f"Error receiving actions: {err}")
                return

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
        chunk_size = max(self.action_chunk_size, 1)
        return queue_size / chunk_size <= self.cfg.chunk_size_threshold

    def _send_observation(self, observation: dict[str, Any]):
        with self.latest_action_lock:
            timestep = max(self.latest_action, 0)

        with self.action_queue_lock:
            must_go = self.must_go.is_set() and self.action_queue.empty()

        timed_observation = TimedObservation(
            timestamp=time.time(),
            timestep=timestep,
            observation=self._flatten_observation(observation),
            must_go=must_go,
        )

        observation_bytes = pickle.dumps(timed_observation)
        observation_iterator = send_bytes_in_chunks(
            observation_bytes,
            services_pb2.Observation,
            log_prefix="[CLIENT] Observation",
            silent=True,
        )
        self.stub.SendObservations(observation_iterator)

        if must_go:
            self.must_go.clear()

        fps_metrics = self.fps_tracker.calculate_fps_metrics(timed_observation.timestamp)
        self.logger.debug(
            f"Sent observation #{timed_observation.get_timestep()} | "
            f"avg_fps={fps_metrics['avg_fps']:.2f} | target={fps_metrics['target_fps']:.2f}"
        )

    def _pop_action(self) -> torch.Tensor | None:
        with self.action_queue_lock:
            if self.action_queue.empty():
                self.action_queue_size.append(0)
                return None

            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        return timed_action.get_action()

    def _run_episode(self, env: SingleMetaworldEnv, episode_seed: int) -> dict[str, Any]:
        self._reset_episode_state()
        observation, info = env.reset(seed=episode_seed)
        self._ensure_policy_setup(observation)

        reward_sum = 0.0
        success = False
        steps = 0
        max_steps = self.cfg.env.episode_length
        dt = 1 / self.cfg.env.fps

        while self.running and steps < max_steps:
            loop_start = time.perf_counter()

            if self._ready_to_send_observation():
                self._send_observation(observation)

            action = self._pop_action()
            if action is not None:
                action_np = action.detach().cpu().numpy()
                observation, reward, terminated, truncated, info = env.step(action_np)
                reward_sum += float(reward)
                success = success or bool(info.get("is_success", False))
                steps += 1
                if terminated or truncated:
                    break

            time.sleep(max(0.0, dt - (time.perf_counter() - loop_start)))

        return {
            "task": self._current_task_name,
            "seed": episode_seed,
            "steps": steps,
            "reward_sum": reward_sum,
            "success": success,
        }

    def evaluate(self) -> list[dict[str, Any]]:
        self.start()
        action_receiver_thread = threading.Thread(target=self.receive_actions, daemon=True)
        action_receiver_thread.start()

        results: list[dict[str, Any]] = []
        task_names = _expand_metaworld_tasks(self.cfg.env.task)

        try:
            for task_name in task_names:
                self._current_task_name = task_name
                self._current_task_description = TASK_DESCRIPTIONS.get(task_name, task_name)

                env = SingleMetaworldEnv(task=task_name, **self.cfg.env.gym_kwargs)
                try:
                    for episode_idx in range(self.cfg.n_episodes):
                        episode_seed = self.cfg.seed + episode_idx
                        result = self._run_episode(env, episode_seed)
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
