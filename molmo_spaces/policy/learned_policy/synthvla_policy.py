"""
Minimal InferencePolicy wrapper for SynthManipMolmoInferenceWrapper.

Wraps the external SynthManipMolmoInferenceWrapper from olmo.models.molmoact.agent to work
with the molmo-spaces simulation pipeline. Handles observation extraction,
action buffering, and action formatting.

Action chunking: The agent predicts action_horizon actions at once (e.g., 16).
We execute execute_horizon actions (e.g., 8) before re-querying the agent.
"""

import numpy as np

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.policy.base_policy import InferencePolicy


class SynthVLAPolicy(InferencePolicy):
    """Minimal InferencePolicy wrapper for SynthManipMolmoInferenceWrapper.

    Extracts observations from the simulator and calls the wrapped agent.
    Uses action buffering: predicts action_horizon actions, executes
    execute_horizon before refreshing the buffer.

    Loads the SynthManipMolmoInferenceWrapper from config.policy_config.checkpoint_path.
    """

    def __init__(
        self,
        config: MlSpacesExpConfig,
        task_type: str,
    ):
        super().__init__(config, task_type)
        self.camera_names = config.policy_config.camera_names
        self.action_move_group_names = config.policy_config.action_move_group_names
        self.action_spec = config.policy_config.action_spec
        self.action_horizon = config.policy_config.action_horizon
        self.execute_horizon = config.policy_config.execute_horizon

        self.action_buffer: list[dict[str, np.ndarray]] = []
        self.buffer_index = 0
        self.step_count = 0

        self.prepare_model()

    def prepare_model(self):
        """Load SynthManipMolmoInferenceWrapper from checkpoint specified in policy_config."""
        from olmo.models.molmoact.inference_wrapper import SynthManipMolmoInferenceWrapper

        checkpoint_path = self.config.policy_config.checkpoint_path
        # logger.info(f"Loading SynthManipMolmoInferenceWrapper from: {checkpoint_path}")
        self.agent = SynthManipMolmoInferenceWrapper(checkpoint_path=checkpoint_path)
        # logger.info("SynthManipMolmoInferenceWrapper loaded successfully")

    def reset(self):
        self.action_buffer = []
        self.buffer_index = 0
        self.step_count = 0

    def _populate_action_buffer(self, observation) -> None:
        """Call agent to get new action chunk and populate the buffer."""
        obs = observation[0] if isinstance(observation, list) else observation

        # Extract images in camera order
        images = []
        for cam_name in self.camera_names:
            if cam_name not in obs:
                raise KeyError(
                    f"Camera '{cam_name}' not in observation. Available: {list(obs.keys())}"
                )
            images.append(obs[cam_name])

        # Extract qpos state
        robot_state = obs["robot_state"]
        qpos_parts = []
        for group_name in self.action_move_group_names:
            qpos_parts.append(robot_state["qpos"][group_name])
        state = np.concatenate(qpos_parts).astype(np.float32)
        goal = self.task.get_task_description()

        # Call agent
        pred_actions = self.agent.get_action_chunk(
            images=images,
            task_description=goal,
            state=state,
        )

        # logger.info(f"Predicted action chunk: shape={pred_actions.shape}")

        # Convert to list of action dicts and store in buffer
        self.action_buffer = []
        for t in range(pred_actions.shape[0]):
            action = {}
            start_idx = 0
            for group_name in self.action_move_group_names:
                dim = self.action_spec[group_name]
                action[group_name] = pred_actions[t, start_idx : start_idx + dim]
                start_idx += dim
            self.action_buffer.append(action)

        self.buffer_index = 0
        # logger.info(f"Populated action buffer with {len(self.action_buffer)} actions")

    def get_action(self, observation) -> dict[str, np.ndarray]:
        """Return single action from buffer, refreshing when needed."""
        # Refresh buffer if empty or executed enough actions
        if self.buffer_index >= self.execute_horizon or not self.action_buffer:
            self._populate_action_buffer(observation)

        action = self.action_buffer[self.buffer_index]
        # logger.info(f"Executing action {self.buffer_index}/{len(self.action_buffer)} (refresh at {self.execute_horizon})")

        self.buffer_index += 1
        self.step_count += 1

        return action
