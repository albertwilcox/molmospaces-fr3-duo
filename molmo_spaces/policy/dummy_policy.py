from typing import TYPE_CHECKING

import numpy as np

from molmo_spaces.policy.base_policy import NONE_PHASE, BasePolicy
from molmo_spaces.tasks.task import BaseMujocoTask

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig


class DummyPolicy(BasePolicy):
    """
    A Dummy Policy that return null actions.
    """

    @property
    def type(self):
        return "dummy"

    def __init__(self, config: "MlSpacesExpConfig", task: BaseMujocoTask | None = None) -> None:
        self.config = config
        # Required attributes for sensors that expect a policy with target poses
        self.target_poses = {"grasp": np.eye(4)}
        self.current_phase = NONE_PHASE

    def reset(self):
        """
        Reset the policy state. No state to reset for DummyPolicy.
        """
        pass

    def get_action(self, obervation):
        """
        Dummy action to take based on the action space.

        Args:
            info: The current information about the task or environment (not used).

        Returns:
        """
        return dict()
