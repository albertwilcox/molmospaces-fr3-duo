import random
from threading import Semaphore

import numpy as np


def subsample_trajectory(
    trajectory, first_useful=1, last_useful=-2, skip_rate=3, apply_jitter=True
):
    # E.g.
    # state 0 may be useless (initial state for task) -> first_useful = 1
    # state -1 may be useless (task is solved), and
    # state -2 may be very valuable (close to succeeding) -> last_useful=-2

    traj_len = len(trajectory)

    # randomly shift first_useful loc if apply_jitter
    jitter = random.randint(0, skip_rate - 1) if apply_jitter and skip_rate > 1 else 0

    # indices up to before the last_useful one
    idxs = list(range(first_useful + jitter, traj_len + last_useful, skip_rate))

    # always add the last_useful index
    idxs.append(traj_len + last_useful)

    return dict(trajectory=[trajectory[it] for it in idxs], times=idxs)


# From https://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf A-Res algorithm


class StateBuffer:
    def __init__(self, max_len: int = 1_000, target_success_count: int = 10) -> None:
        self.max_len = max_len
        self.target_success_count = target_success_count

        self.insertion_counter = 0
        self.index_to_trajectory = {}

        self.weights = np.empty(shape=(self.max_len,), dtype=np.float64)
        self.key_bases = np.empty(shape=(self.max_len,), dtype=np.float64)
        self.keys = np.empty(shape=(self.max_len,), dtype=np.float64)
        self.smallest_key_index = -1

        self.mutex = Semaphore()

    @property
    def num_entries(self):
        return min(self.max_len, self.insertion_counter)

    def _get_trajectory_weight(self, reward, success_counts, eps=1e-10):
        # weigh trajectory by the disability to succeed from a distant time step
        steps_to_end = np.array(list(reversed(range(1, len(success_counts) + 1))))
        return reward * max(
            np.sum(np.maximum(self.target_success_count - success_counts, 0) * steps_to_end)
            / np.sum(self.target_success_count * steps_to_end),
            eps,
        )

    def _get_state_weights(self, success_counts, eps=1e-5):
        # weigh state by the disability to succeed from a near time step
        steps_to_end = np.array(list(reversed(range(1, len(success_counts) + 1))))
        raw_weights = np.maximum(self.target_success_count - success_counts, eps) / steps_to_end
        return raw_weights / np.sum(raw_weights)

    def _sample_state_index(self, state_weights):
        return np.random.choice(len(state_weights), p=state_weights)

    def _sample_trajectory_index(self):
        return np.random.choice(
            self.num_entries,
            p=self.weights[: self.num_entries] / np.sum(self.weights[: self.num_entries]),
        )

    def _make_key(self, base, weight):
        return np.power(base, 1.0 / weight)

    def _update_success_counts(self, trajectory_index, state_index, delta: int) -> None:
        # Assumes caller has acquired mutex

        trajectory_data = self.index_to_trajectory[trajectory_index]

        # Update success counts
        assert delta in [-1, 1]
        success_counts = trajectory_data["success_counts"]
        success_counts[state_index] += delta

        # Update trajectory weight
        new_weight = self._get_trajectory_weight(trajectory_data["reward"], success_counts)
        self.weights[trajectory_index] = new_weight

        # Update trajectory key
        new_key = self._make_key(self.key_bases[trajectory_index], new_weight)
        self.keys[trajectory_index] = new_key

        # Update smallest key index
        if new_key < self.keys[self.smallest_key_index]:
            self.smallest_key_index = trajectory_index

    def update_failure(self, trajectory_dict) -> None:
        self.mutex.acquire()
        try:
            # Check if used trajectory is still in reservoir
            if (
                trajectory_dict["trajectory_insertion_counter"]
                == self.index_to_trajectory[trajectory_dict["trajectory_index"]][
                    "insertion_counter"
                ]
            ):
                self._update_success_counts(
                    trajectory_dict["trajectory_index"], trajectory_dict["state_index"], -1
                )
        finally:
            self.mutex.release()

    def sample_state(self):
        self.mutex.acquire()
        try:
            if self.num_entries < 1:
                return None

            trajectory_index = self._sample_trajectory_index()
            trajectory_data = self.index_to_trajectory[trajectory_index]

            state_weights = self._get_state_weights(trajectory_data["success_counts"])
            state_index = self._sample_state_index(state_weights)

            # We'll pretend the next run from the chosen state will be successful, so that
            # other tasks can sample from the same trajectory with updated statistics
            self._update_success_counts(trajectory_index, state_index, 1)

            trajectory_data["resampled_count"] += 1

            return dict(
                trajectory_index=trajectory_index,
                trajectory_insertion_counter=trajectory_data["insertion_counter"],
                state_index=state_index,
                state=trajectory_data["states"][state_index],
                task_info=trajectory_data["task_info"],
            )
        finally:
            self.mutex.release()

    def insert(self, trajectory, reward, task_info) -> None:
        if reward <= 0:  # TODO assuming the reward can be > 0 for any task
            return

        self.mutex.acquire()
        try:
            success_counts = np.zeros((len(trajectory),), dtype=np.int64)
            new_key_base = random.random()
            new_weight = self._get_trajectory_weight(reward, success_counts)
            new_key = self._make_key(new_key_base, new_weight)

            if self.insertion_counter < self.max_len:
                trajectory_index = self.insertion_counter
                if self.smallest_key_index < 0 or new_key < self.keys[self.smallest_key_index]:
                    self.smallest_key_index = trajectory_index
                self.keys[trajectory_index] = new_key

            elif new_key > self.keys[self.smallest_key_index]:
                trajectory_index = self.smallest_key_index
                self.keys[trajectory_index] = new_key
                self.smallest_key_index = np.argmin(self.keys)

            else:
                trajectory_index = -1

            if trajectory_index >= 0:
                self.weights[trajectory_index] = new_weight
                self.key_bases[trajectory_index] = new_key_base

                self.index_to_trajectory[trajectory_index] = dict(
                    states=trajectory,
                    reward=reward,
                    success_counts=success_counts,
                    insertion_counter=self.insertion_counter,
                    task_info=task_info,
                    resampled_count=0,
                )

                self.insertion_counter += 1

        finally:
            self.mutex.release()


if __name__ == "__main__":

    def main() -> None:
        buffer = StateBuffer(max_len=3, target_success_count=2)
        for it in range(100):
            if buffer.num_entries > 0 and random.random() > 0.1:
                info = buffer.sample_state()
                if random.random() < 0.25:
                    print(
                        buffer.keys[: buffer.num_entries], buffer.insertion_counter, "before fail"
                    )
                    buffer.update_failure(info)
            else:
                info = {"trajectory_index": -1}
                traj = subsample_trajectory([1] * random.choice([16, 32]))["trajectory"]
                buffer.insert(traj, random.random() * 20 + 0.01, {})

            if it % 1 == 0:
                print(
                    buffer.keys[: buffer.num_entries],
                    buffer.insertion_counter,
                    info["trajectory_index"],
                )

        print(buffer.keys[: buffer.num_entries], buffer.insertion_counter, "keys")
        print(buffer.weights[: buffer.num_entries], buffer.insertion_counter, "weights")
        print(buffer.key_bases[: buffer.num_entries], buffer.insertion_counter, "key bases")

        print("DONE")

    main()
