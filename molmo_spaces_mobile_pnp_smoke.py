"""Standalone smoke test for the mobile pick-and-place-with-navigation expert.

Run (from the datagen venv, with MuJoCo env vars exported):

    export MLSPACES_CACHE_DIR=/mnt/disk/msr2/molmo-spaces-resources MUJOCO_GL=egl
    python molmo_spaces_mobile_pnp_smoke.py

It builds :class:`MobileFrankaPickAndPlaceConfig`, samples a task on a house,
instantiates :class:`MobilePickAndPlaceStateMachinePolicy`, runs the episode loop
and reports per-phase transitions plus final ``judge_success()``. It iterates over
houses / seeds until it achieves a success (or exhausts the budget).
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback

import numpy as np

from molmo_spaces.configs.mobile_pick_and_place_config import MobileFrankaPickAndPlaceConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("mobile_pnp_smoke")


def run_episode(config, sampler, house_id: int, seed: int) -> tuple[bool, str]:
    """Sample and run one episode. Returns (success, note)."""
    np.random.seed(seed)
    try:
        task = sampler.sample_task(house_index=house_id)
    except Exception as e:  # noqa: BLE001
        return False, f"sample_task raised: {type(e).__name__}: {e}"
    if task is None:
        return False, "sample_task returned None"

    policy = config.policy_config.policy_cls(config, task)
    task.register_policy(policy)

    obs, _ = task.reset()

    transitions: list[str] = []
    last_phase = policy.get_phase()
    transitions.append(last_phase)

    max_steps = config.task_horizon or 1500
    step = 0
    policy_done = False
    while step < max_steps:
        action = policy.get_action(obs)
        phase = policy.get_phase()
        if phase != last_phase:
            transitions.append(phase)
            log.info(f"[house {house_id} seed {seed}] phase -> {phase} at step {step}")
            last_phase = phase

        if action is None:
            policy_done = True
            break

        obs, _, _, _, infos = task.step(action)
        step += 1

        if action.get("done") or bool(np.all(task.is_done())):
            policy_done = True
            break

    success = bool(task.judge_success())
    info0 = task.get_info()[0]
    note = (
        f"phases={'->'.join(transitions)} steps={step} policy_done={policy_done} "
        f"supported={info0.get('supported_by_receptacle')} "
        f"robot_contact={info0.get('robot_contact')} "
        f"recep_disp={np.round(np.asarray(info0.get('receptacle_pos_displacement', 0.0)), 3)}"
    )
    return success, note


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--houses", type=int, nargs="+", default=[0, 2, 3, 14, 1, 4, 5, 6, 7, 8])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    config = MobileFrankaPickAndPlaceConfig()
    log.info(f"Built config: task_type={config.task_type} tag={config.tag}")
    log.info(f"policy_cls={config.policy_config.policy_cls.__name__}")

    sampler = config.task_sampler_config.task_sampler_class(config)

    any_success = False
    for house_id in args.houses:
        for seed in args.seeds:
            log.info(f"=== Trying house {house_id}, seed {seed} ===")
            try:
                success, note = run_episode(config, sampler, house_id, seed)
            except Exception as e:  # noqa: BLE001
                log.error(f"[house {house_id} seed {seed}] episode crashed: {e}")
                traceback.print_exc()
                continue
            log.info(f"[house {house_id} seed {seed}] success={success} | {note}")
            if success:
                any_success = True
                print("\n" + "=" * 70)
                print(f"✅ SUCCESS on house_id={house_id} seed={seed}")
                print(f"   judge_success() == True")
                print(f"   {note}")
                print("=" * 70)
                return 0

    if not any_success:
        print("\n❌ No success across the tried houses/seeds.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
