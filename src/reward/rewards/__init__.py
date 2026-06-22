"""Reward strategies: dispatch on ``args.reward`` to a concrete ``Reward``.

``make_reward`` builds the right strategy (BDE_IP / QED / pLogP); the
``MultiMolecules`` env delegates reward computation to ``Reward.compute``.
"""
from src.reward.rewards.base import Reward
from src.reward.rewards.bde_ip import BdeIpReward
from src.reward.rewards.qed import QedReward
from src.reward.rewards.plogp import PlogpReward


def make_reward(args, device, init_mols, bde_cache) -> Reward:
    """Construct the reward strategy selected by ``args.reward``."""
    reward = args.reward.lower()
    if reward == "bde_ip":
        return BdeIpReward(args, device, init_mols, bde_cache)
    elif reward == "qed":
        return QedReward(args)
    elif reward == "plogp":
        return PlogpReward(args)


__all__ = ["Reward", "make_reward"]
