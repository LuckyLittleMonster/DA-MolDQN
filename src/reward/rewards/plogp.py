"""pLogP reward: discounted penalized-logP score."""
from src.reward.plogp import plogp_value


class PlogpReward:
    """docstring for PlogpReward"""

    bde_cache = None

    def __init__(self, args):
        self.discount_factor = args.discount_factor
        # self.plogp_weight = 1.0

    def compute(self, molecules, current_step, max_steps):
        rs = []
        sims = []
        for mol in molecules:
            score = plogp_value(mol)
            sims.append(-1)
            reward = score * self.discount_factor ** (max_steps-current_step)
            rs.append(reward)
        return {'reward': rs, 'plogp':rs, 'sim': sims}
