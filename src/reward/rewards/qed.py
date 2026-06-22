"""QED reward: discounted weighted combine of QED and (negated) SA score."""
from src.reward.qed import qed_value
from src.reward.sa import sa_score


class QedReward:
    """docstring for QedReward"""

    bde_cache = None

    def __init__(self, args):
        self.discount_factor = args.discount_factor
        self.qed_weight = 0.8
        self.sa_weight = 0.2
        if len(args.reward_weight) == 0:
            # use default weights
            pass
        elif len(args.reward_weight) == 1:
            # assume that the one value is bde weight, which is the same as main_multi.py
            self.qed_weight = args.reward_weight[0]
            self.sa_weight = 1.0 - self.qed_weight

        elif len(args.reward_weight) == 2:
            self.qed_weight = args.reward_weight[0]
            self.sa_weight = args.reward_weight[1]

    def compute(self, molecules, current_step, max_steps):
        rs = []
        qeds = []
        sas = []
        for molecule in molecules:
            qed = qed_value(molecule)
            qeds.append(qed)
            SA_score = sa_score(molecule)
            sas.append(SA_score)
            reward = (qed * self.qed_weight - self.sa_weight * SA_score) * self.discount_factor ** (max_steps-current_step)
            rs.append(reward)
        return {'reward': rs, 'QED':qeds, 'SA_score':sas}
