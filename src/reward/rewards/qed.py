"""QED reward: discounted weighted combine of QED and (negated) SA score."""
from src.reward.qed import qed_value
from src.reward.sa import sa_score


class QedReward:
    """docstring for QedReward"""

    bde_cache = None

    def __init__(self, config):
        self.discount_factor = config.reward.discount_factor
        weights = config.reward.qed
        self.qed_weight = weights.qed
        self.sa_weight = weights.sa

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
