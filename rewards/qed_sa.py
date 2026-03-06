from rdkit.Chem import QED
from rdkit import RDConfig
import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from .base import RewardCalculator


class QEDReward(RewardCalculator):
    def __init__(self, args):
        self.discount_factor = args.discount_factor
        self.max_steps = args.max_steps_per_episode
        self.qed_weight = 0.8
        self.sa_weight = 0.2

        rw = list(args.reward.reward_weight)
        if len(rw) == 1:
            self.qed_weight = rw[0]
            self.sa_weight = 1.0 - self.qed_weight
        elif len(rw) >= 2:
            self.qed_weight = rw[0]
            self.sa_weight = rw[1]

    @property
    def reward_keys(self):
        return ['QED', 'SA_score']

    def compute(self, molecules, current_step=0, **kwargs):
        rs = []
        qeds = []
        sas = []
        for molecule in molecules:
            if molecule is None:
                rs.append(-1000)
                qeds.append(0.0)
                sas.append(10.0)
                continue
            qed = QED.qed(molecule)
            qeds.append(qed)
            SA_score = sascorer.calculateScore(molecule)
            sas.append(SA_score)
            reward = (qed * self.qed_weight - self.sa_weight * SA_score) * \
                     self.discount_factor ** (self.max_steps - current_step)
            rs.append(reward)
        return {'reward': rs, 'QED': qeds, 'SA_score': sas}
