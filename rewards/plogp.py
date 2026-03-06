from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import RDConfig
import os
import sys

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from .base import RewardCalculator


class PLogPReward(RewardCalculator):
    def __init__(self, args):
        self.discount_factor = args.discount_factor
        self.max_steps = args.max_steps_per_episode

    @property
    def reward_keys(self):
        return ['plogp', 'sim']

    def compute(self, molecules, current_step=0, **kwargs):
        rs = []
        plogps = []
        sims = []

        logp_mean = 2.4570953396190123
        logp_std = 1.434324401111988
        sa_mean = 3.0525811293166134
        sa_std = 0.8335207024513095
        cycle_mean = 0.0485696876403053
        cycle_std = 0.2860212110245455

        for mol in molecules:
            if mol is None:
                plogps.append(-30)
                rs.append(-30)
                sims.append(-1)
                continue
            try:
                mol.UpdatePropertyCache()
                cycles = Chem.GetSymmSSSR(mol)
                if cycles:
                    max_cycle = max([len(cycle) for cycle in cycles])
                    cycle = max(0, max_cycle - 6)
                else:
                    cycle = 0
                logp = Descriptors.MolLogP(mol)
                sa = sascorer.calculateScore(mol)
                logp = (logp - logp_mean) / logp_std
                sa = (sa - sa_mean) / sa_std
                cycle = (cycle - cycle_mean) / cycle_std
                score = logp - sa - cycle
            except Chem.AtomValenceException:
                score = -30
            plogps.append(score)
            sims.append(-1)
            reward = score * self.discount_factor ** (self.max_steps - current_step)
            rs.append(reward)
        return {'reward': rs, 'plogp': plogps, 'sim': sims}
