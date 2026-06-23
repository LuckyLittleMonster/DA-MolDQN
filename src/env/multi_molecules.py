"""RL environment: ``MultiMolecules``.

A thin environment over the ``Molecule`` base. It owns env state (the OH
maintenance flags) and a ``bde_cache`` for hit-rate logging, then delegates all
reward computation to a ``Reward`` strategy (BDE_IP / QED / pLogP); see
``src.reward.rewards``.
"""
from src.environment import Molecule
from src.cache import LRUCache
from src.reward.rewards import make_reward


def count_OH(mol):
    OH_count = 0;
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetNumImplicitHs() > 0: # 8 for 'O'
            OH_count += 1
    return OH_count


class MultiMolecules(Molecule):
    """docstring for DistributedMolecules"""
    def __init__(self, config, device, **kwargs):
        super(MultiMolecules, self).__init__(
            config=config,
            **kwargs)
        self.device = device
        self.config = config

        # maintain_OH policy (typed): the env derives the per-mol cenv flag from
        # the policy + the init molecules. NO_LIMIT -> -2, AT_LEAST_ONE -> -1,
        # SAME_AS_INIT -> count_OH(init_mol_i) per-mol, EXACT -> count.
        maintain_oh = config.env.maintain_oh
        self.maintain_OH_flags = [
            maintain_oh.per_mol_flag(m, count_OH) for m in self.init_mols
        ]

        # bde_cache is created unconditionally so environment.bde_cache stays a
        # valid LRUCache for the trainer's hit-rate logging in ALL reward modes
        # (unused for qed/plogp). BdeIpReward shares this same object.
        self.bde_cache = LRUCache(config.env.lru_cache_capacity * len(self.init_mols))
        self.reward = make_reward(config, self.device, self.init_mols, self.bde_cache)
        self.init_rewards = self.find_reward(self.init_mols)

    def find_reward(self, molecules=None):
        if molecules is None:
            molecules = self.states
        return self.reward.compute(molecules, self.current_step, self.max_steps)
