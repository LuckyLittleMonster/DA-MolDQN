"""RL environment: ``MultiMolecules``.

A thin environment over the ``Molecule`` base. It owns env state (the OH
maintenance flags) and a ``bde_cache`` for hit-rate logging, then delegates all
reward computation to a ``Reward`` strategy (BDE_IP / QED / pLogP); see
``src.reward.rewards``.
"""
from src import config_defaults as hyp
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
    def __init__(self, args, device, **kwargs):
        super(MultiMolecules, self).__init__(
            args = args,
            **kwargs)
        self.device = device

        # parser.add_argument('--maintain_OH', type=str, default=None, help=
        #     "default: None or 'None': no limitation\n"
        #     "same: The number of OH bonds are always same to the initial molecules.\n"
        #     "exist: All molecules must have one or more OH bonds.\n"
        #     "n: all mols should have the n of OH bonds\n")

        # maintail_OH:
        #     -2: no limitation
        #     -1: at least 1 OH bond
        #     0 ~ N: has the number of OH bonds, it is the same as the initial mol

        if args.maintain_OH is None:
            self.maintain_OH_flags = [-2 for _ in self.init_mols]
        elif args.maintain_OH == 'same':
            self.maintain_OH_flags = [count_OH(m) for m in self.init_mols]
        elif args.maintain_OH == 'exist':
            self.maintain_OH_flags = [-1 for _ in self.init_mols]
        else:
            self.maintain_OH_flags = [int(args.maintain_OH) for _ in self.init_mols]

        # bde_cache is created unconditionally so environment.bde_cache stays a
        # valid LRUCache for the trainer's hit-rate logging in ALL reward modes
        # (unused for qed/plogp). BdeIpReward shares this same object.
        self.bde_cache = LRUCache(hyp.lru_cache_capacity * len(self.init_mols))
        self.reward = make_reward(args, self.device, self.init_mols, self.bde_cache)
        self.init_rewards = self.find_reward(self.init_mols)

    def find_reward(self, molecules=None):
        if molecules is None:
            molecules = self.states
        return self.reward.compute(molecules, self.current_step, self.max_steps)
