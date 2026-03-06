from environment import Molecule
from utils import LRUCache


def count_OH(mol):
    OH_count = 0
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetTotalNumHs() > 0:
            OH_count += 1
    return OH_count


class MultiMolecules(Molecule):
    """Multi-molecule MDP environment with reward computation."""
    def __init__(self, args, device, **kwargs):
        super(MultiMolecules, self).__init__(
            args=args,
            **kwargs)

        self.bde_ip_reward = args.reward.type.lower() == 'bde_ip'
        self.discount_factor = args.discount_factor
        self.device = device

        # LRU caches (only for BDE_IP reward)
        if self.bde_ip_reward:
            lru_cap = args.reward.lru_cache_capacity * len(self.init_mols)
            self.bde_cache = LRUCache(lru_cap)
            self.ip_cache = LRUCache(lru_cap)
        else:
            self.bde_cache = None
            self.ip_cache = None

        # maintain_OH flags
        if args.maintain_OH is None:
            self.maintain_OH_flags = [-2 for _ in self.init_mols]
        elif args.maintain_OH == 'same':
            self.maintain_OH_flags = [count_OH(m) for m in self.init_mols]
        elif args.maintain_OH == 'exist':
            self.maintain_OH_flags = [-1 for _ in self.init_mols]
        else:
            self.maintain_OH_flags = [int(args.maintain_OH) for _ in self.init_mols]

        # Create reward calculator
        from rewards.factory import create_reward
        self.reward_calculator = create_reward(args.reward.type, device, args, self.init_mols)

        self.init_rewards = self.find_reward(self.init_mols)

    def find_reward(self, molecules=None):
        if molecules is None:
            molecules = self.states
        if self.bde_ip_reward:
            return self.reward_calculator.compute(
                molecules, bde_cache=self.bde_cache, ip_cache=self.ip_cache)
        else:
            return self.reward_calculator.compute(
                molecules, current_step=self.current_step)

    def find_reward_overlap(self, molecules=None, prefetch_fn=None, cleanup_fn=None):
        """find_reward with overlap: BDE||ETKDG (Level 1) + AIMNet||prefetch (Level 2).
        Returns (reward_dict, prefetch_result).
        """
        if molecules is None:
            molecules = self.states
        if self.bde_ip_reward:
            return self.reward_calculator.compute_overlap(
                molecules, prefetch_fn=prefetch_fn, cleanup_fn=cleanup_fn,
                bde_cache=self.bde_cache, ip_cache=self.ip_cache)
        else:
            if cleanup_fn:
                cleanup_fn()
            reward = self.find_reward(molecules)
            prefetch_result = prefetch_fn() if prefetch_fn else None
            return reward, prefetch_result
