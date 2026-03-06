from .bde_ip import BDEIPReward
from .qed_sa import QEDReward
from .plogp import PLogPReward


def create_reward(reward_type, device, args, init_mols=None):
    reward_type = reward_type.lower()
    if reward_type == 'bde_ip':
        return BDEIPReward(device, args, init_mols)
    elif reward_type == 'qed':
        return QEDReward(args)
    elif reward_type == 'plogp':
        return PLogPReward(args)
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
