# view a generated path
import sys
import rdkit
import pickle
import gzip
import re
import os
import numpy as np
import pprint
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import RDConfig
import os
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer


def sort(l):
    l.sort()
    l.sort(key = len) # stable sort by default
    return l

def penalized_logP(mol):
    """
    Logarithm of partition coefficient, penalized by cycle length and synthetic accessibility.

        
    The code are from torchdrug torchdrug/metrics/metric.py
    https://github.com/DeepGraphLearning/torchdrug/blob/6066fbd82360abb5f270cba1eca560af01b8cc90/torchdrug/metrics/metric.py#L84
    
    This version also supports smiles string and should return the exactly same plogp values as torchdrug.

    Parameters:
        # pred (PackedMolecule): molecules to evaluate
        smiles_list: smiles strings to evaluate


    """
    # statistics from ZINC250k
    logp_mean = 2.4570953396190123
    logp_std = 1.434324401111988
    sa_mean = 3.0525811293166134
    sa_std = 0.8335207024513095
    cycle_mean = 0.0485696876403053
    cycle_std = 0.2860212110245455

    # mol = Chem.MolFromSmiles(smiles)
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
    return score


class Results(object):
    def __init__(self, path):
        super(Results, self).__init__()
        self.path = path
        self._p = None
        self.experiment = os.path.basename(path).split('.')[0]
        self.max_bde = 76
        self.min_ip = 145
        dir_path = os.path.dirname(self.path)
        file_name = os.path.basename(self.path)
        self.cache_path = os.path.join(dir_path, f'.{file_name}_cache.pickle')
        self.cache = {}
        
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
    
    def store(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
        
    def __getitem__(self, key):
        return self.p[key]
    
    @property
    def p(self):
        if self._p is None:
            if os.path.exists(self.path):
                with open(path, 'rb') as f:
                    self._p = pickle.load(f)
            gz_path = self.path + '.gz'
            if os.path.exists(gz_path):
                with gzip.open(gz_path, 'rb') as f:
                    self._p = pickle.load(f)
        return self._p
        
    def match(self, reg, sorted = True):
        # return a list of keys that match the regex
        l = [k for k in self.p.keys() if re.match(reg, str(k))]
        if sorted:
            l = sort(l)
        return l
    def get_pbde_pip_legacy(self, regex = None):
        pbdes, pips = [], []
        if regex is None:
            regex = '.*_best_actions_path$'    
        paths = self.match(regex)
        # print(paths)
        for p in paths:
            # print(self[p])
            pbdes.append(self[p][2])
            pips.append(self[p][3])
        pbdes = np.array(pbdes).reshape(-1, 10)
        pips = np.array(pips).reshape(-1, 10)
        # print(pbdes.shape, pips.shape)
        return pbdes[:,-1], pips[:,-1]

    def calc_ofr_legacy(self, regex = None):
        pbdes, pips = self.get_pbde_pip_legacy(regex)
        # print(pbdes.shape, pips.shape)
        n_init_mols = float(len(pbdes))
        n_valid_mols = 0
        for bde, ip in zip(pbdes, pips):
            if bde < self.max_bde and ip > self.min_ip:
                n_valid_mols += 1
        # print(n_valid_mols, n_init_mols)
        return 1 - n_valid_mols / n_init_mols
    def get_episode_time_legacy(self, regex = None):
        if regex is None:
            regex = '.*_episode_time'    
        paths = self.match(regex)
        ns = np.array([self[p] for p in paths])
        ns = np.mean(ns, axis=0)
        # print(ns.shape)
        return ns
    def get_episode_lru_cache_hit_rate_legacy(self, regex = None):
        if regex is None:
            regex = '.*episode_lru_cache_hit_rate'
        paths = self.match(regex)
        ns = np.array([self[p] for p in paths])
        ns = np.mean(ns, axis=0)
        # print(ns.shape)
        return ns

    def get_computation_time_legacy(self, regex = None):
        if regex is None:
            regex = '.*computation_time'
        paths = self.match(regex)
        ns = np.array([self[p][0] for p in paths])
        # print(ns.shape)
        return ns
    def clip_rewards(self, rewards: dict, flatten = True):
        for k, v in rewards.items():
            rewards[k] = np.array(v).flatten()
            if k.lower() == 'reward':
                if 'plogp' not in rewards:
                    rewards[k] = np.clip(rewards[k], 0, 3.5)
        return rewards
        
    def get_rewards(self, regex = None, last_n_episodes = 1, use_cache = True):
        if use_cache and 'rewards' in self.cache:
            return self.cache['rewards']
        if regex is None:
            regex = '.*_episodes'
        paths = self.match(regex)
        rewards = {}
        for p in paths:
            # print(self[p]['rewards'])
            for k, v in self[p]['rewards'].items():
                if k not in rewards:
                    rewards[k] = [v[-last_n_episodes:]]
                else:
                    rewards[k].append(v[-last_n_episodes:])
        rewards = self.clip_rewards(rewards)
        self.cache['rewards'] = rewards
        self.store()
        return rewards
    def get_initial_rewards(self, regex = None, use_cache = True):
        if use_cache and 'initial_rewards' in self.cache:
            return self.cache['initial_rewards']
        if regex is None:
            regex = '.*_path'
        paths = self.match(regex)
        initial_rewards = {}
        for p in paths:
            # pprint.pp(self[p]['last'])
            # print(p)
            # print("--------------------------------")
            _, reward_path = self[p]['last'][-1]
            for k, v in reward_path.items():
                init_rs = []
                for val in v:
                    init_rs.append(val[0])
                if k not in initial_rewards:
                    initial_rewards[k] = [init_rs]
                else:
                    initial_rewards[k].append(init_rs)
        
        initial_rewards = self.clip_rewards(initial_rewards)
        self.cache['initial_rewards'] = initial_rewards
        self.store()
        return initial_rewards

class GNNResults(object):
    def __init__(self, path):
        super(GNNResults, self).__init__()
        self.path = path
        self._mols = None
        self.cache = {}
        self.cache_path = os.path.join(os.path.dirname(self.path), f'.{os.path.basename(self.path)}_cache.pickle')
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
    
    def store(self):
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)

    @property
    def mols(self):
        if self._mols is None:
            with open(self.path, 'r') as file:
                ss = file.read().splitlines()
                self._mols = []
                for s in ss:
                    try:
                        m = Chem.MolFromSmiles(s)
                        self._mols.append(m)
                    except Exception as e:
                        # print(e)
                        pass
        return self._mols
    
    def calc_QED(self, use_cache = True, sort = True):
        if use_cache and 'qeds' in self.cache:
            return self.cache['qeds']
        qeds = [QED.qed(mol) for mol in self.mols]
        qeds = np.array(qeds)
        if sort:
            qeds = np.sort(qeds)
        self.cache['qeds'] = qeds
        self.store()
        return qeds

    def calc_PlogP(self, use_cache = True, sort = True):
        if use_cache and 'plogps' in self.cache:
            return self.cache['plogps']
        plogps = [penalized_logP(mol) for mol in self.mols]
        plogps = np.array(plogps)
        if sort:
            plogps = np.sort(plogps)
        self.cache['plogps'] = plogps
        self.store()
        return plogps

    def calc_SA(self, use_cache = True, sort = True):
        if use_cache and 'sas' in self.cache:
            return self.cache['sas']
        sas = [sascorer.calculateScore(mol) for mol in self.mols]
        sas = np.array(sas)
        if sort:
            sas = np.sort(sas)
        self.cache['sas'] = sas
        self.store()
        return sas

if __name__ == "__main__":
    if len(sys.argv) != 2:
        exp_path = '../Experiments/ZincQEDFT_.pickle'
    else:
        exp_path = sys.argv[1]
    
    if exp_path.endswith('.pickle'):
        results = Results(exp_path)
        rewards = results.get_rewards(use_cache=False)
        print(rewards['reward'].shape)
        initial_rewards = results.get_initial_rewards(use_cache=False)
        print(initial_rewards['reward'].shape)
    elif exp_path.endswith('.txt'):
        results = GNNResults(exp_path)
        if 'qed' in exp_path.lower():
            qeds = results.calc_QED(use_cache=False)
            print(qeds)
        elif 'plogp' in exp_path.lower():
            plogps = results.calc_PlogP(use_cache=False)
            print(plogps)