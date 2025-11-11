# view a generated path
import sys
import rdkit
import pickle
import re
import os
import numpy as np

def sort(l):
    l.sort()
    l.sort(key = len) # stable sort by default
    return l

def test():
    print('hello')

class Results(object):
    def __init__(self, path):
        super(Results, self).__init__()
        self.path = path
        self.p = pickle.load(open(path, 'rb'))
        # print(self.p.keys())
        self.experiment = os.path.basename(path).split('.')[0]
        self.max_bde = 76
        self.min_ip = 145
        
    def __getitem__(self, key):
        return self.p[key]
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

    def get_rewards(self, regex = None):
        if regex is None:
            regex = '.*_episodes'
        paths = self.match(regex)
        print(paths)
        # ns = np.array([self[p][0] for p in paths])
        # print(ns.shape)
        return ns

if __name__ == "__main__":
    path = '../Experiments/ZincQEDFT_.pickle'
    r = Results(path)
    print(r.match('.*_path$'))