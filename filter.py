import os
import shutil
from datetime import datetime

import math
import utils
import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import QED
import copy
import pdb
import sys
import pickle
import time
import remote_pdb
import random
import heapq
from similarity_filter import AntiOxidantDataSet
import sascorer
import pandas as pd
import csv



parser = argparse.ArgumentParser()

parser.add_argument('--experiment_path', type = str, default = './Experiments')
parser.add_argument('--trial', type=int, default=32, help="experiment trials")
parser.add_argument('--world_size', type=int, default=1000, help="number of ranks")
# parser.add_argument('--include_found_mols', action='store_true', help="if ture, it will also save the mols found in anti_dataset")

parser.add_argument('--trial_batch', type=int, default = None, help="experiment trials")
parser.add_argument('--num_init_mol_start', type=int, default = 0, help="Number of initial molecules")
parser.add_argument('--num_init_mol', type=int, default = None, help="Number of initial molecules")
parser.add_argument('--step', type=int, default = 1, help="")
parser.add_argument('--exclusive_mol', nargs="+", default=None, help="")



class MolFilter(object):
    """docstring for MolFilter"""
    def __init__(self, args):
        super(MolFilter, self).__init__()

        self.experiment_path = args.experiment_path
        self.trial = args.trial
        if args.trial_batch is None:
            self.out_trial = self.trial
        else :
            self.out_trial = args.trial_batch 

        # They are used for filtered.csv
        self.init_smiles = []
        self.init_bde = []
        self.init_ip = []
        self.generated_smiles = []
        self.predicted_bde = []
        self.predicted_ip = []
        self.reward = []
        self.similarity = []
        self.SA_score = []
        self.find_in_anti_oxidant = []
        self.rank = []
        self.id = []

        self.max_bde = 76
        self.min_ip = 145

        self.sim_tool = AntiOxidantDataSet('/home/l1062811/data/git/RL4-working/Data/anti.txt')
        self.df_bde = pd.read_csv('/home/l1062811/data/git/RL4-working/Data/anti-bde.csv', sep='\t')
        self.df_ip = pd.read_csv('/home/l1062811/data/git/RL4-working/Data/anti-ip.csv',sep='\t')

        self.df_bde.set_index('structure',inplace=True)
        self.df_ip.set_index('structure',inplace=True)

        # They are used for path.csv.
        self.path_rank = []
        self.path_id = []
        self.path_step = []
        self.path_smiles = []
        self.path_bde = []
        self.path_ip = []
        self.path_reward = []

    def find_init_bde(self, smiles):
        try:
            init_bde = self.df_bde.loc[smiles].BDE_DFT
            return init_bde
        except:
            return "" # not found
    def find_init_ip(self, smiles):
        try:
            init_ip = self.df_ip.loc[smiles].IP
            return init_ip
        except:
            return "" # not found

    def add(self, init_smiles, path, rewards, pbdes, pips, rank, id):
        is_good = self.add_gm(init_smiles, path[-1], rewards[-1], pbdes[-1], pips[-1], rank, id)
        if is_good:
            # save the generated path

            # save the initial mol, and its dft bde and ip
            self.path_rank.append(rank)
            self.path_id.append(id)
            self.path_step.append(0)
            self.path_smiles.append(init_smiles)

            self.path_bde.append(self.find_init_bde(init_smiles))
            self.path_ip.append(self.find_init_ip(init_smiles))
            self.path_reward.append(float(rewards[0]))

            for step, mol, pbde, pip, r in zip(range(len(pbdes)), path[1:], pbdes, pips, rewards[1:]):
                self.path_rank.append(rank)
                self.path_id.append(id)
                self.path_step.append(step + 1)
                self.path_smiles.append(Chem.MolToSmiles(mol))
                self.path_bde.append(float(pbde))
                self.path_ip.append(float(pip))
                self.path_reward.append(float(r))



    def add_gm(self, init_smiles, gm, final_reward, final_pbde, final_pip, rank, id):

        final_reward = float(final_reward)
        final_pbde = float(final_pbde)
        final_pip = float(final_pip)

        if final_pbde < self.max_bde and final_pip > self.min_ip:
            
            self.init_smiles.append(init_smiles)
            self.init_bde.append(self.find_init_bde(init_smiles))
            self.init_ip.append(self.find_init_ip(init_smiles))

            self.generated_smiles.append(Chem.MolToSmiles(gm))
            self.predicted_bde.append(final_pbde)
            self.predicted_ip.append(final_pip)
            self.reward.append(final_reward)
            self.similarity.append(self.sim_tool.sim(gm))
            self.SA_score.append(sascorer.calculateScore(gm))
            self.find_in_anti_oxidant.append(self.sim_tool.find(gm))
            self.rank.append(rank)
            self.id.append(id)

            return True

        return False
    def save(self):
        df_filtered = pd.DataFrame({
            'init_smiles':self.init_smiles,
            'init_bde':self.init_bde,
            'init_ip':self.init_ip,
            'generated_smiles':self.generated_smiles,
            'reward':self.reward,
            'predicted_bde':self.predicted_bde,
            'predicted_ip':self.predicted_ip,
            'similarity':self.similarity,
            'SA_score':self.SA_score,
            # 'relative_length':relative_length,
            # 'relative_natoms':relative_natoms,
            # 'relative_natoms_nbonds':relative_natoms_nbonds,
            'find_in_anti_oxidant':self.find_in_anti_oxidant,
            'mol_rank':self.rank,
            'mol_id':self.id
            })
        f_path = '{}/filtered/trial_{}_filtered.csv'.format(
            self.experiment_path, self.out_trial)
        print('{} mols are saved to {}'.format(len(self.init_smiles), f_path))
        df_filtered.to_csv(f_path, index = False, sep = ',')

        df_path = pd.DataFrame({
            'mol_rank':self.path_rank,
            'mol_id':self.path_id,
            'step':self.path_step,
            'smiles': self.path_smiles,
            'bde': self.path_bde,
            'ip': self.path_ip,
            'reward':self.path_reward
            })
        fp_path = '{}/filtered/trial_{}_path.csv'.format(
            self.experiment_path, self.out_trial)
        print('The mol generating paths are saved to {}'.format(fp_path))

        df_path.to_csv(fp_path, index = False, sep = ',')


def main(args):

    if args.trial_batch is None:
        mf = MolFilter(args)
        for rank in range(args.world_size):
            ba_path = '{}/trial_{}_rank_{}_best_actions_path.pickle'.format(args.experiment_path, args.trial, rank)
            init_smiles_path = '{}/trial_{}_rank_{}_init_mols.pickle'.format(args.experiment_path, args.trial, rank)
            try:
                with open(ba_path, 'rb') as f:
                    ps, rs, pbdes, pips = pickle.load(f)
                print(ba_path)
                with open(init_smiles_path, 'rb') as f:
                    init_s = pickle.load(f)
                for init, path, r, pbde, pip, id in zip(init_s, ps, rs, pbdes, pips, range(len(init_s))):
                    mf.add(init, path, r, pbde, pip, rank, id)

            except FileNotFoundError as e:
                # print('Not Found: {}'.format(ba_path))
                break
        mf.save()
    else :
        mf = MolFilter(args)
        for i in np.arange(args.num_init_mol_start, args.num_init_mol, args.step):
            if args.exclusive_mol is not None and str(i) in args.exclusive_mol:
                # print(i)
                continue
            t = '{}{}'.format(args.trial_batch, i)
            for rank in range(args.world_size):
                ba_path = '{}/trial_{}_rank_{}_best_actions_path.pickle'.format(args.experiment_path, t, rank)
                init_smiles_path = '{}/trial_{}_rank_{}_init_mols.pickle'.format(args.experiment_path, t, rank)
                try:
                    with open(ba_path, 'rb') as f:
                        ps, rs, pbdes, pips = pickle.load(f)
                    print(ba_path)
                    with open(init_smiles_path, 'rb') as f:
                        init_s = pickle.load(f)
                    for init, path, r, pbde, pip, id in zip(init_s, ps, rs, pbdes, pips, range(len(init_s))):
                        mf.add(init, path, r, pbde, pip, rank, id)

                except FileNotFoundError as e:
                    # print('Not Found: {}'.format(ba_path))
                    break
        mf.save()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)



