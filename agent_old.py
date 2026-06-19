import torch
import torch.nn as nn
import numpy as np
import torch.optim as opt
import utils
import hyp
from dqn import MolDQN, make_transformer_model, GraphTransformer
from rdkit import Chem
from rdkit.Chem import QED
from rdkit.Chem import AllChem
from rdkit import RDConfig
import os
# disable opt warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from rdkit.Chem import Descriptors
import sys
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from sklearn import preprocessing
from environment import Molecule
from rdkit import DataStructs
import pandas as pd
import pickle
import pdb
import time
import csv
from eval import EnsembleCalculator, load_models, to_numpy


# TF may use all memory 
# https://wiki.ncsa.illinois.edu/display/ISL20/Managing+GPU+memory+when+using+Tensorflow+and+Pytorch
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

# disable opt warning
# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

from alfabet import model as bde_model
from utils import LRUCache


REPLAY_BUFFER_CAPACITY = hyp.replay_buffer_size

def ev2kcal_per_mol(ev):
    return ev * 23.0609

def calc_react_idx(data):
    ip = data['energy'][0] - data['energy'][1]
    ea = data['energy'][1] - data['energy'][2]
    f_el = data['charges'][1] - data['charges'][0]
    f_nuc = data['charges'][2] - data['charges'][1]
    chi = 0.5 * (ip + ea)
    eta = 0.5 * (ip - ea)
    omega = (chi ** 2) / (2 * eta)
    f_rad = 0.5 * (f_el + f_nuc)
    _omega = np.expand_dims(omega, axis=-1)
    omega_el = f_el * _omega
    omega_nuc = f_nuc * _omega
    omega_rad = f_rad * _omega
    return dict(ip=ip, ea=ea, f_el=f_el, f_nuc=f_nuc, f_rad=f_rad,
                chi=chi, eta=eta, omega=omega,
                omega_el=omega_el, omega_nuc=omega_nuc, omega_rad=omega_rad)


def get_scaler(path, real_col_id = 1):
    real = []
    # global real
    with open (path) as f:
        s = csv.reader(f, delimiter="\t");
        next(s)
        for r in s:
            if r[real_col_id] != '':
                real.append([float(r[real_col_id])])
    return preprocessing.MinMaxScaler().fit(real)

def count_OH(mol):
    OH_count = 0;
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 8 and atom.GetNumImplicitHs() > 0: # 8 for 'O'
            OH_count += 1
    return OH_count


class AimnetNseModel():
    """The original model is not pickable and can't be used with spawn. This class is a warpper of EnsembleCalculator."""
    def __init__(self, path, device):
        self.path = path
        self.device = device
        self.model = load_models([path]).to(device)

    def __setstate__(self, state):
        self.path = state['path']
        self.device = state['device']
        self.model = load_models([self.path]).to(self.device)

    def __getstate__(self):
        return dict(path = self.path, device = self.device)

class MultiMolecules(Molecule):
    """docstring for DistributedMolecules"""
    def __init__(self, args, device, **kwargs):
        super(MultiMolecules, self).__init__(
            args = args, 
            **kwargs)

        self.lru_cache = LRUCache(hyp.lru_cahce_capacity * len(self.init_mols))
        self.lru_cache_ic = LRUCache(hyp.lru_cahce_capacity * len(self.init_mols)) # invalid 3d conformers
        self.discount_factor = args.discount_factor
        self.device = device

        self.bde_ip_reward = False
        self.qed_reward = False
        self.plogp_reward = False

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

        # if args.maintain_OH is None:
        #     self.maintain_OH_flags = [count_OH(m) for m in self.init_mols]
        # else :
        #     self.maintain_OH_flags = [args.maintain_OH for _ in self.init_mols]

        if args.reward.lower() == "BDE_IP".lower():
            self.bde_ip_reward = True
            self.bde_factor = hyp.bde_factor
            self.ip_factor = hyp.ip_factor

            self.bed_weight = 0.8
            self.ip_weight = 0.2
            self.rrab_weight = 0.5
            if len(args.reward_weight) == 0:
                # use default weights
                pass
            elif len(args.reward_weight) == 1:
                # assume that the one value is bde weight, which is the same as main_multi.py
                self.bed_weight = args.reward_weight[0]
                self.ip_weight = 1.0 - self.bed_weight
                
            elif len(args.reward_weight) == 2:
                self.bed_weight = args.reward_weight[0]
                self.ip_weight = args.reward_weight[1]
            else :
                self.bed_weight = args.reward_weight[0]
                self.ip_weight = args.reward_weight[1]
                self.rrab_weight = args.reward_weight[2]

            self.bde_scaler = get_scaler('./Data/anti-bde.csv')
            self.ip_scaler = get_scaler('./Data/anti-ip.csv')
            self.ip_model_path = [
                'aimnetnse-models/aimnet-nse-cv0.jpt',
                'aimnetnse-models/aimnet-nse-cv1.jpt',
                'aimnetnse-models/aimnet-nse-cv2.jpt',
                'aimnetnse-models/aimnet-nse-cv3.jpt',
                'aimnetnse-models/aimnet-nse-cv4.jpt']
            self.ip_model = [AimnetNseModel(ipmp, self.device) for ipmp in self.ip_model_path]
            self.init_mols_n = [m.GetNumAtoms() + m.GetNumBonds() for m in self.init_mols]

        elif args.reward.lower() == "qed":
            self.qed_reward = True
            self.qed_weight = 0.8
            self.sa_weight = 0.2
            if len(args.reward_weight) == 0:
                # use default weights
                pass
            elif len(args.reward_weight) == 1:
                # assume that the one value is bde weight, which is the same as main_multi.py
                self.qed_weight = args.reward_weight[0]
                self.sa_weight = 1.0 - self.qed_weight
                
            elif len(args.reward_weight) == 2:
                self.qed_weight = args.reward_weight[0]
                self.sa_weight = args.reward_weight[1]

        elif args.reward.lower() == "plogp":
            self.plogp_reward = True
            # self.plogp_weight = 1.0

        self.init_rewards = self.find_reward(self.init_mols)

    def read_bde_from_df(self, pred, mol = None):
        if len(pred.bde_pred) < 1:
            return 0.0, False # invalid bde
        elif len(pred.bde_pred) == 1:
            return min(pred.bde_pred), True
        else:
            # to do : use rdkit's build in functions for para and meta parts.
            return min(pred.bde_pred), True

            # the mol has at least 2 O-H bonds.
            meta_para_m = Chem.RWMol(mol)
            meta_para_m = Chem.AddHs(meta_para_m)
            oids = []
            for ohbid in pred.bond_index:
                b = meta_para_m.GetBondWithIdx(ohbid).GetBeginAtom()
                e = meta_para_m.GetBondWithIdx(ohbid).GetBeginAtom()
                if b.GetAtomicNum() == 8:
                    oids.append(b.GetIdx())
                else :
                    oids.append(e.GetIdx())

            # If it also has an N in the ring. 
            meta_bdes = []
            para_bdes = []
            for a in meta_para_m.GetAtoms():
                if a.GetAtomicNum() == 7 and a.IsInRingSize(6): # 7 is N,
                    for oid, pb in zip(oids, pred.bde_pred):
                        find_atoms_not_in_ring = False
                        sp = Chem.rdmolops.GetShortestPath(meta_para_m, a.GetIdx(), oid)
                        for n in sp [:-1]:
                            if not meta_para_m.GetAtomWithIdx(n).IsInRing():
                                find_atoms_not_in_ring = True
                                break;

                        if (not find_atoms_not_in_ring) and len(sp) == 4: 
                            # meta
                            # the path also includes O, 
                            meta_bdes.append(pb)

                        if (not find_atoms_not_in_ring) and len(sp) == 5: 
                            # para
                            # the path also includes O, 
                            # pass
                            para_bdes.append(pb)

            if len(meta_bdes) > 0:
                return min(meta_bdes), True

            return min(pred.bde_pred), True


    def predict_BDE(self, smiles, bacthed=False):
        """
            alfabet uses the SMILES string as their input. They convert the smiles to rdkit.ROMol later. 
        """


        if bacthed:
            pass
            # try:
                
            #     rt = bde_model.predict(smiles)
            #     pred_all = rt[['molecule', 'bond_index', 'bde_pred','bond_type']][rt.bond_type.str.contains("O-H|H-O")]
            #     pbdes = []
            #     valids = []

            #     for s, mol in zip(smiles, mols):
            #         pred = pred_all.loc[pred_all['molecule'] == s]
            #         pbde, valid = self.read_bde_from_df(mol, pred)
            #         pbdes.append(pbde)
            #         valids.append(valid)

            #     return pbdes, valids

            # except Exception as e:
            #     print("predict_BDE Exception: ")
            #     print(e)

        else:

            # predict the bde for each mol individually
            pbdes = []
            valids = []

            for s in smiles:
                try :
                    rt = bde_model.predict([s])
                    pred = rt[['molecule', 'bond_index', 'bde_pred','bond_type']][rt.bond_type.str.contains("O-H|H-O")]
                    pbde, valid = self.read_bde_from_df(pred = pred)
                    pbdes.append(pbde)
                    valids.append(valid)
                except Exception as e:
                    print(e)
                    print("Error Seq BDE :")
                    print(smiles)
                    pbdes.append(0.0)
                    valids.append(False)
                    # raise e

            return pbdes, valids

    def rwmol2multi_data(self, mol):
        
        mol = Chem.RWMol(mol)
        mol = Chem.AddHs(mol)
        # https://sourceforge.net/p/rdkit/mailman/message/33386856/
        # EmbedMolecule may return -1 for some mols.
        try:
            cids = AllChem.EmbedMultipleConfs(mol, numConfs=len(self.ip_model) * 2,useRandomCoords=True, maxAttempts=24)
            if len(cids) < len(self.ip_model):
                # with open("error_smiles_ip.txt", 'a') as f:
                #     pm = Chem.RemoveHs(mol)
                #     if len(self.path) >= 2 and self.vaild_conformer:
                #         f.write("{s}\t{g}\n".format(s = Chem.MolToSmiles(self._path[-2]), g = Chem.MolToSmiles(pm)))
                #         self.vaild_conformer = False
                return dict(), False
            coords = [mol.GetConformer(cid).GetPositions() for cid in cids[0:len(self.ip_model)]]
            coords = [torch.tensor(coord, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(self.device) for coord in coords]
            numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
            numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(self.device)
            charge = torch.tensor([1, 0, -1]).to(self.device)  # cation, neutral, anion
            mult = torch.tensor([2, 1, 2]).to(self.device)
            return [dict(coord=coord, numbers=numbers, charge=charge, mult=mult) for coord in coords], True
        except Exception as e:
            print("IP Exception: ")
            print(e)
            # raise e
            return dict(), False

    def rwmol2data(self, mol):
        
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.RWMol(mol)
        mol = Chem.AddHs(mol)
        # https://sourceforge.net/p/rdkit/mailman/message/33386856/
        # EmbedMolecule may return -1 for some mols.
        try:
            cr, cb = self.lru_cache_ic.get(smiles)
            if cb:
                return dict(), False
            cid = AllChem.EmbedMolecule(mol, useRandomCoords = True, maxAttempts= 7)
            if cid == -1:
                # with open("error_smiles_ip.txt", 'a') as f:
                #     pm = Chem.RemoveHs(mol)
                #     if len(self.path) >= 2 and self.vaild_conformer:
                #         f.write("{s}\t{g}\n".format(s = Chem.MolToSmiles(self._path[-2]), g = Chem.MolToSmiles(pm)))
                #         self.vaild_conformer = False
                self.lru_cache_ic.put(smiles, 0)
                return dict(), False
            coords = mol.GetConformer(cid).GetPositions()
            coords = torch.tensor(coords, dtype=torch.float).unsqueeze(0).repeat(3, 1, 1).to(self.device)
            numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
            numbers = torch.tensor(numbers, dtype=torch.long).unsqueeze(0).repeat(3, 1).to(self.device)
            charge = torch.tensor([1, 0, -1]).to(self.device)  # cation, neutral, anion
            mult = torch.tensor([2, 1, 2]).to(self.device)
            return dict(coord=coords, numbers=numbers, charge=charge, mult=mult), True
        except Exception as e:
            print("IP Exception: ")
            print(e)
            # raise e
            return dict(), False

    def predict_IP(self, mol, use_random_model = True):

        # use_random_model = False
        
        if use_random_model:
            data, valid = self.rwmol2data(mol)
            if not valid:
                return 0.0, False

            model_id = np.random.randint(0, len(self.ip_model))
            model_id = 4
            ip_model = self.ip_model[model_id]
            # disable optimizations for safety. with some combinations of pytorch/cuda it's getting very slow
            with torch.jit.optimized_execution(False), torch.no_grad():   
                pred = ip_model.model(data)
            pred['charges'] = pred['charges'].sum(-1)
            pred = to_numpy(pred)
            # calculate indicies
            pred.update(calc_react_idx(pred))
            # write
            for k, v in pred.items():
                pred[k] = v.tolist()
            pred_ip = ev2kcal_per_mol( pred['ip'])
            
            return pred_ip, True
        else:
        
            datas, valid = self.rwmol2multi_data(mol)
            if not valid:
                return 0.0, False
            pred_ip = 0.0
            last_data = None
            for ip_model, data in zip(self.ip_model, datas):

                # disable optimizations for safety. with some combinations of pytorch/cuda it's getting very slow
                with torch.jit.optimized_execution(False), torch.no_grad():   
                    pred = ip_model.model(data)
                pred['charges'] = pred['charges'].sum(-1)
                pred = to_numpy(pred)
                # calculate indicies
                pred.update(calc_react_idx(pred))
                # write
                for k, v in pred.items():
                    pred[k] = v.tolist()
                pred_ip = pred_ip + ev2kcal_per_mol( pred['ip'])
            
            return pred_ip / len(self.ip_model), True

    
    def predict_BDE_cache(self, molecules, useCache):
        # bde predictor with cache

        # remove duplicated smiles. 
        smiles = [Chem.MolToSmiles(mol) for mol in molecules]
        smiles_p = {}
        for s, i in  zip(smiles, range(len(smiles))):
            if s in smiles_p:
                smiles_p[s].append(i)
            else:
                smiles_p[s] = [i]

        bde_ps = [-1000 for s in smiles]
        bde_vs = [False for s in smiles]

        # find BDE in the cache
        smiles_uncached = {}

        for s, ids in smiles_p.items():
            p, v = self.lru_cache.get(s)
            if v:
                for i in ids:
                    bde_ps[i] = p
                    bde_vs[i] = True
            else:
                smiles_uncached[s] = ids

        if len(smiles_uncached) > 0:
            bde_ucps, bde_ucvs = self.predict_BDE(smiles_uncached.keys())
            for s, p, v, ids in zip(smiles_uncached.keys(), bde_ucps, bde_ucvs, smiles_uncached.values()):
                if v:
                    self.lru_cache.put(s, p)
                    for i in ids:
                        bde_ps[i] = p
                        bde_vs[i] = True
        return bde_ps, bde_vs

    def find_bde_ip_reward(self, molecules, useCache):
        # only use cache for bde 
        rewards = []
        bde_ps, bde_vs = self.predict_BDE_cache(molecules, useCache)
        ip_ps = []
        rrabs = []
        for bdep, bdev, molecule, init_mol_n in zip(bde_ps, bde_vs, molecules, self.init_mols_n):
            n = molecule.GetNumAtoms() + molecule.GetNumBonds()
            rrab = float(init_mol_n - n) / float(init_mol_n)
            rrabs.append(rrab)
            if bdev:
                bden = self.bde_scaler.transform([[bdep * self.bde_factor]])
                ipp, valid_ip = self.predict_IP(molecule)
                # ipp = 140
                # valid_ip = True
                ip_ps.append(ipp)
                if not valid_ip:
                    rewards.append(-1000.0)
                    continue
                ipn = self.ip_scaler.transform([[ipp * self.ip_factor]])
                bde = bden[0][0]
                ip = ipn[0][0]
                r = 2.0 * (self.bed_weight * (1.0 - bde) + self.ip_weight * ip) + self.rrab_weight * rrab
                rewards.append(r)
            else :
                # with open("error_smiles_bde.txt", 'a') as f:
                #     f.write("{s} \t {r}\n".format(s = Chem.MolToSmiles(molecule), r = bdep))
                ip_ps.append(-1000.0)
                rewards.append(-1000.0)
        return {'reward':rewards, 'BDE':bde_ps, 'IP':ip_ps, 'RRAB': rrabs}
            
    def find_qed_reward(self, molecules):
        rs = []
        qeds = []
        sas = []
        for molecule in molecules:
            qed = QED.qed(molecule)
            qeds.append(qed)
            SA_score = sascorer.calculateScore(molecule)
            sas.append(SA_score)
            reward = (qed * self.qed_weight - self.sa_weight * SA_score) * self.discount_factor ** (self.max_steps-self.current_step)
            rs.append(reward)
        return {'reward': rs, 'QED':qeds, 'SA_score':sas}

    def find_plogp_reward(self, molecules):
        rs = []
        sims = []

        logp_mean = 2.4570953396190123
        logp_std = 1.434324401111988
        sa_mean = 3.0525811293166134
        sa_std = 0.8335207024513095
        cycle_mean = 0.0485696876403053
        cycle_std = 0.2860212110245455

        for mol in molecules:
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

            sims.append(-1)
            reward = score * self.discount_factor ** (self.max_steps-self.current_step)
            rs.append(reward)
        return {'reward': rs, 'plogp':rs, 'sim': sims}


    def find_reward(self, molecules = None, useCache = False):
        if molecules is None:
            molecules = self.states
        if self.bde_ip_reward:
            return self.find_bde_ip_reward(molecules, useCache)
        elif self.qed_reward:
            return self.find_qed_reward(molecules)
        elif self.plogp_reward:
            return self.find_plogp_reward(molecules)

class DistributedAgent(object):
    """docstring for DistributedAgent"""
    def __init__(self, input_length, gpu_index, device, args, rank):
        super(DistributedAgent, self).__init__()
        self.gpu_index = gpu_index
        self.device = device
        print(device)
        torch.cuda.set_device(gpu_index)

        self.observation_type = args.observation_type
        self.max_batch_size = args.max_batch_size

        if self.observation_type != "vector":
            # todo: initialize the same weights in target_dqn and dqn? 
            # MolDQN didn't do that, I am not sure if I need to add it.
            # --Huanyi
            self.dqn = MolDQN(input_length, 1).to(self.device)
            self.target_dqn = MolDQN(input_length, 1).to(self.device)
        else:
            self.dqn = make_transformer_model(**hyp.transformer_params).to(self.device)
            self.target_dqn = make_transformer_model(**hyp.transformer_params).to(self.device)
        if args.checkpoint is not None:
            if rank == 0:
                # load pre-trained models
                dqn_checkpoint = torch.load(f'./Experiments/models/{args.checkpoint}_best_model_dqn.pth')
                dqn_model_state = dqn_checkpoint['model_state_dict']
                self.dqn.load_state_dict(dqn_model_state)

            target_dqn_checkpoint = torch.load(f'./Experiments/models/{args.checkpoint}_best_model_target_dqn.pth')
            target_dqn_model_state = target_dqn_checkpoint['model_state_dict']
            self.target_dqn.load_state_dict(target_dqn_model_state)
            self.eps_threshold = target_dqn_checkpoint['eps_threshold']
        
        self.dqn = nn.parallel.DistributedDataParallel(self.dqn, 
            device_ids=[self.gpu_index], 
            output_device = self.gpu_index) 

        for p in self.target_dqn.parameters():
            p.requires_grad = False

        self.observation_type = args.observation_type
        if self.observation_type == 'rdkit':
            self.use_cxx_incremental_fingerprint = 0
        elif self.observation_type == 'list':
            self.use_cxx_incremental_fingerprint = 1
        elif self.observation_type == 'numpy':
            self.use_cxx_incremental_fingerprint = 2
        elif self.observation_type == 'vector':
            self.use_cxx_incremental_fingerprint = 0
        else:
            self.use_cxx_incremental_fingerprint = None

        self.replay_buffer = utils.ReplayBuffer(hyp.replay_buffer_size)
        # the original replay buffer is confusing and inefficient, 
        # use torchrl.data.ReplayBuffer instead
        # self.replay_buffer = torchrl.data.ReplayBuffer(hyp.replay_buffer_size)
        self.optimizer = getattr(opt, hyp.optimizer)(
            self.dqn.parameters(), lr=hyp.learning_rate
        )

    def get_action(self, observations, epsilon_threshold):
        isGreedy = True

        if np.random.uniform() < epsilon_threshold:
            if isinstance(observations, list):
                al = observations[0].shape[0]
            else:
                al = observations.shape[0]
            action = np.random.randint(0, al)
            isGreedy = False
        elif self.observation_type != 'vector':
            q_value = self.dqn.forward(observations).cpu()
            action = torch.argmax(q_value).numpy()
        else:
            node_features = observations[0]
            adjacency_matrix = observations[1]
            distance_matrix = observations[2]
            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
            q_value = self.dqn.forward(node_features, batch_mask, adjacency_matrix, distance_matrix, None).cpu()
            action = torch.argmax(q_value).numpy()
        return action, isGreedy

    # def training_step(self):
    #     batch_size = min(self.replay_buffer.__len__(), self.max_batch_size)
    #     # states, next_states, rewards, dones = agent.replay_buffer.sample(batch_size) 
    #     states, next_states, rewards, dones = [], [], [], []

    #     data_batch = self.replay_buffer.sample(batch_size)
    #     # data = (reward, float(done), saved_observations)

    #     for data in data_batch:
    #         reward, done, saved_observations = data

    #         if self.observation_type == 'rdkit':
    #             observations = torch.tensor(saved_observations, device = agent.device).float()
    #         elif self.observation_type == 'list':
    #             st, fingerprints = saved_observations
    #             observations = np.vstack([utils.get_observations_from_list(fp, st) for fp in fingerprints])
    #             observations = torch.tensor(observations, device = agent.device).float()
    #         elif self.observation_type == 'numpy':
    #             observations = torch.tensor(saved_observations, device = agent.device).float()
    #         elif self.observation_type == 'vector':
    #             observations = [torch.tensor(o, device = agent.device) for ob in saved_observations]
    #         states.append(observations[-1])
    #         next_state.append(observations)
    #         rewards.append(reward)
    #         dones.append(done)

    #     # self.dqn.training_step(states, next_states, rewards, dones, batch_size)

    #     q_t = torch.zeros(batch_size, 1, requires_grad=False)
    #     v_tp1 = torch.zeros(batch_size, 1, requires_grad=False)
    #     for i in range(batch_size):
    #         # v_tp1 is the best action for the next step
    #         v_tp1[i] = torch.max(agent.target_dqn(next_states[i]))


    #     # for epoch in range(epochs):
    #     #     loss = []
    #     #     for adjacency_matrix, node_features, distance_matrix, targets in data_loader:
    #     #         batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0
    #     #         l, _ = model.training_step(node_features, batch_mask, adjacency_matrix, distance_matrix, None, targets)
    #     #         loss.append(l.detach().cpu().numpy().mean())
    #     #     loss = np.array(loss).mean()
    #     #     print(f'Epoch: {epoch}, loss: {loss:.6f}')

    #     # def training_step(self, src, src_mask, adj_matrix, distances_matrix, edges_att, y_real):
    #     #     self.optimizer.zero_grad()
    #     #     y_pred = self.forward(src, src_mask, adj_matrix, distances_matrix, edges_att)
    #     #     loss = self.loss(y_pred, y_real)
    #     #     loss.backward()
    #     #     self.optimizer.step()
    #     #     return loss, y_pred

    #     if self.observation_type != 'vector':
    #         state = torch.FloatTensor(states).reshape(batch_size, hyp.fingerprint_length + 1).to(self.device)
    #         q_t = agent.dqn(state).to(self.device)
    #     else:
    #         state = 


    #     rewards = torch.FloatTensor(rewards).reshape(q_t.shape).to(device)
    #     q_t = q_t.to(device)

    #     v_tp1 = v_tp1.to(device)
    #     dones = torch.FloatTensor(dones).reshape(q_t.shape).to(device)
    #     q_tp1_masked = (1 - dones) * v_tp1
    #     q_t_target = rewards + hyp.gamma * q_tp1_masked
    #     td_error = q_t - q_t_target
    #     q_loss = torch.where(
    #         torch.abs(td_error) < 1.0,
    #         0.5 * td_error * td_error,
    #         1.0 * (torch.abs(td_error) - 0.5),
    #     )
    #     q_loss = q_loss.mean()
    #     agent.optimizer.zero_grad()
    #     q_loss.backward()
    #     torch.distributed.barrier()
    #     agent.optimizer.step()
    #     with torch.no_grad():
    #         for p, p_targ in zip(agent.dqn.parameters(), agent.target_dqn.parameters()):
    #             p_targ.data.mul_(hyp.polyak)
    #             p_targ.data.add_((1 - hyp.polyak) * p.data)
    #     loss = q_loss.item()
    #     batch_losses.append(loss) 

    def training_step(self):
        batch_size = min(self.replay_buffer.__len__(), self.max_batch_size)
        # states, next_states, rewards, dones = agent.replay_buffer.sample(batch_size) 
        states, next_states, rewards, dones = [], [], [], []
        # for transformer
        states_nf, states_am, states_dm = [], [], [] 
        next_states_nf, next_states_am, next_states_dm = [], [], [] 

        data_batch = self.replay_buffer.sample(batch_size)

        for data in data_batch:
            # data = (reward, float(done), saved_observations)
            reward, done, saved_observations = data

            if self.observation_type != 'vector':

                if self.observation_type == 'rdkit':
                    observations = torch.tensor(saved_observations, device = self.device).float()
                elif self.observation_type == 'list':
                    st, fingerprints = saved_observations
                    observations = np.vstack([utils.get_observations_from_list(fp, st) for fp in fingerprints])
                    observations = torch.tensor(observations, device = self.device).float()
                elif self.observation_type == 'numpy':
                    observations = torch.tensor(saved_observations, device = self.device).float()
                states.append(observations[-1])
                next_states.append(observations)
                
            else:
                # observations = [torch.tensor(o, device = agent.device) for ob in saved_observations]
                # saved_observations is [features_list, adjacency_list, distance_list]
                states_nf.append(torch.tensor(saved_observations[0][-1], device = self.device))
                states_am.append(torch.tensor(saved_observations[1][-1], device = self.device))
                states_dm.append(torch.tensor(saved_observations[2][-1], device = self.device))

                next_states_nf.append(torch.tensor(saved_observations[0], device = self.device))
                next_states_am.append(torch.tensor(saved_observations[1], device = self.device))
                next_states_dm.append(torch.tensor(saved_observations[2], device = self.device))

            rewards.append(reward)
            dones.append(done)
            
        # self.dqn.training_step(states, next_states, rewards, dones, batch_size)
        # q_t -> q
        # q = torch.zeros(batch_size, 1, requires_grad=False)
        if self.observation_type != 'vector':
            # state = torch.FloatTensor(states).reshape(batch_size, hyp.fingerprint_length + 1).to(self.device)
            states = torch.stack(states, dim = 0)
            q = self.dqn(states) #.to(self.device)
        else:
            # states_nf = torch.tensor(states_nf, dtype = torch.float32, device=self.device)
            # states_am = torch.tensor(states_am, dtype = torch.float32, device=self.device)
            # states_dm = torch.tensor(states_am, dtype = torch.float32, device=self.device)
            states_nf = torch.stack(states_nf, dim = 0)
            states_am = torch.stack(states_am, dim = 0)
            states_dm = torch.stack(states_dm, dim = 0)
            batch_mask = torch.sum(torch.abs(states_nf), dim=-1) != 0
            # to tensor

            q = self.dqn(states_nf, batch_mask, states_am, states_dm, None) # edges_att is None

        # v_tp1 -> max_q #_in_next_states
        max_q = torch.zeros(batch_size, 1, requires_grad=False)

        if self.observation_type != 'vector':
            for i in range(batch_size):
                # max_q is the best action for the next step
                max_q[i] = torch.max(self.target_dqn(next_states[i]))
        else:
            for i in range(batch_size):
                # max_q is the best action for the next step


                # may optimize this
                batch_mask = torch.sum(torch.abs(next_states_nf[i]), dim=-1) != 0
                max_q[i] = torch.max(self.target_dqn(
                    next_states_nf[i], batch_mask, next_states_am[i], next_states_dm[i], None))


        max_q = max_q.to(self.device)
        rewards = torch.tensor(rewards, dtype = torch.float32, device = self.device) # .reshape(q_t.shape)
        dones = torch.tensor(dones, dtype = torch.float32, device = self.device) # .reshape(q_t.shape)

        mask = (1 - dones) * max_q
        target = rewards + hyp.gamma * mask
        td_error = q - target
        loss = torch.where(
            torch.abs(td_error) < 1.0,
            0.5 * td_error * td_error,
            1.0 * (torch.abs(td_error) - 0.5),
        )

        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.distributed.barrier()
        self.optimizer.step()
        with torch.no_grad():
            for p, p_targ in zip(self.dqn.parameters(), self.target_dqn.parameters()):
                p_targ.data.mul_(hyp.polyak)
                p_targ.data.add_((1 - hyp.polyak) * p.data)
        # loss = loss.item()
        # batch_losses.append(loss) 
        return loss.item()

