"""Tools for manipulating graphs and converting from atom and pair features."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import rdkit

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch.nn as nn
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import RDConfig
import pandas as pd
import numpy as np
import pickle
# import joblib
#from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
# from sklearn import preprocessing
import torch
import numpy as np
import random
import copy
import hyp
import sys
import os
import threading
from sklearn.metrics import pairwise_distances


from collections import OrderedDict


sys.path.append(os.path.join(os.path.dirname(rdkit.__file__), RDConfig.RDContribDir, "SA_Score"))
#print(sys.path)
import sascorer

#debug
import time
cache = {}
fc = 0

morganFingerprintGen = rdFingerprintGenerator.GetMorganGenerator(fpSize=hyp.fingerprint_length, radius=hyp.fingerprint_radius)

def get_fingerprint(smiles, fingerprint_length, fingerprint_radius):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule. # smiles is not a smiles string. It is RWMol. --Huanyi Qin
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """

    #str = AllChem.MolToSmiles(smiles)
    # print(smiles)
    if smiles is None:
        return np.zeros((hyp.fingerprint_length,))
    return morganFingerprintGen.GetFingerprintAsNumPy(smiles)

def get_fingerprint_cache(smiles, fingerprint_length, fingerprint_radius):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule. # smiles is not a smiles string. It is RWMol. --Huanyi Qin
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
    #global fc
    global cache
    if smiles is None:
        return np.zeros((hyp.fingerprint_length,))
    str = AllChem.MolToSmiles(smiles)
    if str in cache:
        #print(len(cache[str]))
        return cache[str]
    else :
        cache[str] = morganFingerprintGen.GetFingerprintAsNumPy(smiles)
    return cache[str]

def print_dict():
    global fc
    print(str(len(cache)) + ":" + str(fc))

def get_fingerprint_old(smiles, fingerprint_length, fingerprint_radius):
    """Get Morgan Fingerprint of a specific SMILES string.

  Args:
    smiles: String. The SMILES string of the molecule.
    hparams: tf.contrib.training.HParams. Hyper parameters.

  Returns:
    np.array. shape = [hparams.fingerprint_length]. The Morgan fingerprint.
  """
    if smiles is None:
        return np.zeros((hyp.fingerprint_length,))
    # molecule = Chem.MolFromSmiles(smiles)
    molecule = smiles
    if molecule is None:
        return np.zeros((hyp.fingerprint_length,))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, hyp.fingerprint_radius, hyp.fingerprint_length
    )
    arr = np.zeros((1,))
    # ConvertToNumpyArray takes ~ 0.19 ms, while
    # np.asarray takes ~ 4.69 ms
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def atom_valences(atom_types):
    """Creates a list of valences corresponding to atom_types.

  Note that this is not a count of valence electrons, but a count of the
  maximum number of bonds each element will make. For example, passing
  atom_types ['C', 'H', 'O'] will return [4, 1, 2].

  Args:
    atom_types: List of string atom types, e.g. ['C', 'H', 'O'].

  Returns:
    List of integer atom valences.
  """
    periodic_table = Chem.GetPeriodicTable()
    return [
        max(list(periodic_table.GetValenceList(atom_type))) for atom_type in atom_types
    ]

def atom_label(atom):
    #try:
    if atom.HasProp("atomLabel"):
        label = atom.GetProp("atomLabel")
    #except:
    else:
        label = 'None'
    return label


def get_scaffold(mol):
    """Computes the Bemis-Murcko scaffold for a molecule.

  Args:
    mol: RDKit Mol.

  Returns:
    String scaffold SMILES.
  """
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol), isomericSmiles=True)


def contains_scaffold(mol, scaffold):
    """Returns whether mol contains the given scaffold.

  NOTE: This is more advanced than simply computing scaffold equality (i.e.
  scaffold(mol_a) == scaffold(mol_b)). This method allows the target scaffold to
  be a subset of the (possibly larger) scaffold in mol.

  Args:
    mol: RDKit Mol.
    scaffold: String scaffold SMILES.

  Returns:
    Boolean whether scaffold is found in mol.
  """
    pattern = Chem.MolFromSmiles(scaffold)
    matches = mol.GetSubstructMatches(pattern)
    return bool(matches)


def get_largest_ring_size(molecule):
    """Calculates the largest ring size in the molecule.

  Refactored from
  https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py

  Args:
    molecule: Chem.Mol. A molecule.

  Returns:
    Integer. The largest ring size.
  """
    cycle_list = molecule.GetRingInfo().AtomRings()
    if cycle_list:
        cycle_length = max([len(j) for j in cycle_list])
    else:
        cycle_length = 0
    return cycle_length


def penalized_logp(molecule):
    """Calculates the penalized logP of a molecule.

  Refactored from
  https://github.com/wengong-jin/icml18-jtnn/blob/master/bo/run_bo.py
  See Junction Tree Variational Autoencoder for Molecular Graph Generation
  https://arxiv.org/pdf/1802.04364.pdf
  Section 3.2
  Penalized logP is defined as:
   y(m) = logP(m) - SA(m) - cycle(m)
   y(m) is the penalized logP,
   logP(m) is the logP of a molecule,
   SA(m) is the synthetic accessibility score,
   cycle(m) is the largest ring size minus by six in the molecule.

  Args:
    molecule: Chem.Mol. A molecule.

  Returns:
    Float. The penalized logP value.

  """
    log_p = Descriptors.MolLogP(molecule)
    sas_score = sascorer.calculateScore(molecule)
    largest_ring_size = get_largest_ring_size(molecule)
    cycle_score = max(largest_ring_size - 6, 0)
    return log_p - sas_score - cycle_score





def batchify(iterable, batch_size):
    for ndx in range(0, len(iterable), batch_size):
        batch = iterable[ndx: min(ndx + batch_size, len(iterable))]
        yield batch




def score_from_one_prediction(bde, ip, scaler_bde, scaler_ip, weight):
    """
    Proposed score (to validate with Sylvain and Sophie.)
    :param bde float: bde value for the tested smile
    :param ip float: ip value for the tested smile
    :param scaler_bde: scaler trained on bde dataset
    :param scaler_ip: scaler trained on the ip dataset
    :return score
    """
    bde = scaler_bde.transform([[bde]])
    ip = scaler_ip.transform([[ip]])
    # score between 0 and 1 ??? => to discuss with DAI
    return 2*((1-weight)*ip[0][0] + (1-bde[0][0])*weight)  



def push_grad(local_dqn, global_grad, ix, counter, num_init_mol, gpu_ix):
    print('Counter on worker {} is :'.format(ix), counter)
    if gpu_ix == 0:  ## gpu_ix == 0 means the local_dqn is on the same gpu of global_grad
        if (counter % num_init_mol > 0):
            for lp, gp_grad in zip(local_dqn.parameters(), global_grad):
                gp_grad += lp.grad.div(float(num_init_mol))
        elif (counter % num_init_mol == 0):
          # global_grad = [lp.grad.div(float(num_init_mol)).clone() for lp in local_dqn.parameters()]
            for gp_grad in global_grad:
                gp_grad -= gp_grad 
            for lp, gp_grad in zip(local_dqn.parameters(), global_grad):
                gp_grad += lp.grad.div(float(num_init_mol)).clone() 
    else:  ## gpu_ix != 0 requires moving local_dqn to the gpu of global_grad
        global_device = (global_grad[0]).device
        if (counter % num_init_mol > 0):
            for lp, gp_grad in zip(local_dqn.parameters(), global_grad):
                local_grad = lp.grad.clone().to(global_device)
                gp_grad += local_grad.div(float(num_init_mol))
        elif (counter % num_init_mol == 0):
            for gp_grad in global_grad:
                gp_grad -= gp_grad 
            for lp, gp_grad in zip(local_dqn.parameters(), global_grad):
                local_grad = lp.grad.clone().to(global_device)
                gp_grad += local_grad.div(float(num_init_mol))         
    return None



def clean_grad(global_grad):
  for grad in global_grad:
    grad = torch.zeros_like(grad)


def clean_counter(counter, num_init_mol):
  if counter >= num_init_mol:
    counter -= num_init_mol
  return None 


def get_observations(fp, remaining_steps):
    return np.append(np.array(fp, dtype='uint8'), remaining_steps)

def get_observations_from_list(fp_list, remaining_steps):
    a = np.zeros(hyp.fingerprint_length + 1, dtype='uint8')
    for f in fp_list:
        a[f] = 1
    a[-1] = remaining_steps
    return a

class ReplayBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0
        self.setseed = True

    def __len__(self):
        return len(self._storage)

    def add(self, data):
        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        states, next_states, rewards, dones = [], [], [], []
        for i in idxes:
            data = self._storage[i]
            fingerprints, steps, reward, done, obs_t, obs_tp1 = data

            if self.use_cxx_incremental_fingerprint == 0:
                states.append(obs_t)
                next_states.append(obs_tp1)
            elif self.use_cxx_incremental_fingerprint == 1:
                states.append(get_observations_from_list(fingerprints[-1], steps))
                observations = np.vstack([get_observations_from_list(fp, steps) for fp in fingerprints])
                next_states.append(torch.FloatTensor(observations).reshape(-1, hyp.fingerprint_length + 1))
            elif self.use_cxx_incremental_fingerprint == 2:
                states.append(obs_t)
                next_states.append(obs_tp1)
            
            rewards.append(reward)
            dones.append(done)
        
        return states, next_states, rewards, dones

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        states, next_states, rewards, dones
        """
        if self.setseed :
            random.seed(0)
            self.setseed = False
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return [self._storage[i] for i in idxes]


class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.hit_count = 0
        self.total_count = 0
        self.hit_count_episode = 0
        self.total_count_episode = 0

    def get(self, key):
        # print(key)
        self.total_count += 1
        self.total_count_episode += 1
        if key not in self.cache:
            return -1000, False

        else:
            self.hit_count += 1
            self.hit_count_episode += 1
            self.cache.move_to_end(key)
            return self.cache[key], True

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last = False)

    def hit_rate(self, episode = False):
        
        if episode:
            if self.total_count_episode == 0:
                return 0.0
            return self.hit_count_episode / self.total_count_episode
        else:
            if self.total_count == 0:
                return 0.0
            return self.hit_count / self.total_count
    def reset_episode_hit_rate(self):
        self.hit_count_episode = 0
        self.total_count_episode = 0



def get_atom_vectors(mol, remaining_steps = 0, add_dummy_node=True, one_hot_formal_charge=False):
    """featurize atom vectors from molecule and remaing_steps.

    Args:
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph. Defaults to True.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded. Defaults to False.

    Returns:
        a list of graph descriptors (node features, adjacency matrices, distance matrices),
    """

    if True:
        mol = Chem.RWMol(mol)
        try:
            mol = Chem.AddHs(mol)
            AllChem.EmbedMolecule(mol, maxAttempts=30)
            AllChem.UFFOptimizeMolecule(mol)
            mol = Chem.RemoveHs(mol)
        except:
            AllChem.Compute2DCoords(mol)
    else:
        AllChem.Compute2DCoords(mol)

    afm, adj, dist = featurize_mol(mol, remaining_steps, add_dummy_node, one_hot_formal_charge)

    return [afm, adj, dist]


def featurize_mol(mol, remaining_steps, add_dummy_node, one_hot_formal_charge):
    """Featurize molecule.

    Args:
        mol (rdchem.Mol): An RDKit Mol object.
        add_dummy_node (bool): If True, a dummy node will be added to the molecular graph.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A tuple of molecular graph descriptors (node features, adjacency matrix, distance matrix).
    """
    node_features = np.array([get_atom_features(atom, remaining_steps, one_hot_formal_charge)
                              for atom in mol.GetAtoms()])

    adj_matrix = np.eye(mol.GetNumAtoms())
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetIdx()
        end_atom = bond.GetEndAtom().GetIdx()
        adj_matrix[begin_atom, end_atom] = adj_matrix[end_atom, begin_atom] = 1

    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)

    if add_dummy_node:
        m = np.zeros((node_features.shape[0] + 1, node_features.shape[1] + 1))
        m[1:, 1:] = node_features
        m[0, 0] = 1.
        node_features = m

        m = np.zeros((adj_matrix.shape[0] + 1, adj_matrix.shape[1] + 1))
        m[1:, 1:] = adj_matrix
        adj_matrix = m

        m = np.full((dist_matrix.shape[0] + 1, dist_matrix.shape[1] + 1), 1e6)
        m[1:, 1:] = dist_matrix
        dist_matrix = m

    return node_features, adj_matrix, dist_matrix


def get_atom_features(atom, remaining_steps, one_hot_formal_charge=True):
    """Calculate atom features.

    Args:
        atom (rdchem.Atom): An RDKit Atom object.
        one_hot_formal_charge (bool): If True, formal charges on atoms are one-hot encoded.

    Returns:
        A 1-dimensional array (ndarray) of atom features.
    """
    attributes = []

    attributes += one_hot_vector(
        atom.GetAtomicNum(),
        [5, 6, 7, 8, 9, 15, 16, 17, 35, 53, 999]
    )

    attributes += one_hot_vector(
        len(atom.GetNeighbors()),
        [0, 1, 2, 3, 4, 5]
    )

    attributes += one_hot_vector(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4]
    )

    if one_hot_formal_charge:
        attributes += one_hot_vector(
            atom.GetFormalCharge(),
            [-1, 0, 1]
        )
    else:
        attributes.append(atom.GetFormalCharge())

    attributes.append(atom.IsInRing())
    attributes.append(atom.GetIsAromatic())
    attributes.append(remaining_steps)

    return np.array(attributes, dtype=np.float32)


def one_hot_vector(val, lst):
    """Converts a value to a one-hot vector based on options in lst"""
    if val not in lst:
        val = lst[-1]
    return map(lambda x: x == val, lst)


import math
from torch.nn.init import _calculate_fan_in_and_fan_out, _no_grad_normal_, _no_grad_uniform_


def earily_stop(val_acc_history, tasks, early_stop_step_single,
                early_stop_step_multi, required_progress):
    """
    Stop the training if there is no non-trivial progress in k steps
    @param val_acc_history: a list contains all the historical validation acc
    @param required_progress: the next acc should be higher than the previous by
        at least required_progress amount to be non-trivial
    @param t: number of training steps
    @return: a boolean indicates if the model should earily stop
    """
    # TODO: add your code here
    if len(tasks) == 1:
        t = early_stop_step_single
    else:
        t = early_stop_step_multi

    if len(val_acc_history)>t:
        if val_acc_history[-1] - val_acc_history[-1-t] < required_progress:
            return True
    return False


def xavier_normal_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))

    return _no_grad_normal_(tensor, 0., std)


def xavier_uniform_small_init_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + 4*fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation

    return _no_grad_uniform_(tensor, -a, a)


def pad_array(array, shape, dtype=np.float32):
    """Pad a 2-dimensional array with zeros.

    Args:
        array (ndarray): A 2-dimensional array to be padded.
        shape (tuple[int]): The desired shape of the padded array.
        dtype (data-type): The desired data-type for the array.

    Returns:
        A 2-dimensional array of the given shape padded with zeros.
    """
    padded_array = np.zeros(shape, dtype=dtype)
    padded_array[:array.shape[0], :array.shape[1]] = array
    return padded_array


def mol_to_observation(mols_desc):
    """Create a padded batch of molecule features.

    node_features, adj_matrix, dist_matrix

    Args:
        batch (list[Molecule]): A batch of raw molecules.
        node_features = x[0]
        adjacency_matrix = x[1]
        distance_matrix = x[2]

    Returns:
        A list of FloatTensors with padded molecule features:
        adjacency matrices, node features, distance matrices, and labels.
    """
    features_list, adjacency_list, distance_list  = [], [], []
    labels = []
    max_size = 0
    for x in mols_desc:
        if x[1].shape[0] > max_size:
            max_size = x[1].shape[0]

    # max_size = 40

    for x in mols_desc:
        features_list.append(pad_array(x[0], (max_size, x[0].shape[1])))
        adjacency_list.append(pad_array(x[1], (max_size, max_size)))
        distance_list.append(pad_array(x[2], (max_size, max_size)))

    # return [FloatTensor(features) for features in (adjacency_list, features_list, distance_list)]
    return [features_list, adjacency_list, distance_list]


