# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Defines the Markov decision process of generating a molecule.

The problem of molecule generation as a Markov decision process, the
state space, action space, and reward function are defined.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import itertools
import time

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import rdFingerprintGenerator
from six.moves import range
from six.moves import zip

#import utils
import utils as utils
import pdb

#debug
import time
from rdkit.Chem import QED
import hyp

from rdkit.Chem import  AllChem
import numpy as np
import src.cenv as cenv

class Result(collections.namedtuple("Result", ["state", "reward", "terminated"])):
    """A namedtuple defines the result of a step for the molecule class.

    The namedtuple contains the following fields:
      state: Chem.RWMol. The molecule reached after taking the action.
      reward: Float. The reward get after taking the action.
      terminated: Boolean. Whether this episode is terminated.
  """

flags = cenv.Flags() # use default flags for test.
cxx_environment = cenv.Environment(hyp.atom_types, hyp.allowed_ring_sizes, hyp.fingerprint_radius, hyp.fingerprint_length, flags)

class Molecule(object):
    """Defines the Markov decision process of generating a molecule."""

    def __init__(
        self,
        args,
        init_mols=None,
    ):
        """Initializes the parameters for the MDP.

    Internal state will be stored as SMILES strings.

    Args:
      atom_types: The set of elements the molecule may contain.
      init_mol: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
        considered as the SMILES string. The molecule to be set as the initial
        state. If None, an empty molecule will be created.
      allow_removal: Boolean. Whether to allow removal of a bond.
      allow_no_modification: Boolean. If true, the valid action set will
        include doing nothing to the current molecule, i.e., the current
        molecule itself will be added to the action set.
      allow_bonds_between_rings: Boolean. If False, new bonds connecting two
        atoms which are both in rings are not allowed.
        DANGER Set this to False will disable some of the transformations eg.
        c2ccc(Cc1ccccc1)cc2 -> c1ccc3c(c1)Cc2ccccc23
        But it will make the molecules generated make more sense chemically.
      allowed_ring_sizes: Set of integers or None. The size of the ring which
        is allowed to form. If None, all sizes will be allowed. If a set is
        provided, only sizes in the set is allowed.
      max_steps: Integer. The maximum number of steps to run.
      target_fn: A function or None. The function should have Args of a
        String, which is a SMILES string (the state), and Returns as
        a Boolean which indicates whether the input satisfies a criterion.
        If None, it will not be used as a criterion.
      record_path: Boolean. Whether to record the steps internally.
    """

        self.allow_bonds_between_rings = hyp.allow_bonds_between_rings
        self.allow_no_modification = hyp.allow_no_modification
        self.allow_removal = hyp.allow_removal
        self.allowed_ring_sizes = set(hyp.allowed_ring_sizes)
        self.atom_types = set(hyp.atom_types)
        self.max_steps = args.max_steps_per_episode
        self.record_top_path = args.record_top_path
        self.record_last_path = args.record_last_path
        self.record_all_path = args.record_all_path
        self.record_path = (self.record_top_path > 0) or (self.record_last_path > 0) or args.record_all_path
        self.observation_type = args.observation_type.lower()
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

        self.morganFingerprintGen = rdFingerprintGenerator.GetMorganGenerator(fpSize=hyp.fingerprint_length, radius=hyp.fingerprint_radius)
        self.init_mols = init_mols
        self.states = []
        self.current_step = 0

    def get_path(self):
        return self.path_mols, self.path_rewards

    def initialize(self):
        """Resets the MDP to its initial state."""
        self.states = self.init_mols
        self.current_step = 0
        if self.record_path:
            self.path_mols = [[s] for s in self.states]
            # self.path_rewards = [[r] for r in self.init_rewards]
            self.path_rewards = {}
            for k, v in self.init_rewards.items():
                self.path_rewards[k] = [[val] for val in v]
        if self.bde_cache is not None:
            self.bde_cache.reset_episode_hit_rate()
        if self.ip_cache is not None:
            self.ip_cache.reset_episode_hit_rate()

    def calc_valid_actions(self):
        """Calculate the valid actions for the state.

        In this design, we do not further modify a aromatic ring. For example,
        we do not change a benzene to a 1,3-Cyclohexadiene. That is, aromatic
        bonds are not modified.

        Args:
          state: String, Chem.Mol, or Chem.RWMol. If string is provided, it is
            considered as the SMILES string. The state to query. If None, the
            current state will be considered.
          force_rebuild: Boolean. Whether to force rebuild of the valid action
            set.

        Returns:
          A set contains all the valid actions for the state. Each action is a
            SMILES string. The action is actually the resulting state.
        """
        global cxx_environment
        vas = []
        des = []
        for state, maintain_OH in zip(self.states, self.maintain_OH_flags):
            # valid_actions, fingerprints = cxx_environment.get_valid_actions_and_fingerprint_smile(
            #    state, self.use_cxx_incremental_fingerprint, maintain_OH)
            # some mathine reports seg fault for transfering Mol Object. It should be a rdkit issue.
            valid_actions, fingerprints = cxx_environment.get_valid_actions_and_fingerprint_smile(
               Chem.MolToSmiles(state), self.use_cxx_incremental_fingerprint, maintain_OH)
            if isinstance(valid_actions[0], str):
                valid_actions = [Chem.MolFromSmiles(state) for state in valid_actions]
            # valid_actions = [Chem.RWMol(state)]
            # fingerprints = [self.morganFingerprintGen.GetFingerprint(mol) for mol in valid_actions]
            # print(fingerprints)
            # debug_fingerprints = [self.morganFingerprintGen.GetFingerprint(mol) for mol in valid_actions]
            # print(debug_fingerprints)
            
            if self.observation_type == "rdkit":
                fingerprints = [self.morganFingerprintGen.GetFingerprint(mol) for mol in valid_actions]
            elif self.observation_type == "vector":
                # vector
                # mol_obs = [utils.get_atom_vectors(mol, remaining_steps) for mol in valid_actions]
                fingerprints = None
                
            vas.append(valid_actions)
            des.append(fingerprints)

        return vas, des


    def step(self, actions, rewards, pbdes = None, pips = None):
        
        self.states = actions
        if self.record_path:
            for path, state in zip(self.path_mols, self.states):
                path.append(state)

            for k, v in rewards.items():
                for r, val in zip(self.path_rewards[k], v):
                    r.append(val)
        self.current_step += 1
