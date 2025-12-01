import sys
import os
from typing import List, Dict

# Add the parent folder to sys.path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(parent_dir))
import logging
import torch
from torch.utils.data import Dataset
try:
    from data_generator import DataGenerator
except:
    logging.warning("Using relative import for data_generator")
    from .data_generator import DataGenerator
import numpy as np

import pdb
import pandas as pd
from PIL import Image
import pickle
import copy

class CustomDataset(Dataset):
    def __init__(self, data_env_config, limit, policy_model = None, model_path = None, mode = 'seq', factor_subset = [], factor_indexes = [], dataset_file=None):
        
        self.data_env = DataGenerator(data_env_config)
        self.mode = mode #options: seq [sequential], cont [controlled factors], triplet [triplet pair with different actions], rand [random reset, inbuilt], step [returns s1, s2, a as state pair]

        self.count = 0
        self.limit = limit
       
        self.classes = list(range(12))
        self.factor_subset = factor_subset
        
        
        self.factor_indexes = factor_indexes

        #policy model used for data generation: load from checkpoint path if needed
        if policy_model is not None:
            self.model = policy_model
            self.model.load(path = model_path, env = self.data_env)
        else:
            self.model = RandomPolicy(self.data_env)

        #store reference to CSV file if test set is loaded
        if dataset_file is not None:

            self.dataset = []
            self.factors = []

            with open(dataset_file, 'rb') as f:
                pickle_data = pickle.load(f)
            
            for factor in pickle_data:
                self.dataset+=pickle_data[factor]
                self.factors+=[factor]*len(pickle_data[factor])
            
            self.mode = 'test_file'
            self.limit = len(self.dataset)

    def __len__(self):
        return self.limit

    def sample_factors(self):
        sampled_factors = {}
        
        #do sampling process between min max uniformly
        for attr in self.data_env.state_attributes:
            valid = False
            while not valid:
                attr_limits = self.data_env.get_low_high_attr(attr)
                rand_vals = []
                for limit in attr_limits:
                    (low, high, typ) = limit
                    rand_val = typ(np.random.uniform(low, high+1))
                    rand_vals.append(rand_val)

                sampled_factors[attr] = rand_vals if len(attr_limits) > 1 else rand_vals[0]
                valid, err = self.data_env.custom_resetter.check_valid_factors(self.data_env.env, sampled_factors)

        return sampled_factors


    def update_factor_subset(self, input_factor: Dict = {}, factor_subset: List = []):
        sampled_factors = copy.deepcopy(input_factor)
        # Only update factors we want to sample for.
        for i,factor in enumerate(factor_subset):
            valid = False
            while not valid:
                attr_limits = self.data_env.get_low_high_attr(factor)
                rand_vals = []
                
                for limit in attr_limits:
                    (low, high, typ) = limit
                    rand_val = typ(np.random.uniform(low, high+1))
                    rand_vals.append(rand_val)
                
                #if the sampled factor is the same as before, iterate again
                if (len(attr_limits)>1 and rand_vals == sampled_factors[factor]) or (len(attr_limits)==1 and rand_vals[0]==sampled_factors[factor]):
                    continue
                
                #if no specific index to update is specified, update the whole factor, else update parts of the factor
                sampled_factors[factor] = rand_vals if len(attr_limits) > 1 else rand_vals[0]
                

                #NOTE & TODO: need to fix this, just patch work for now
                if (factor == 'holding_key') and sampled_factors[factor] == 1:
                    sampled_factors['key_pos'] = (None, None)
                elif (factor == 'door_locked') and sampled_factors[factor] == 1:
                    sampled_factors['door_open'] = 0


                valid, err = self.data_env.custom_resetter.check_valid_factors(self.data_env.env, sampled_factors, strict_check=False)

        return sampled_factors
    
    def update_factor_subset_with_index(self, input_factor: Dict = {}, factor_subset: List = [], factored_indexes: List = []):
        
        sampled_factors = copy.deepcopy(input_factor)
        
        
        # Only update factors we want to sample for.
        for i,factor in enumerate(factor_subset):
            valid = False
            
            while not valid:
                attr_limits = self.data_env.get_low_high_attr(factor)
                rand_vals = []

                
                for limit in attr_limits:
                    (low, high, typ) = limit
                    rand_val = typ(np.random.uniform(low, high+1))
                    rand_vals.append(rand_val)

                rand_vals = rand_vals if len(attr_limits) > 1 else rand_vals[0]
                
                
                #if the sampled factor is the same as before, iterate again
                if (factored_indexes[i] is None and (rand_vals == sampled_factors[factor])) or (factored_indexes[i] is not None and sampled_factors[factor][factored_indexes[i]] == rand_vals[factored_indexes[i]]):
                    continue
                
                
                #if no specific index to update is specified, update the whole factor, else update parts of the factor
                if factored_indexes[i] is None:
                    sampled_factors[factor] = rand_vals
                else:
                    sampled_factors[factor][factored_indexes[i]] = rand_vals[factored_indexes[i]]
                
                #NOTE & TODO: need to fix this, just patch work for now
                if (factor == 'holding_key') and sampled_factors[factor] == 1:
                    sampled_factors['key_pos'] = (None, None)
                elif (factor == 'door_locked') and sampled_factors[factor] == 1:
                    sampled_factors['door_open'] = 0

                
                valid, err = self.data_env.custom_resetter.check_valid_factors(self.data_env.env, sampled_factors, strict_check=False)

                
            
            

        return sampled_factors


    def __getitem__(self, index):
        """Example dictionary returned from this get_item.
        {
            "previous_obs": None,
            "current_obs": None,
            "previous_state": None,
            "current_state": None,
            "previous_norm_state": None,
        "current_norm_state": None,
            "action": None
        }
        """
        if index >= self.limit:
            raise IndexError("Index out of range")

        if self.mode == 'seq':
            #get the current visual observation and underlying state
            obs_pre = self.data_env.get_curr_obs()
            state_pre, norm_state_pre = self.data_env._construct_state()
            
            #predict action and take a step in the environment
            action, __ = self.model.predict(obs_pre, deterministic=True)
            self.data_env.step(action)

            #get the future visual observation and underlying state
            obs_post = self.data_env.get_curr_obs()
            state_post, norm_state_post = self.data_env._construct_state()
            
            # NOTE: only for sequential data generation, use the policy to output a
            # (obs_pre, obs_post), (state_pre, state_post), (norm_state_pre, norm_state_post), action
            item_dict = {}
            item_dict["previous_obs"] = obs_pre
            item_dict["current_obs"] = obs_post
            item_dict["previous_state"] = state_pre
            item_dict["current_state"] = state_post
            item_dict["previous_norm_state"] = norm_state_pre
            item_dict["current_norm_state"] = norm_state_post
            item_dict["action"] = action
            for key in item_dict.keys():
                assert item_dict[key] is not None, f"{key} is None in dataset."
            return item_dict
        elif self.mode == 'cont':
            #NOTE: input controlled_factors as empty dictionary so that all factors are randomized following env rules
            self.data_env.env = self.data_env.custom_resetter.factored_reset(self.data_env.env, self.data_env.env.unwrapped.grid.height, self.data_env.env.unwrapped.grid.width, {})
            # logging.info("after reset")
            #get the current visual observation and underlying state
            obs = self.data_env.get_curr_obs()
            state, norm_state = self.data_env._construct_state()
            action = None
        elif self.mode == 'rand':
            obs, info = self.data_env.reset()
            state, norm_state = self.data_env._construct_state()
            action = None
        elif self.mode == 'triplet':
            #get the current visual observation and underlying state
            obs_pre = self.data_env.get_curr_obs()
            state_pre, norm_state_pre = self.data_env._construct_state()
            
            #predict action and take a step in the environment
            action, __ = self.model.predict(obs, deterministic=True)
            self.data_env.step(action)

            #get the future visual observation and underlying state
            obs_post = self.data_env.get_curr_obs()
            state_post, norm_state_post = self.data_env._construct_state()
            
            item_dict = {}
            item_dict["previous_obs"] = obs_pre
            item_dict["current_obs"] = obs_post
            item_dict["alternate_obs"] = None  # TODO
            item_dict["previous_state"] = state_pre
            item_dict["current_state"] = state_post
            item_dict["alternate_state"] = None  # TODO
            item_dict["previous_norm_state"] = norm_state_pre
            item_dict["current_norm_state"] = norm_state_post
            item_dict["alternate_norm_state"] = None  # TODO
            item_dict["action"] = action
            item_dict["alternate_action"] = None  # TODO
            raise NotImplementedError(f'ERROR: data generation mode cannot be {self.mode}')
        elif self.mode == 'sample':
            sample_factors = self.sample_factors()
            self.data_env.env = self.data_env.custom_resetter.factored_reset(self.data_env.env, self.data_env.env.unwrapped.grid.height, self.data_env.env.unwrapped.grid.width, sample_factors)
            obs_pre = self.data_env.get_curr_obs()
            state_pre, norm_state_pre = self.data_env._construct_state()

            
            updated_factors = self.update_factor_subset(sample_factors, self.factor_subset)
            self.data_env.env = self.data_env.custom_resetter.factored_reset(self.data_env.env, self.data_env.env.unwrapped.grid.height, self.data_env.env.unwrapped.grid.width, updated_factors)
            obs_post = self.data_env.get_curr_obs()
            state_post, norm_state_post = self.data_env._construct_state()
            item_dict = {}
            item_dict["previous_obs"] = obs_pre
            item_dict["current_obs"] = obs_post
            item_dict["previous_state"] = state_pre
            item_dict["current_state"] = state_post
            return item_dict
        
        elif self.mode == 'sample_with_index':
            
            sample_factors = self.sample_factors()
            self.data_env.env = self.data_env.custom_resetter.factored_reset(self.data_env.env, self.data_env.env.unwrapped.grid.height, self.data_env.env.unwrapped.grid.width, sample_factors)
            obs_pre = self.data_env.get_curr_obs()
            state_pre, norm_state_pre = self.data_env._construct_state()
            
            
            updated_factors = self.update_factor_subset_with_index(sample_factors, self.factor_subset, self.factor_indexes)
            
            #TODO: fix strict check later
            self.data_env.env = self.data_env.custom_resetter.factored_reset(self.data_env.env, self.data_env.env.unwrapped.grid.height, self.data_env.env.unwrapped.grid.width, updated_factors, strict_check=False)
            obs_post = self.data_env.get_curr_obs()
            state_post, norm_state_post = self.data_env._construct_state()
            item_dict = {}
            item_dict["previous_obs"] = obs_pre
            item_dict["current_obs"] = obs_post
            item_dict["previous_state"] = state_pre
            item_dict["current_state"] = state_post
            return item_dict
        
        elif self.mode == 'test_file':
            
            sample = self.dataset[index]
            factor = self.factors[index]
            return sample['previous_obs'], sample['current_obs'], sample['previous_state'], sample['current_state'], factor
        
        else:
            raise NotImplementedError(f'ERROR: data generation mode cannot be {self.mode}')

        self.count += 1
        if action is None:
            action = self.data_env.action_space.sample()
        assert action is not None, "Data loader batch collator in torch requires dtypes to be not None"

        item_dict = {}
        item_dict["previous_obs"] = obs
        item_dict["previous_state"] = state
        item_dict["previous_norm_state"] = norm_state
        item_dict["action"] = action
        for key in item_dict.keys():
            assert item_dict[key] is not None, f"{key} is None in dataset."
        return item_dict


class RandomPolicy:

    def __init__(self, env):

        self.action_space = env.action_space
    
    def predict(self, obs, deterministic=True):

        return self.action_space.sample(), None



if __name__ == "__main__":

    dataset = CustomDataset('configs/data_generator/config.yaml', limit=30, mode='cont')

    pdb.set_trace()
    from PIL import Image

    os.makedirs('./temp', exist_ok = True)

    for i, (obs, norm_state, state, action) in enumerate(dataset):

        img = Image.fromarray(obs)
        img.save(f'./temp/{i+1}.png')
        print(i+1)
        print(dataset.count)
        print(dataset.limit)
        print('-------')
