"""
some copied from EvalCallback. modified by waymao
"""
import os
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, sync_envs_normalization
from stable_baselines3.common.evaluation import evaluate_policy
from models.policy_head.vec_video_recorder import VecVideoRecorder
from collections import deque
from types import SimpleNamespace
import numpy as np
from gymnasium import error, logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn
from typing import Callable, List
from matplotlib import pyplot as plt
import gymnasium as gym
import pandas as pd 
import pdb
import torch
import seaborn as sns
import wandb

#NOTE: for older RewardValueCallback used as global variable building CSV
csv_logger = pd.DataFrame(columns=['train/test', 'step', 'seed', 'cumul_reward'])


class InitializeLogsCallback(BaseCallback):
    def __init__(self, verbose=0, max_steps=100):
        super(InitializeLogsCallback, self).__init__(verbose)
        self.max_steps = max_steps
    
    def _on_step(self):
        return True

    def _on_training_start(self) -> None:
        #NOTE: used to align the rollout ep_rew_mean, ep_len_mean and success_rate plots s.t. they start from 0.0
        self.logger.record("rollout/ep_rew_mean", 0.0)
        self.logger.record("rollout/ep_len_mean", self.max_steps)
        self.logger.record("time/fps", 0.0)
        self.logger.record("time/time_elapsed", 0, exclude="tensorboard")
        self.logger.record("time/total_timesteps", 0.0, exclude="tensorboard")
        self.logger.record("rollout/success_rate", 0.0)
        self.logger.dump(step=0)

class AdvantageLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(AdvantageLoggerCallback, self).__init__(verbose)
    
    def _on_step(self):
        return True
        
    def _on_rollout_end(self):
        if 'advantages' in dir(self.model.rollout_buffer):
            advantages = self.model.rollout_buffer.advantages
            mean_adv = advantages.mean()
            std_adv = advantages.std()
            
            self.logger.record("advantage/adv_mean", mean_adv)
            self.logger.record("advantage/adv_std", std_adv)

class CustomVideoRecorder(VecVideoRecorder):

    """
    Custom Video Recorder wrapper for SB3 with following extra features:
    (1) stops video when policy is done 
    """
    
    def __init__(self, venv: VecEnv, video_folder: str, record_video_trigger: Callable[[int], bool], video_length: int = 200, name_prefix: str = "rl-video"):
        super(CustomVideoRecorder, self).__init__(venv, video_folder, record_video_trigger, video_length, name_prefix)

        #[SR] extra variable to track when different environments are "done"
        self.dones = np.full((venv.num_envs,), False)

        #[SR] extra variable to track last observation: to preserve in case some envs are "done" before others
        self.terminal_obs = [None for _ in range(venv.num_envs)]
    
 
    def step_wait(self) -> VecEnvStepReturn:
        
        
        obs, rewards, dones, infos = self.env.step_wait()

        #[SR] boolean or to set self.dones true: stop recording when all envs reach done
        # self.dones = np.logical_or(self.dones, dones)

        # #[SR] in case some dones are True, store in self.terminal_obs and reset the obs
        # for i, done in enumerate(dones):

        #     if done:
        #         self.terminal_obs[i] = obs[i,:,:,:]
        
        # for i in range(len(obs)):

        #     if isinstance(self.terminal_obs[i], np.ndarray):
        #         obs[i,:,:,:] = self.terminal_obs[i]

        self.step_id += 1
        if self.recording:

            self._capture_frame()
            if (self.dones.all() == True) or (len(self.recorded_frames) > self.video_length):
                print(f"Saving video to {self.video_path}")
                self._stop_recording()
                
        elif self._video_enabled():
            self._start_video_recorder()

        return obs, rewards, dones, infos

class CustomEvalCallback(EvalCallback):
    """
    Custom Evaluation Callback for Stable Baselines that supports custom
    name for logging.
    """
    def __init__(self, custom_name, eval_env, max_steps = 100, gamma=None, *args, **kwargs):
        #TODO: chekc kwargs input into model, correctly passed to super() EvalCallback
        self.custom_name = custom_name
        self.gamma = gamma
        self.max_steps = max_steps
        super(CustomEvalCallback, self).__init__(eval_env, *args, **kwargs)

        self.mean_evaluations_results = []
        self.std_evaluations_results = []
        self.mean_evaluations_length = []
        self.std_evaluations_length = []
        self.best_mean_reward = float('-inf')

    def _on_training_start(self) -> None:
        # Force log at step 0
        self.logger.record(f"eval/{self.custom_name}/mean_reward", 0.0)
        self.logger.record(f"eval/{self.custom_name}/std_reward", 0.0)
        self.logger.record(f"eval/{self.custom_name}/mean_ep_length", self.max_steps)
        self.logger.record(f"eval/{self.custom_name}/std_ep_length", 0.0)
        self.logger.record(f"eval/{self.custom_name}/success_rate", 0.0)
        self.logger.record(f"time/{self.custom_name}/total_timesteps", 0)
        self.logger.dump(0)
        
    def _on_step(self) -> bool:
        # NOTE: this function is identical to the original _on_step function
        # but with the addition of the custom name for logging
        continue_training = True



        #[SR] divided by self.eval_env.num_envs for periodic logging
        if self.eval_freq > 0 and (self.n_calls % (self.eval_freq // self.eval_env.num_envs))== 0:
            
            #NOTE:reseed the vecenv environment for better variance when running custom evaluations
            random_seed = np.random.randint(0, 1000000)
            self.eval_env.env_method("reset", seed=random_seed)
            
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError as e:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    ) from e

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.mean_evaluations_results.append(np.mean(episode_rewards))
                self.std_evaluations_results.append(np.std(episode_rewards))
                self.mean_evaluations_length.append(np.mean(episode_lengths))
                self.std_evaluations_length.append(np.std(episode_lengths))

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    mean_results = self.mean_evaluations_results,
                    std_results = self.std_evaluations_results,
                    ep_lengths=self.evaluations_length,
                    mean_ep_lengths = self.mean_evaluations_length,
                    std_ep_lengths = self.std_evaluations_length,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            if self.gamma is not None:
                mean_discounted_return = np.mean(episode_rewards * np.power(self.gamma, episode_lengths))
                std_discounted_return = np.std(episode_rewards * np.power(self.gamma, episode_lengths))
            self.last_mean_reward = mean_reward

            if self.verbose >= 1:
                print("=----------------------------=")
                print(f"{self.custom_name}: num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current Logger
            self.logger.record(f"eval/{self.custom_name}/mean_reward", float(mean_reward))
            self.logger.record(f"eval/{self.custom_name}/std_reward", float(std_reward))
            self.logger.record(f"eval/{self.custom_name}/mean_ep_length", mean_ep_length)
            self.logger.record(f"eval/{self.custom_name}/std_ep_length", std_ep_length)
            if self.gamma is not None:
                self.logger.record(f"eval/{self.custom_name}/mean_discounted_return", float(mean_discounted_return))
                self.logger.record(f"eval/{self.custom_name}/std_discounted_return", float(std_discounted_return))



            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose >= 1:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                    print("=----------------------------=")

                self.logger.record(f"eval/{self.custom_name}/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record(f"time/{self.custom_name}/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback on new best model, if needed
                if self.callback_on_new_best is not None:
                    continue_training = self.callback_on_new_best.on_step()

            # Trigger callback after every evaluation, if needed
            if self.callback is not None:
                continue_training = continue_training and self._on_event()

        return continue_training

class ValuePlottingCallback(BaseCallback):

    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, num_envs: int, name_prefix: str = "", verbose=1):
        super(ValuePlottingCallback, self).__init__(verbose)

        #[SR] dummy base env to generate value function
        self.base_env = env
        self.save_freq = save_freq
        self.log_dir = log_dir
        self.name_prefix = name_prefix


        #[SR] add separate video folder for value logging
        self.value_folder = os.path.join(log_dir, 'value_functions')
        os.makedirs(self.value_folder, exist_ok=True)

        

        self.num_envs = num_envs
        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:
        
        # Save a GIF every save_freq steps
        if self.n_calls % (self.save_freq//self.num_envs) == 0:
            self._plot_value_func()
            
        return True
   
    def _plot_value_func(self):
        '''extract the value function for a particular observation'''
        
        original_obs, info = self.base_env.reset()

        width = self.base_env.env.env.unwrapped.width
        height = self.base_env.env.env.unwrapped.height

        # Detect the rendered observation shape
        obs_img = info['obs']  # This should be the RGB image (H, W, C)
        img_height, img_width = obs_img.shape[:2]

        # Compute the scaling factor: how many pixels per grid cell?
        scale_x = img_width / width
        scale_y = img_height / height

        value_function = np.full((width, height), np.nan)

        for w in range(self.base_env.env.env.unwrapped.width):
            for h in range(self.base_env.env.env.unwrapped.height):

                obj = self.base_env.env.env.unwrapped.grid.get(w, h)

                if obj is None:

                    # place the agent here
                    self.base_env.env.env.unwrapped.agent_pos = (w, h)

                    value_estim = []
                    
                    for dir in range(4):
                        
                        self.base_env.env.env.unwrapped.agent_dir = dir

                        obs = self.base_env.get_curr_obs()
                        
                        obs_tensor, vector_env = self.model.policy.obs_to_tensor(obs)

                        if callable(getattr(self.model.policy, "predict_values", None)):
                            value = self.model.policy.predict_values(obs_tensor).item()
                        elif callable(getattr(self.model, "q_net", None)):
                            value = torch.sum(self.model.q_net(obs_tensor), dim=1).item()

                        value_estim.append(value)

                    value_function[w, h] = np.mean(value_estim)
        
        
        #normalize the value function so it is 0-1
        value_function = value_function / np.nansum(value_function)

        #plotting value function over the observation
        fig = plt.figure(frameon=False)
        plt.imshow(info['obs'])
        plot_extent = [0, width * scale_x, height * scale_y, 0]
        plt.imshow(value_function.T, alpha=0.5, cmap='viridis', extent = plot_extent)
        plt.colorbar()
        plt.title(f"Value Function @ Step {self.num_timesteps}")


        value_name = f"{self.name_prefix}-step-{self.num_timesteps}.png"
        value_path = os.path.join(self.value_folder, value_name)

        plt.savefig(value_path)
        plt.close()

    def _on_training_end(self):
        plt.close()

class RewardValueCallback(BaseCallback):
    def __init__(self, env: gym.Env, save_freq: int, log_dir: str, csv_log_dir: str, num_envs:int, verbose=0, train=True):
        super(RewardValueCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.max_steps = env.env.max_steps
        self.env = env
        self.save_freq = save_freq
        self.train = train

        self.num_envs = num_envs

        self.csv_log_dir = os.path.join(csv_log_dir, 'episodic_reward_logs.csv')


        os.makedirs(csv_log_dir, exist_ok=True)

        os.makedirs(self.log_dir, exist_ok=True)

    def _on_step(self) -> bool:

        # Save a GIF every save_freq steps
        if self.n_calls % (self.save_freq//self.num_envs) == 0:
            self._log_rewards_and_value()

            global csv_logger
            csv_logger.to_csv(self.csv_log_dir, index=False)
            

        return True
    
    
    def _log_rewards_and_value(self):
        # Reset the environment
        obs, info = self.env.reset()
        dones = False
        step = 0
        cumulative_reward = 0


        while not dones and step < self.max_steps:
            # Predict action and get value function (self.model.predict returns the action and value)
            actions, _states = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, __, infos = self.env.step(actions)

            # Log the value function and reward to TensorBoard
            cumulative_reward += rewards
            self.writer.add_scalar(f"{'train' if self.train else 'test'} reward per step", rewards, self.num_timesteps + step)
            step += 1

        
        #NOTE: error check -- ensuring cumulative reward isn't > 1 or < 0
        assert (cumulative_reward <= 1.0 and cumulative_reward >= 0.0), f'ERROR: cumulative reward is {cumulative_reward}'

        # Log the cumulative reward
        self.writer.add_scalar(f"{'train' if self.train else 'test'} cumul reward", cumulative_reward, self.num_timesteps)
        # self.writer.add_scalar(f"{'train' if self.train else 'test'} avg reward", cumulative_reward/step, self.num_timesteps)

        # Print the logged rewards
        print("=-------------------------=")
        print(f"{'train' if self.train else 'test'} cumul reward @ {self.num_timesteps}: ", cumulative_reward)
        # print(f"{'train' if self.train else 'test'} avg reward @ {self.num_timesteps}: ", cumulative_reward/step)

        #log the cumulative rewards to csv file
        global csv_logger
        csv_logger = pd.concat([pd.DataFrame([{'train/test': 'train' if self.train else 'test', 'step': self.num_timesteps, 'seed': self.model.seed, 'cumul_reward': cumulative_reward}]), csv_logger], ignore_index=True)
        
    
    def _on_training_end(self):
        self.writer.close()

class _AuxRingBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = deque(maxlen=capacity)

    def __len__(self):
        return len(self.data)

    def add(self, obs, next_obs, action=None, done=None, info=None, labels=None):
        self.data.append({
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "done": done,
            "info": info,
            "labels": labels
        })

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.choice(len(self.data), size=batch_size, replace=False)
        samples = [self.data[i] for i in idx]

        def to_tensor(x):
            # x may be numpy/torch/scalar
            if isinstance(x, torch.Tensor):
                return x
            return torch.as_tensor(x)

        def stack(items):
            if isinstance(items[0], dict):
                return {k: stack([it[k] for it in items]) for k in items[0].keys()}
            ts = [to_tensor(it) for it in items]
            return torch.stack(ts, dim=0).to(device)

        obs_b = stack([s["obs"] for s in samples])
        next_obs_b = stack([s["next_obs"] for s in samples])

        # Optional fields with safe defaults
        if samples[0]["action"] is None:
            actions_b = None
        else:
            actions_b = stack([s["action"] for s in samples])

        if samples[0]["done"] is None:
            dones_b = None
        else:
            dones_b = stack([s["done"] for s in samples]).reshape(batch_size, -1)

        infos_b = [s["info"] for s in samples]

        return SimpleNamespace(
            observations=obs_b,
            next_observations=next_obs_b,
            actions=actions_b,
            dones=dones_b,
            infos=infos_b,
        )


    '''
    Callback for training custom auxiliary losses for encoders directly using SAC's
    replay buffer 
    '''
    def __init__(
        self, 
        custom_name: str,
        num_envs: int,
        train_every: int = 256,
        batch_size: int = 256,
        aux_loss_updates: int = 3,
        verbose: int = 0,
    ):
        super(CustomLossCallback, self).__init__(verbose)
        #custom naming for logged metric (i.e. loss)
        self.custom_name = custom_name
        self.batch_size = batch_size
        self.train_every = train_every//num_envs
        self.aux_loss_updates = aux_loss_updates
        self.steps_since_update = 0

    
    def _init_callback(self) -> None:    
        #store buffer of expert state and observation
        self.observation_buffer = []
    
    def log_heatmap(self, matrix, key_name, step):
        fig, ax = plt.subplots()
        sns.heatmap(matrix.detach().cpu().numpy(), ax=ax, cmap="viridis", cbar=True)
        # Use wandb directly here — self.logger cannot log images
        wandb.log({key_name: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def _on_step(self)->bool:
        self.steps_since_update += 1

        if (self.steps_since_update >= self.update_freq 
            and len(self.replay_buffer) >= max(self.batch_size, self.train_every)
            and 'compute_loss' in dir(self.model)):
            for _ in range(self.aux_loss_updates):
                batch = replay_buffer.sample(self.batch_size, env=self.model._vec_normalize_env)
                # batch usually has: observations, next_observations, actions, rewards, dones
                total_loss, specific_losses = self.encoder.compute_loss(batch)
                # backpropagate the loss function
                self.model.features_extractor_class.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                self.model.features_extractor_class.optimizer.step()
                
                # log the loss during training features extractor
                self.logger.record(f"custom/{self.custom_name}/loss", float(total_loss.item()))
                for (metric, metric_val, visualize) in specific_losses.items():
                    if not visualize:
                        self.logger.record(f"custom/{self.custom_name}/{metric}", float(metric_val.item()))
                    else:
                        self.log_heatmap(metric_val, key_name=f"custom/{metric}", step=self.num_timesteps)
            self.steps_since_update = 0
            self.replay_buffer = []
        else:
            
            for env in range(len(self.locals['infos'])):
                self.replay_buffer.append(torch.Tensor(self.locals['infos'][env]['obs']).permute(2, 0, 1))
            
        return True
  
    """
    Callback for training the encoder with a supervised objective
    directly using PPO's rollout buffer.
    """
    def __init__(
        self, 
        custom_name: str,        
        update_freq: int = 256,
        batch_size: int = 256,
        learning_rate: float = 5e-5,
        verbose: int = 0,
    ):
        super(SupervisedEncoderCallback, self).__init__(verbose)
        self.update_freq = update_freq//4
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.steps_since_update = 0

        #custom naming for logged metric (i.e. loss)
        self.custom_name = custom_name
        
    def _init_callback(self) -> None:
        
        #defin the loss function to use during training
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
        # Bring the features_extractor to the right device and create an optimizer for the model
        self.model.policy.features_extractor.to(self.model.device)
        self.optimizer = torch.optim.Adam(
            self.model.policy.features_extractor.parameters(),
            lr=self.learning_rate
        )
        #collect buffer of expert states as labels 
        self.expert_state_buffer = []
        self.observation_buffer = []
        
    def _on_step(self) -> bool:
        
        self.steps_since_update += 1
        
        # Perform supervised update at specified frequency
        # and only after buffer has been filled at least once
        if (self.steps_since_update >= self.update_freq and 
            hasattr(self.model, 'rollout_buffer') and 
            self.model.rollout_buffer is not None):
            
            self._update_features_extractor_from_buffer()
            self.steps_since_update = 0
            self.expert_state_buffer = []
            self.observation_buffer = []
        else:
            for env in range(len(self.locals['infos'])):
                sublisted_expert_state = list(self.locals['infos'][env]['state_dict'].values())
                self.expert_state_buffer.append(torch.Tensor([item for sublist in sublisted_expert_state for item in (sublist if isinstance(sublist, tuple) else [sublist])]).to(torch.int64))
                self.observation_buffer.append(torch.Tensor(self.locals['infos'][env]['obs']).permute(2, 0, 1))
        
        return True
    
    def _update_features_extractor_from_buffer(self):
        """Train the encoder using data from PPO's rollout buffer"""
        #in case there are no observations in observation buffer, skip
        if len(self.observation_buffer) == 0:
            return 
        
        observations = torch.stack(self.observation_buffer)
        buffer_size = len(observations)
        
        # process the ground truth labels
        labels = torch.stack(self.expert_state_buffer)
        
        # Convert to tensors
        observations = torch.as_tensor(observations).float().to(self.model.device)
        #NOTE: needs to be 1-hot encoded vectors for the label
        labels = torch.as_tensor(labels).to(self.model.device)
        
        # Forward pass
        with torch.set_grad_enabled(True):
            pred_features = self.model.policy.features_extractor(observations, test=False)
            loss = 0
            accuracy = 0

            for i, feat in enumerate(pred_features):
                loss += self.loss_fn(feat, labels[:,i])
                # Get predicted class indices
                preds = torch.argmax(feat, dim=1)
                accuracy += (preds == labels[:,i]).float().mean()
            loss /= (len(pred_features))
            accuracy /= (len(pred_features))
        
        # Backward pass and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log the loss during training features extractor
        self.logger.record(f"supervised/{self.custom_name}/loss", float(loss.item()))
        self.logger.record(f"supervised/{self.custom_name}/acc", float(accuracy.item()))

class AuxiliaryLossCallback(BaseCallback):
    """
    Callback for training custom auxiliary losses for encoders using a custom
    auxiliary ring buffer (stores obs/next_obs/action/done/info).
    """

    def __init__(
        self,
        custom_name: str,
        num_envs: int,
        train_every: int = 256,
        batch_size: int = 256,
        learning_rate: float = 2e-4,
        aux_loss_updates: int = 3,
        buffer_capacity: int = 100_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)

        self.custom_name = custom_name
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Interpret train_every in *env steps*; convert to callback calls (which are per env step across vec envs)
        self.train_every = max(1, train_every // max(1, num_envs))
        self.aux_loss_updates = aux_loss_updates
        self.buffer_capacity = buffer_capacity

        self.steps_since_update = 0

    def _init_callback(self) -> None:
        self.data_buffer = _AuxRingBuffer(capacity=self.buffer_capacity)
        self.optimier_dict = self.model.policy.feature_extractor.build_optimizers()
       
    def log_heatmap(self, matrix, key_name, step):
        fig, ax = plt.subplots()
        sns.heatmap(matrix.detach().cpu().numpy(), ax=ax, cmap="viridis", cbar=True)
        wandb.log({key_name: wandb.Image(fig)}, step=step)
        plt.close(fig)

    def _on_step(self) -> bool:
        """
        1) Push latest transitions (including infos) into aux ring buffer
        2) Every train_every steps, sample batches from aux buffer and update encoder
        """
        # ----------------------------
        # 1) Collect transition(s)
        # ----------------------------
        # SB3 usually provides these in callback locals (names can vary by algo/version/wrappers)
        obs = self.locals.get("obs", None)
        next_obs = self.locals.get("new_obs", None) or self.locals.get("next_obs", None)
        actions = self.locals.get("actions", None)
        dones = self.locals.get("dones", None)
        infos = self.locals.get("infos", None)
        labels = []
        for i in range(len(infos)):
            sublisted_expert_state = list(self.locals['infos'][i]['state_dict'].values())
            labels.append(
                torch.Tensor([item for sublist in sublisted_expert_state for item in (sublist if isinstance(sublist, tuple) else [sublist])]).to(torch.int64)
            )  

        # Normalize infos to list-of-dicts (vec env) or [dict] (single env)
        if infos is None:
            infos = [None]
        elif isinstance(infos, dict):
            infos = [infos]

        # Helper to index potentially-batched structures
        def _at(x, i):
            if x is None:
                return None
            if isinstance(x, dict):
                return {k: _at(v, i) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return x[i]
            try:
                return x[i]
            except Exception:
                return x

        # Only add if we have a valid transition
        if obs is not None and next_obs is not None and actions is not None:
            # Determine n_envs for this step
            n_envs = len(infos)

            for i in range(n_envs):
                self.data_buffer.add(
                    obs=_at(obs, i),
                    next_obs=_at(next_obs, i),
                    action=_at(actions, i),
                    done=_at(dones, i),
                    info=infos[i] if i < len(infos) else None,
                    labels=_at(labels, i)
                )

        # ----------------------------
        # 2) Periodic encoder updates
        # ----------------------------

        # Train only at frequency, and only if we have enough samples
        if self.steps_since_update < self.train_every:
            return True
        if len(self.aux_buffer) < self.batch_size:
            return True
        if not hasattr(self.model.policy.feature_extractor, "compute_loss"):
            return True
        self.steps_since_update += 1
        device = self.model.device

        feature_extractor = self.model.policy.feature_extractor

        for _ in range(self.aux_loss_updates):
            batch = self.aux_buffer.sample(self.batch_size, device=device)
            total_loss, specific_metrics = feature_extractor.compute_loss(batch)

            for _, optimizer in self.optimizer_dict.items():
                optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            for _, optimizer in self.optimizer_dict.items():
                optimizer.step()
            if hasattr(feature_extractor, "post_step") and callable(feature_extractor.aux_post_step):
                post_step_metrics = feature_extractor.post_step()
            specific_metrics = specific_metrics | post_step_metrics

            # logging all metrics
            self.logger.record(f"custom/{self.custom_name}/loss", float(total_loss.detach().cpu().item()))
            for metric, (metric_val, visualize) in specific_metrics.items():
                if not visualize:
                    self.logger.record(
                        f"custom/{self.custom_name}/{metric}",
                        float(metric_val.detach().cpu().item()),
                    )
                else:
                    self.log_heatmap(metric_val, key_name=f"custom/{metric}", step=self.num_timesteps)

        self.steps_since_update = 0
        return True
