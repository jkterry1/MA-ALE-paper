import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import sys
import gym
import random
import numpy as np

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from ray.rllib.env import PettingZooEnv
from pettingzoo.atari import boxing_v0, combat_tank_v0, joust_v1, surround_v0, space_invaders_v0
from supersuit import clip_reward_v0, sticky_actions_v0, resize_v0
from supersuit import frame_skip_v0, frame_stack_v0, agent_indicator_v0, flatten_v0

#from cyclic_reward_wrapper import cyclic_reward_wrapper

tf1, tf, tfv = try_import_tf()

class AtariModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="atari_model"):
        super(AtariModel, self).__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        inputs = tf.keras.layers.Input(shape=(84,84,4), name='observations')
        inputs2 = tf.keras.layers.Input(shape=(2,), name='agent_indicator')
        # Convolutions on the frames on the screen
        layer1 = tf.keras.layers.Conv2D(
                32,
                [8, 8],
                strides=(4, 4),
                activation="relu",
                data_format='channels_last')(inputs)
        layer2 = tf.keras.layers.Conv2D(
                64,
                [4, 4],
                strides=(2, 2),
                activation="relu",
                data_format='channels_last')(layer1)
        layer3 = tf.keras.layers.Conv2D(
                64,
                [3, 3],
                strides=(1, 1),
                activation="relu",
                data_format='channels_last')(layer2)
        layer4 = tf.keras.layers.Flatten()(layer3)
        concat_layer = tf.keras.layers.Concatenate()([layer4, inputs2])
        layer5 = tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(concat_layer)
        action = tf.keras.layers.Dense(
                num_outputs,
                activation="linear",
                name="actions",
                kernel_initializer=normc_initializer(0.01))(layer5)
        value_out = tf.keras.layers.Dense(
                1,
                activation=None,
                name="value_out",
                kernel_initializer=normc_initializer(0.01))(layer5)
        self.base_model = tf.keras.Model([inputs, inputs2], [action, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model([input_dict["obs"][:,:,:,0:4], input_dict["obs"][:,0,0,4:6]])
        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])


if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    
    methods = ["ADQN", "PPO", "RDQN"]
    
    assert len(sys.argv) == 3, "Input the learning method as the second argument"
    env_name = sys.argv[1].lower()
    method = sys.argv[2]
    assert method in methods, "Method should be one of {}".format(methods)
    
    if env_name=='boxing':
        game_env = boxing_v0
    elif env_name=='combat_tank':
        game_env = combat_tank_v0
    elif env_name=='joust':
        game_env = joust_v1
    elif env_name=='surround':
        game_env = surround_v0
    elif env_name=='space_invaders':
        game_env = space_invaders_v0
    else:
        raise TypeError("{} environment not supported!".format(game_env))

    def env_creator(args):
        env = game_env.env(obs_type='grayscale_image')
        env = clip_reward_v0(env, lower_bound=-1, upper_bound=1)
        env = sticky_actions_v0(env, repeat_action_probability=0.25)
        env = resize_v0(env, 84, 84)
        #env = color_reduction_v0(env, mode='full')
        #env = frame_skip_v0(env, 4)
        env = frame_stack_v0(env, 4)
        env = agent_indicator_v0(env, type_only=False)
        #env = flatten_v0(env)
        return env
    
    register_env(env_name, lambda config: PettingZooEnv(env_creator(config)))
    
    test_env = PettingZooEnv(env_creator({}))
    obs_space = test_env.observation_space
    act_space = test_env.action_space
    
    ModelCatalog.register_custom_model("AtariModel", AtariModel)
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "AtariModel",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)
    policies = {"policy_0": gen_policy(0)}
    
    # for all methods
    policy_ids = list(policies.keys())

    if method == "A2C":
        tune.run(
            "A2C",
            name="A2C",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_base/"+env_name,
            config={
        
                # Enviroment specific
                "env": env_name,
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": False,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                "lr_schedule": [[0, 0.0007],[20000000, 0.000000000001]],
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "ADQN":
        # APEX-DQN
        tune.run(
            "APEX",
            name="ADQN",
            stop={"episodes_total": 600000},
            checkpoint_freq=10,
            local_dir="~/ray_results_atari/"+env_name,
            config={
        
                # Enviroment specific
                "env": env_name,
                "double_q": True,
                "dueling": True,
                "num_atoms": 1,
                "noisy": False,
                "prioritized_replay": False,
                "n_step": 1,
                #"lr": 0.0001,
                "lr": 0.0000625,
                "adam_epsilon": 0.00015,
                "buffer_size": int(1e6),
                "exploration_config": {
                    "final_epsilon": 0.01,
                    "epsilon_timesteps": 200000,
                },
                "prioritized_replay_alpha": 0.5,
                "final_prioritized_replay_beta": 1.0,
                "prioritized_replay_beta_annealing_timesteps": 2000000,

                "num_gpus": 1,
                
                "log_level": "INFO",
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "rollout_fragment_length": 4,
                "train_batch_size": 32,
                "target_network_update_freq": 8000,
                "timesteps_per_iteration": 10000,
                "learning_starts": 20000,
                "compress_observations": True, 
                "gamma": 0.99,
                # Method specific
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "DQN":
        # plain DQN
        tune.run(
            "DQN",
            name="DQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_base/"+env_name,
            config={
                # Enviroment specific
                "env": env_name,
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
                # Method specific
                "dueling": False,
                "double_q": False,
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "IMPALA":
        tune.run(
            "IMPALA",
            name="IMPALA",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_base/"+env_name,
            config={
        
                # Enviroment specific
                "env": env_name,
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                "clip_rewards": True,
                "lr_schedule": [[0, 0.0005],[20000000, 0.000000000001]],
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    elif method == "PPO":
        tune.run(
            "PPO",
            name="PPO",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            local_dir="~/ray_results_atari/"+env_name,
            config={
        
                # Enviroment specific
                "env": env_name,
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": False,
                "gamma": .99,
        
        
                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 5000,
                "rollout_fragment_length": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "batch_mode": 'truncate_episodes',
                #"observation_filter": 'NoFilter',
                #"vf_share_layers": True,
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )

    # pseudo-rainbow DQN
    elif method == "RDQN":
        tune.run(
            "DQN",
            name="RDQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=100,
            local_dir="~/ray_results_atari/"+env_name,
            config={
        
                # Enviroment specific
                "env": env_name,
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 10000,
                "buffer_size": int(1e5),
                #"compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
                "lr": 0.0001,
                "exploration_config": {
                    "epsilon_timesteps": 2,
                    "final_epsilon": 0.0,
                },
                "target_network_update_freq": 500,
                # Method specific
                "num_atoms": 51,
                "dueling": True,
                "double_q": True,
                "n_step": 3,
                #"batch_mode": "complete_episodes",
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.5,
                "final_prioritized_replay_beta": 1.0,
                "prioritized_replay_beta_annealing_timesteps": 400000,
                #"gpu": True,

                # # alternative 1
                #"noisy": True,
                # # alternative 2
                "parameter_noise": True,

                # based on expected return
                "v_min": 0,
                "v_max": 150,
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )
