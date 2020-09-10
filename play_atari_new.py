import os
os.environ['SDL_AUDIODRIVER'] = 'dsp'

import sys
import gym
import random
import numpy as np
import pickle

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.tune.registry import register_env, register_trainable
from ray.rllib.utils import try_import_tf
from ray.rllib.env import PettingZooEnv

from pettingzoo.utils import observation_saver
from pettingzoo.atari import boxing_v0, combat_tank_v0, joust_v0, surround_v0, space_invaders_v0
from supersuit.aec_wrappers import clip_reward, sticky_actions, resize
from supersuit.aec_wrappers import frame_skip, frame_stack, agent_indicator

from numpy import float32

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents.ppo import PPOTrainer

from skimage.io import imsave

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
                8,
                strides=4,
                activation="relu",
                data_format='channels_last')(inputs)
        layer2 = tf.keras.layers.Conv2D(
                64,
                4,
                strides=2,
                activation="relu",
                data_format='channels_last')(layer1)
        layer3 = tf.keras.layers.Conv2D(
                64,
                3,
                strides=1,
                activation="relu",
                data_format='channels_last')(layer2)
        layer4 = tf.keras.layers.Flatten()(layer3)
        concat_layer = tf.keras.layers.Concatenate()([layer4, inputs2])
        layer5 = tf.keras.layers.Dense(512, activation="relu")(concat_layer)
        action = tf.keras.layers.Dense(
                num_outputs,
                activation="linear",
                name="actions")(layer5)
        value_out = tf.keras.layers.Dense(
                1,
                activation=None,
                name="value_out",
                kernel_initializer=normc_initializer(0.01))(layer5)
        self.base_model = tf.keras.Model([inputs,inputs2], [action, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model([input_dict["obs"][:,:,:,0:4],input_dict["obs"][:,0,0,4:6]])
        return model_out, state
    
    def value_function(self):
        return tf.reshape(self._value_out, [-1])

if __name__ == "__main__":
    # RDQN - Rainbow DQN
    # ADQN - Apex DQN
    
    methods = ["ADQN", "PPO", "RDQN"]
    
    assert len(sys.argv) == 4, "Input the learning method as the second argument"
    env_name = sys.argv[1].lower()
    method = sys.argv[2].upper()
    checkpoint = sys.argv[3] 
    assert method in methods, "Method should be one of {}".format(methods)
    
    #checkpoint_path = "../ray_results_base/"+env_name+"/"+method.upper()+"/checkpoint_980/checkpoint-980"
    #checkpoint_path = "../ray_results_base/"+env_name+"/"+method.upper()+'/APEX_boxing_0_2020-08-26_19-03-06prr7aba9'+"/checkpoint_2430/checkpoint-2430"
    checkpoint_path = "./ray_results/{}/{}/v1/checkpoint_{}/checkpoint-{}".format(env_name,method,checkpoint, checkpoint)
    
    if method == "RDQN":
        Trainer = DQNTrainer
    elif method == "ADQN":
        Trainer = ApexTrainer
    elif method == "PPO":
        Trainer = PPOTrainer

    if env_name=='boxing':
        game_env = boxing_v0
    elif env_name=='combat_tank':
        game_env = combat_tank_v0
    elif env_name=='joust':
        game_env = joust_v0
    elif env_name=='surround':
        game_env = surround_v0
    elif env_name=='space_invaders':
        game_env = space_invaders_v0
    else:
        raise TypeError('{} environment not supported!'.format(env_name))

    def env_creator(args):
        env = game_env.env(obs_type='grayscale_image')
        #env = clip_reward(env, lower_bound=-1, upper_bound=1)
        #env = sticky_actions(env, repeat_action_probability=0.25)
        env = resize(env, 84, 84)
        #env = color_reduction(env, mode='full')
        #env = frame_skip(env, 4)
        env = frame_stack(env, 4)
        env = agent_indicator(env, type_only=False)
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

    # get the config file - params.pkl
    config_path = os.path.dirname(checkpoint_path)
    config_path = os.path.join(config_path, "../params.pkl")
    with open(config_path, "rb") as f:
        config = pickle.load(f)

    ray.init()

    RLAgent = Trainer(env=env_name, config=config)
    RLAgent.restore(checkpoint_path)

    # init obs, action, reward
    env = env_creator(0)
    observation = env.reset()
    
    rewards = [0]
    total_reward = 0
    done = False
    iteration = 0
    policy_agent = 'first_0'
    while not done:
        
        #imsave("./"+str(iteration)+".png",np.reshape(observation,(30,50))) 
        #env.render()
        if env.agent_selection == policy_agent:
            observation = env.observe(policy_agent)
            action, _, _ = RLAgent.get_policy("policy_0").compute_single_action(observation, prev_reward=rewards[-1]) # prev_action=action_dict[agent_id]
        else:
            action = env.action_spaces[policy_agent].sample() #same action space for all agents
        print('Agent: {}, action: {}'.format(env.agent_selection,action))
        env.step(action, observe=False)
        print('reward: {}, done: {}'.format(env.rewards, env.dones))
        reward = env.rewards[policy_agent]
        rewards.append(reward)
        done = any(env.dones.values())
        total_reward += reward
        iteration += 1

    #env.close()
    print("done", done, total_reward)

