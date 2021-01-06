from timeit import default_timer as timer
import numpy as np
from .writer import ExperimentWriter
from .experiment import Experiment
import os

class MultiagentEnvExperiment():
    '''An Experiment object for training and testing agents that interact with one environment at a time.'''
    def __init__(
            self,
            preset,
            env,
            render=False,
            quiet=False,
            write_loss=True,
            steps_per_save=None,
            name=None
    ):
        if name is None:
            name = preset.__class__.__name__
        self._writer = ExperimentWriter(self, name, env.name, loss=write_loss)
        self._writers = {
            agent : ExperimentWriter(self, "{}_{}".format(name, agent), env.name, loss=write_loss)
            for agent in env.agents
        }
        self._name = name
        self._preset = preset
        self._agent = preset.agent(writer=self._writers,train_steps=float("inf"))
        self._env = env
        self._render = render
        self._frame = 1
        self._episode = 1
        self._steps_per_save = steps_per_save

        if render:
            self._env.render(mode="human")

    @property
    def frame(self):
        return self._frame

    @property
    def episode(self):
        return self._episode

    def train(self, frames=np.inf, episodes=np.inf):
        while not self._done(frames, episodes):
            self._run_training_episode()

    def test(self, frames=100000, episodes=100):
        returns = []
        num_steps = 0
        for episode in range(episodes):
            episode_return, ep_steps = self._run_test_episode()
            returns.append(episode_return)
            num_steps += ep_steps
            self._log_test_episode(episode, episode_return)
        return returns

    def _run_training_episode(self):
        # initialize timer
        start_time = timer()
        start_frame = self._frame

        # initialize the episode
        state = self._env.reset()
        returns = {agent : 0 for agent in self._env.agents}

        for agent in self._env.agent_iter():
            action = self._agent.act(agent, state)
            state = self._env.step(action)
            returns[self._env._env.agent_selection] += state.reward
            if self._steps_per_save is not None and self._frame % self._steps_per_save == 0:
                save_path = os.path.join(self._writer.log_dir,"checkpoint{}.pth".format(self._frame))
                print("saved at:",save_path)
                self._preset.save(save_path)
            self._frame += 1
            if len(self._env._env.agents) == 1 and self._env._env.dones[self._env._env.agent_selection]:
                break
        # stop the timer
        end_time = timer()
        fps = (self._frame - start_frame) / (end_time - start_time)

        # log the results
        self._log_training_episode(returns, fps)

        # update experiment state
        self._episode += 1

    def _log_training_episode(self, returns, fps):
        print('returns: {}'.format(returns))
        print('fps: {}'.format(fps))
        for agent in self._env.agents:
            self._writers[agent].add_evaluation('returns/frame', returns[agent], step="frame")

    def _run_test_episode(self):
        # initialize the episode
        state = self._env.reset()
        action = self._agent.eval(state)
        returns = {agent: 0 for agent in self._anv.agents}
        num_steps = 0

        # loop until the episode is finished
        while not state.done:
            if self._render:
                self._env.render()
            action = self._agent.act(agent, state)
            state = self._env.step(action)
            returns[self._env._env.agent_selection] += state.reward
            num_steps += 1

        return returns, num_steps

    def _done(self, frames, episodes):
        return self._frame > frames or self._episode > episodes

    def _make_writer(self, agent_name, env_name, write_loss):
        return ExperimentWriter(self, agent_name, env_name, loss=write_loss)
