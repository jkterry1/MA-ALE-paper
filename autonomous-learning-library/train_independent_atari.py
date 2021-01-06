import argparse
from all.environments import MultiAgentAtariEnv
from all.experiments.multiagent_env_experiment import MultiagentEnvExperiment
from all.presets import atari
from all.presets.multiagent_atari import independent
from timeit import default_timer as timer


def main():
    parser = argparse.ArgumentParser(description="Run an multiagent Atari benchmark.")
    parser.add_argument("env", help="Name of the Atari game (e.g. Pong).")
    parser.add_argument(
        "agent", help="Name of the agent (e.g. dqn). See presets for available agents."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="The name of the device to run the agent on (e.g. cpu, cuda, cuda:0).",
    )
    parser.add_argument(
        "--frames", type=int, default=40e6, help="The number of training frames."
    )
    parser.add_argument(
        "--render", type=bool, default=False, help="Render the environment."
    )
    args = parser.parse_args()

    # self._writer = ExperimentWriter(self, multiagent.__name__, env.name, loss=write_loss)
    # self._writers = {
    #     agent : ExperimentWriter(self, "{}_{}".format(multiagent.__name__, agent), env.name, loss=write_loss)
    #     for agent in env.agents
    # }
    env = MultiAgentAtariEnv(args.env, device=args.device)
    agent_name = args.agent
    base_builder = getattr(atari, agent_name)()
    preset = independent({agent:base_builder for agent in env.agents}).env(env).hyperparameters(replay_buffer_size=350000,replay_start_size=100)
    # agent = preset.agent()
    print(preset._name)

    experiment = MultiagentEnvExperiment(preset.build(), env, write_loss=False, name="independent_"+args.env+"_"+args.agent, steps_per_save=200000)
    experiment.train(frames=20e6)

if __name__ == "__main__":
    main()
