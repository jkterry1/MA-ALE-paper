from all.agents.multi.independent import IndependentMultiagent
from ..builder import preset_builder
from ..preset import Preset
from torch.nn import ModuleDict

class Independent(Preset):
    def __init__(self, base_presets):
        super().__init__()
        self.agents = list(base_presets.keys())
        self.agents = ModuleDict({
            agent : base_presets[agent]
                for agent in self.agents})

    def agent(self, writer, train_steps):
        return IndependentMultiagent({
            agent : self.agents[agent].agent(writer[agent], train_steps)
            for agent in self.agents
        })

    def test_agent(self):
        return IndependentMultiagent({
            agent : self.agents[agent].test_agent()
            for agent in self.agents
        })

class IndependentBuilder:
    def __init__(self, builders, env=None):
        self.builders = builders
        self._env = env
        self._name = "independent_"+"_".join(list(sorted(builder._name for builder in self.builders.values())))

    def name(self, name):
        return IndependentBuilder({agent:builder.name(name) for agent, builder in self.builders.items()}, env=self._env)

    def hyperparameters(self, **hyperparameters):
        return IndependentBuilder({agent:builder.hyperparameters(**hyperparameters) for agent, builder in self.builders.items()}, env=self._env)

    def env(self, env):
        return IndependentBuilder(self.builders, env=env)

    def device(self, device):
        return IndependentBuilder({agent:builder.device(device) for agent, builder in self.builders.items()}, env=self._env)

    def build(self):
        env = self._env
        agents = {
            agent : self.builders[agent]
                        .env(env.subenvs[agent])
                        .build()
                for agent in env.agents}
        return Independent(agents)

independent = IndependentBuilder
__all__ = ["independent"]
