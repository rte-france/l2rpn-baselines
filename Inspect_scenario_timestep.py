from grid2op.Parameters import Parameters
from grid2op.Runner import Runner
from grid2op import make
from grid2op.PlotGrid import PlotMatplot
import grid2op

# Method fast_forward_chronics doesnt work properly
path_dataset='rte_case14_realistic'#None
env=make(path_dataset,param=param,backend=backend,test=True)

param = Parameters()
param.init_from_dict({"NO_OVERFLOW_DISCONNECTION": True})

scenario_name='000'#None
timestep=0#None

#catch the id of your scenario
for id, sp in enumerate(env.chronics_handler.real_data.subpaths):
    env.set_id(id)
    env.reset()
    current_name = os.path.basename(env.chronics_handler.get_id())
    assert current_name == os.path.basename(sp)

#jump to your timestep
if timestep >0:
    env.fast_forward_chronics(nb_timestep= timestep)
    obs = env.get_obs()

#do-nothiong action to simulate and get an observation
action_def={}#do-nothing
action=env.action_space(action_def)
new_obs,_reward,_done,_info=obs.step(action)

#plot observation
plot_helper = PlotMatplot(env.observation_space)
fig_obs = plot_helper.plot_obs(new_obs)
fig_obs.show()