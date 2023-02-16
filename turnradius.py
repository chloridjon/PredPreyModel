import numpy as np
import __calculations as calc
import __model as m
from importlib import reload
reload(m)
reload(calc)

#initialize model
M = m.model()

#add prey
M.add_agents(n = 1, from_file = True, heterogene = True)
M.simulate(10)

#predator parameters
n_pred = 1
length = 10
s_pred = 16

#add predators
M.add_agents(r = np.array([100.,100.]), phi = 0, type = "pred", length = length,
             alpha = np.random.uniform(0,np.pi,10), mu_prey = 0.8, s = s_pred, atk_distance = 200, beta = 0)

ts = M.create_timeseries(80)
ts.plot_trajectories(savefig = True)
ts.fast_animation(path = "turnradius.mp4")