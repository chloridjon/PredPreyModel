import numpy as np
import __calculations as calc
import __model as m
import __timeseries as ts
from importlib import reload
reload(m)
reload(calc)

#initialize model
M = m.model()

#add prey
M.add_agents(n = 10, from_file = True, heterogene = True)
M.simulate(10)

#predator parameters
n_pred = 1
length = 10
s_pred = 16

#add predators
M.add_agents(r = np.array([300.,300.]), phi = 0, type = "pred", length = length, mu_con_pred = 5, r_con_pred = 100,
             alpha = np.random.uniform(0,np.pi,10), mu_prey = 0.5, s = s_pred, atk_distance = 200, beta = 0)

M.add_agents(r = np.array([-300.,-300.]), phi = 0, type = "pred", length = length, mu_con_pred = 5, r_con_pred = 100,
             alpha = np.random.uniform(0,np.pi,10), mu_prey = 0.5, s = s_pred, atk_distance = 200, beta = 0)

M.add_agents(r = np.array([300.,-300.]), phi = 0, type = "pred", length = length, mu_con_pred = 5, r_con_pred = 100,
             alpha = np.random.uniform(0,np.pi,10), mu_prey = 0.5, s = s_pred, atk_distance = 200, beta = 0)

M.live_simulation(200, sub = 10, save = True, path = "3predators.mp4")
