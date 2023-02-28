import numpy as np
import __calculations as calc
import __model as m
import __timeseries as ts
from importlib import reload
reload(m)
reload(calc)
import __parameters as p

#initialize model
M = m.model()

#add prey
M.add_agents(n = 50, from_file = True, heterogene = True)
M.simulate(10)

#predator parameters
r_com = M.center_of_mass()
r = 150
phi = 0
r_pred = r_com + np.array([r*np.cos(phi) , r*np.sin(phi)])
phi_pred = 0
s_pred = 16
length = 10
atk_distance = 300
mu = 4.98531305
alpha = 0
beta = 1.41661166

M.add_agents(n=1, type = "pred", r = r_pred, phi = phi_pred, s = s_pred, length = length,
             atk_distance = atk_distance, mu_prey = mu, alpha = np.random.uniform(0,2*np.pi, 10), beta = beta)

M.live_simulation(100, sub = 10, save = False, path = f"Animations/animation_fleeangle_only_test.mp4")
