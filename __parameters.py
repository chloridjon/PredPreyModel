import numpy as np
import __calculations as calc

# agent parameters

n = 10
r = calc.random_positions(n, d = 20)
phi = np.random.normal(0, np.pi/4, n)
s = np.random.normal(8, 1, n)


# prey parameters

mu_rep_con = 5
mu_att_con = 0.5
mu_alg_con = 2

mu_rep_pred = 10
mu_att_pred = 7

a_con = -0.5
a_pred = -0.15

r_rep_con = 15
r_att_con = 60
r_alg_con = 30

r_rep_pred = 70
r_att_pred = 90

A = 0.1
s0 = 8

interaction_con = "voronoi"
interaction_pred = "all"

con_force = calc.preyprey_force
pred_force = calc.predprey_force

near_predator = "no interaction"


# predator parameters

length = 10
alpha = np.pi
beta = 6.92265909
atk_distance = 150

mu_con_pred = 0
mu_prey = 0.8

a_con_pred = -0.5
a_prey = -0.15

r_con_pred = 20
r_prey = 1000

con_pred_force = calc.predpred_force
atk_force = calc.preypred_force
pos_force = calc.pos_force

max_atk_timesteps = 100
atk_point_radius = 10
prepare_point_radius = 10

# global model parameters

sigma = 0.02


# other parameters

theta = np.pi/3 # for fleeangle only model
trace = 7 # for plotting the trace, how many points as trace

# functions to create parameter dictionaries

