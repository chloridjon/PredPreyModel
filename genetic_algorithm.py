import numpy as np
import __calculations as calc
import __model as m
import __timeseries as ts
from importlib import reload
reload(m)
reload(calc)
import pygad

def model_function(mu, beta, atk_distance):
    M = m.model()
    M.add_agents(n = 10, from_file = True, heterogene = True) # addprey
    M.simulate(10)
    # pred parameters
    r_com = M.center_of_mass()
    r = 150
    phi = 0
    r_pred = r_com + np.array([r*np.cos(phi) , r*np.sin(phi)])
    phi_pred = 0
    s_pred = 16
    length = 10

    M.add_agents(n = 1, type = "pred", r = r_pred, phi = phi_pred, s = s_pred, length = length,
                 atk_distance = atk_distance, mu_prey = mu, alpha = np.random.uniform(0,2*np.pi), beta = beta)
    M.add_agents(n = 1, type = "pred", r = r_pred, phi = phi_pred, s = s_pred, length = length,
                 atk_distance = atk_distance, mu_prey = mu, alpha = np.random.uniform(0,2*np.pi), beta = beta)
    M.add_agents(n = 1, type = "pred", r = r_pred, phi = phi_pred, s = s_pred, length = length,
                 atk_distance = atk_distance, mu_prey = mu, alpha = np.random.uniform(0,2*np.pi), beta = beta)
    ts = M.create_timeseries(100, sub = 10)
    d_avg = ts.average_deviation()
    d_compact = ts.average_spatial_compactness()

    return d_avg + d_compact


desired_output = 0

def fitness_func(solution, solution_idx):
    output = model_function(solution[0], solution[1], solution[2])
    fitness = 1/np.abs(output - desired_output)
    return fitness

num_generations = 20
num_parents_mating = 4

fitness_function = fitness_func

sol_per_pop = 8
num_genes = 3

gene_space = [{"low" : 0, "high" : 10}, {"low" : -np.pi, "high" : np.pi}, {"low" : 0, "high" : 300}] # mu, beta, atk_distance

parent_selection_type = "sss"
keep_parents = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 50

ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       parent_selection_type=parent_selection_type,
                       keep_parents=keep_parents,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       gene_space=gene_space,
                       parallel_processing = 12)

ga_instance.run()
ga_instance.plot_fitness(save_dir = "Graphs/Fitness_pyGAD_withcompactness.png")

solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Parameters of the best solution : {solution}".format(solution=solution))

filename = 'Data/genetic_mu_beta_atkdistance_N3'
ga_instance.save(filename=filename)