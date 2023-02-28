import __calculations as calc
import __model as m
import __timeseries as ts
import numpy as np
import matplotlib.pyplot as plt
import time
import __parameters as p

#parameters for model -> others in parameter file
n_prey = 10
n_pred = 1

#variables for heatmap
size_heatmap = 5
x_array = np.linspace(0,50,size_heatmap) #length
y_array = np.linspace(0,np.pi/3,size_heatmap) #attack angle
mup_array = np.flip(np.linspace(0.2,1.2,size_heatmap)) #prey turn radius
colormesh = np.zeros((size_heatmap, size_heatmap))

#number of Runs for average
N = 50

N_run = 0
for _ in range(N):
    #time calcs
    N_run += 1
    print(f"Starting Run {N_run}")
    start = time.time()

    #modeling
    for x_count, x in enumerate(x_array): #length
        for y_count, y in enumerate(y_array): # attack_angle
            #set up of model
            M = m.model()

            #add prey
            p.n = n_prey
            M.add_agents(from_file = True, heterogene = True)
            M.simulate(10)
            
            #add pred
            p.n = n_pred
            p.length = x[x_count] ########## set heatmap parameters here
            p.beta = y[y_count] ########## set heatmap parameters here
            M.add_agents(type = "pred", from_file = True, heterogene = True)

            #timeseries operations
            ts = M.create_timeseries(20, sub = 10)
            colormesh[x_count, y_count] += ts.average_polarization()/N
    
    #time calcs
    end = time.time()
    print(f"Finishing Run {N_run} at {np.around(end-start)} seconds")
    T = (N-N_run)*(end-start)/3600
    hours = np.floor(T)
    minutes = np.floor((T % 1) * 60)
    seconds = np.around((((T % 1) * 60) % 1) * 60)
    print(f"Estimated time until completion: {hours} hours, {minutes} minutes and {seconds} seconds")

#plotting
plt.pcolormesh(y_array, x_array, colormesh)
plt.ylabel("length (Predator)")
plt.xlabel("attack angle (Predator) [Â°]")
plt.xticks(y_array)
plt.yticks(x_array)
plt.colorbar(label = "Polarization")
plt.savefig(f"Metric_Heatmaps/polarization_N{n_prey}x{N}_metric.pdf")