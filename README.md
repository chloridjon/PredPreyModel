# Predator Prey Model
##  Model to simulate an interaction between predator(s) and a group of prey agents.
This model only works on python <=3.10, because some calculations are done using numba. A detailed explanations of the model theory and the current results can be found in the [Documentation](PPModel.pdf) 
## Important files
__model.py - Core Code of the Agent-Based Model; contains model class and the agent classes <br>
__calculations.py - Various Calculations for the Model and Timerseries, optimized with numba <br>
__parameters.py - File to set the parameters for the model agents (also possible to set parameters without file) <br>
__timeseries.py - Timeseries object on which various calculations can be performed <br>
<br>
live_simualtion.py - live animation of the swarm <br>
multiple_simualtions.py - iterate over many simulations with different parameters (e.g. interaction strength) to extract a certain measure (e.g polarization)<br>
trajectory.py - Plot the trajectories of some agents, e.g. to see turnradii<br>
video_simulation.py - Creates a video from the timeseries object<br>
genetic_algorithm.py - starts a genetic algorithm optimization with pyGAD
