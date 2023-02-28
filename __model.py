import numpy as np
import pandas as pd
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.style as mplstyle
from matplotlib.colors import LinearSegmentedColormap
import __calculations as calc
reload(calc)
import __timeseries as ts
reload(ts)
from scipy.spatial import Voronoi, voronoi_plot_2d
import __parameters as p
reload(p)

class model():
    def __init__(self):
        self.agents = []
        self.prey = []
        self.pred = []
        self.n_pred = 0
        self.n_prey = 0
        self.voronoi = False
        self.voronoi_object = 0
    
    def list_preys(self):
        self.update_agents()
        preys = pd.DataFrame()
        preys["index"] = [ag.index for ag in self.prey]
        print(preys)

    def list_preds(self):
        self.update_agents()
        preds = pd.DataFrame()
        preds["index"] = [ag.index for ag in self.pred]
        print(preds)

    def center_of_mass(self):
        r_com = np.array([0.,0.])
        for ag in self.prey:
            r_com += ag.position/self.n_prey
        return r_com
    
    def average_orientation(self):
        v_com = np.array([0.,0.])
        for ag in self.prey:
            v_com += np.array([np.cos(ag.phi),np.sin(ag.phi)])/self.n_prey
        return v_com

    def spatial_positions(self):
        Q = np.zeros(self.n_prey)
        r_com = self.center_of_mass()
        v_com = self.average_orientation()
        for i in range(self.n_prey):
            r_comi = self.prey[i].position - r_com
            Q[i] = np.dot(v_com, r_comi)/(calc.length(v_com)* calc.length(r_comi))
        return Q
    
    def position_of_back_individual(self):
        indices = np.argsort(self.spatial_positions())
        pos = self.prey[indices[0]].position
        return pos
    
    def add_agents(self, n = 1, type = "prey", from_file = False, heterogene = False,
                   r = [0,0], phi = 0, s = 8,
                   # parameters for preys (also possible from parameter file)
                   mu_att_con = 0.5, mu_rep_con = 5, mu_alg_con = 2,
                   mu_att_pred = 14, mu_rep_pred = 20, #interaction
                   a_con = -0.5, a_pred = -0.15,
                   r_att_con = 120, r_rep_con = 15, r_alg_con = 30,
                   r_att_pred = 90, r_rep_pred = 70,
                   A = 0.1, s0 = 8, 
                   interaction_con = "voronoi", interaction_pred = "all",
                   con_force = calc.preyprey_force, pred_force = calc.predprey_force,
                   #parameters for preds
                   length = 10, alpha = np.pi, beta = 0, atk_distance = 150,
                   mu_con_pred = 5, mu_prey = 0.25, #interaction
                   a_con_pred = -0.5, a_prey = -0.15,
                   r_con_pred = 100, r_prey = 1000,
                   con_pred_force = calc.predpred_force, atk_force = calc.preypred_force,
                   pos_force = calc.pos_force):
        """
        n : number of agents
        type : agent type
        from_file : if True, than takes input of __parameters.py file
        heterogene : if True for every parameter an array of n x p needs to be passed for parameters that are heterogene,
                     if False the parameters will be set the same for each agent as it is passed
        r : position
        phi : orientation in 2d space
        s : speed
        mu_X : interaction strength
        a_X : steepness of the zone transition between forces
        r_X : radius of the interaction zone
        A : stubborness (how robust agent speed stays)
        s0 : prefered speed
        interaction_X : mechanism to calculate neighbors for interaction
        X_force : function to calculate force vector (see __calculations.py)
        length : length of predator (distance between head and tail)
        alpha : attack angle (according to orienation of the school, where does the attack start;
                e.g. 0 for from front, pi for from behind, etc.)
        beta : offset angle (offset from connecting vector between pred adn center of mass -> see PDF)
        atk_distance : distance from which the predator attacks
        """
        if not from_file:
            if not heterogene:
                # from homogenous function parameters 
                if type == "prey":
                    for i in range(n):
                        ind = str(self.n_prey + i + 1)
                        self.agents.append(prey(index = ind, position = r, phi = phi, s = s,
                                            mu_con = [mu_att_con, mu_rep_con, mu_alg_con],
                                            mu_pred = [mu_rep_pred, mu_att_pred],
                                            a_con = a_con, a_pred = a_pred,
                                            r_con = [r_att_con, r_rep_con, r_alg_con], r_pred = [r_rep_pred, r_att_pred],
                                            A = A, s0 = s0,
                                            interaction_con = interaction_con, interaction_pred = interaction_pred,
                                            pred_force = pred_force, con_force = con_force))
                elif type == "pred":
                    for i in range(n):
                        ind = "p" + str(self.n_pred + i + 1)
                        self.agents.append(pred(index = ind, position = r, phi = phi, s = s, length = length,
                                                alpha = alpha, beta = beta, atk_distance = atk_distance,
                                                mu_con_pred = mu_con_pred, mu_prey = mu_prey, a_con_pred = a_con_pred,
                                                a_prey = a_prey, r_con_pred = r_con_pred, r_prey=r_prey,
                                                con_pred_force = con_pred_force, atk_force = atk_force,
                                                pos_force = pos_force))
            elif heterogene:
                # from heterogenous function parameters
                paras = {
                        "r" : r, "phi" : phi, "s" : s,
                        "mu_rep_con" : mu_rep_con, "mu_att_con" : mu_att_con, "mu_alg_con" : mu_alg_con,
                        "mu_rep_pred" : mu_rep_pred, "mu_att_pred" : mu_att_pred,
                        "a_con" : a_con, "a_pred" : a_pred, 
                        "r_rep_con" : r_rep_con, "r_att_con" : r_att_con, "r_alg_con" : r_alg_con,
                        "r_rep_pred" : r_rep_pred, "r_att_pred" : r_att_pred,
                        "A" : A, "s0" : s0,
                        "interaction_con" : interaction_con, "interaction_pred" : interaction_pred,
                        "con_force" : con_force, "pred_force" : pred_force,
                        "length" : length, "alpha" : alpha, "beta" : beta, "atk_distance" : atk_distance,
                        "mu_con_pred" : mu_con_pred, "mu_prey" : mu_prey, "a_con_pred" : a_con_pred,
                        "a_prey" : a_prey, "r_con_pred" : r_con_pred, "r_prey" : r_prey,
                        "con_pred_force" : con_pred_force, "atk_force" : atk_force, "pos_force" : pos_force
                    }
                # convert every single value to a n x list, that is not already a list of values
                for key, value in paras.items():
                    try:
                        L = len(paras[key])
                    except:
                        L = 0
                    if (L != n):
                        paras[key] = [value]*n

                if type == "prey":
                    for i in range(n):
                        ind = str(self.n_prey + i + 1)
                        self.agents.append(prey(index = ind, position = paras["r"][i], phi = paras["phi"][i], s = paras["s"][i],
                                            mu_con = [paras["mu_att_con"][i], paras["mu_rep_con"][i], paras["mu_alg_con"][i]],
                                            mu_pred = [paras["mu_rep_pred"][i], paras["mu_att_pred"][i]],
                                            a_con = paras["a_con"][i], a_pred = paras["a_pred"][i],
                                            r_con = [paras["r_att_con"][i], paras["r_rep_con"][i], paras["r_alg_con"][i]],
                                            r_pred = [paras["r_rep_pred"][i], paras["r_att_pred"][i]],
                                            A = paras["A"][i], s0 = paras["s0"][i],
                                            interaction_con = paras["interaction_con"][i], interaction_pred = paras["interaction_pred"][i],
                                            pred_force = paras["pred_force"][i], con_force = paras["con_force"][i]))
                elif type == "pred":
                    for i in range(n):
                        ind = "p" + str(self.n_pred + i + 1)
                        self.agents.append(pred(index = ind, position = paras["r"][i], phi = paras["phi"][i], s = paras["s"][i],
                                                length = paras["length"][i], alpha = paras["alpha"][i], beta = paras["beta"][i],
                                                atk_distance = paras["atk_distance"][i], mu_con_pred = paras["mu_con_pred"][i],
                                                mu_prey = paras["mu_prey"][i], a_con_pred = paras["a_con_pred"][i],
                                                a_prey = paras["a_prey"][i], r_con_pred = paras["r_con_pred"][i],
                                                r_prey=paras["r_prey"][i], con_pred_force = paras["con_pred_force"][i],
                                                atk_force = paras["atk_force"][i], pos_force = paras["pos_force"][i]))
        elif from_file:
            if not heterogene:
                # from homogene file parameters
                if type == "prey":
                    for i in range(p.n):
                        ind = str(self.n_prey + i + 1)
                        self.agents.append(prey(index = ind, position = r, phi = phi, s = s,
                                                mu_con = [p.mu_att_con, p.mu_rep_con, p.mu_alg_con],
                                                mu_pred = [p.mu_rep_pred, p.mu_att_pred], a_con = p.a_con,
                                                a_pred = p.a_pred, r_con = [p.r_att_con, p.r_rep_con, p.r_alg_con],
                                                r_pred = [p.r_rep_pred, p.r_att_pred], A = p.A, s0 = p.s0,
                                                interaction_con = p.interaction_con, interaction_pred = p.interaction_pred,
                                                pred_force = p.pred_force, con_force = p.con_force))
                elif type == "pred":
                    for i in range(p.n):
                        ind = "p" + str(self.n_pred + 1)
                        self.agents.append(pred(index = ind, position = r, phi = phi, s = s, length = p.length,
                                                alpha = p.alpha, beta = p.beta, atk_distance = p.atk_distance,
                                                mu_con_pred = p.mu_con_pred, mu_prey = p.mu_prey, a_con_pred = p.a_con_pred,
                                                a_prey = p.a_prey, r_con_pred = p.r_con_pred, r_prey = p.r_prey,
                                                con_pred_force = p.con_pred_force, atk_force = p.atk_force,
                                                pos_force = p.pos_force))
            elif heterogene:
               # from heterogenous function parameters
                paras = {
                        "r" : p.r, "phi" : p.phi, "s" : p.s,
                        "mu_rep_con" : p.mu_rep_con, "mu_att_con" : p.mu_att_con, "mu_alg_con" : p.mu_alg_con,
                        "mu_rep_pred" : p.mu_rep_pred, "mu_att_pred" : p.mu_att_pred,
                        "a_con" : p.a_con, "a_pred" : p.a_pred, 
                        "r_rep_con" : p.r_rep_con, "r_att_con" : p.r_att_con, "r_alg_con" : p.r_alg_con,
                        "r_rep_pred" : p.r_rep_pred, "r_att_pred" : p.r_att_pred,
                        "A" : p.A, "s0" : p.s0,
                        "interaction_con" : p.interaction_con, "interaction_pred" : p.interaction_pred,
                        "con_force" : p.con_force, "pred_force" : p.pred_force,
                        "length" : p.length, "alpha" : p.alpha, "beta" : p.beta, "atk_distance" : p.atk_distance,
                        "mu_con_pred" : p.mu_con_pred, "mu_prey" : p.mu_prey, "a_con_pred" : p.a_con_pred,
                        "a_prey" : p.a_prey, "r_con_pred" : p.r_con_pred, "r_prey" : p.r_prey,
                        "con_pred_force" : p.con_pred_force, "atk_force" : p.atk_force, "pos_force" : p.pos_force
                    }
                # convert every single value to a n x list, that is not already a list of values
                for key, value in paras.items():
                    try:
                        L = len(paras[key])
                    except:
                        L = 0
                    if (L != n):
                        paras[key] = [value]*n

                if type == "prey":
                    for i in range(p.n):
                        ind = str(self.n_prey + i + 1)
                        self.agents.append(prey(index = ind, position = paras["r"][i], phi = paras["phi"][i], s = paras["s"][i],
                                            mu_con = [paras["mu_att_con"][i], paras["mu_rep_con"][i], paras["mu_alg_con"][i]],
                                            mu_pred = [paras["mu_rep_pred"][i], paras["mu_att_pred"][i]],
                                            a_con = paras["a_con"][i], a_pred = paras["a_pred"][i],
                                            r_con = [paras["r_att_con"][i], paras["r_rep_con"][i], paras["r_alg_con"][i]],
                                            r_pred = [paras["r_rep_pred"][i], paras["r_att_pred"][i]],
                                            A = paras["A"][i], s0 = paras["s0"][i],
                                            interaction_con = paras["interaction_con"][i], interaction_pred = paras["interaction_pred"][i],
                                            pred_force = paras["pred_force"][i], con_force = paras["con_force"][i]))
                elif type == "pred":
                    for i in range(p.n):
                        ind = "p" + str(self.n_pred + i + 1)
                        self.agents.append(pred(index = ind, position = paras["r"][i], phi = paras["phi"][i], s = paras["s"][i],
                                                length = paras["length"][i], alpha = paras["alpha"][i], beta = paras["beta"][i],
                                                atk_distance = paras["atk_distance"][i], mu_con_pred = paras["mu_con_pred"][i],
                                                mu_prey = paras["mu_prey"][i], a_con_pred = paras["a_con_pred"][i],
                                                a_prey = paras["a_prey"][i], r_con_pred = paras["r_con_pred"][i],
                                                r_prey=paras["r_prey"][i], con_pred_force = paras["con_pred_force"][i],
                                                atk_force = paras["atk_force"][i], pos_force = paras["pos_force"][i]))

        self.update_agents()
        if (interaction_con == "voronoi") and type == "prey":
            self.voronoi = True
    
    def update_agents(self):
        """
        updates lists of different agents and numbers
        """
        self.prey = [ag for ag in self.agents if ag.type == "prey"]
        self.pred = [ag for ag in self.agents if ag.type == "pred"]
        self.n_prey = len(self.prey)
        self.n_pred = len(self.pred)
        self.prey_positions = [prey.position for prey in self.prey]
        self.phis = [prey.phi for prey in self.prey]
        self.pred_positions = [pred.position for pred in self.pred]
        self.tail_positions = [pred.tail_position for pred in self.pred]
        
    def move_agents(self, t_step):
        """

        """
        attack_time = 0

        prey_positions = np.array(self.prey_positions)
        prey_phis = np.array(self.phis)
        pred_positions = np.array(self.pred_positions)
        tail_positions = np.array(self.tail_positions)
        if self.voronoi:
            self.voronoi_object = calc.voronoi(prey_positions)
            voronoi_regions = calc.voronoi_regions(self.voronoi_object)
        else:
            voronoi_regions = 0
            self.voronoi_object = 0
        for ag in self.agents:
            ag.move(prey_positions, prey_phis, pred_positions, tail_positions, t_step, voronoi_regions)

        #phase transitions of agents
        for ag in self.agents:
            if ag.type == "pred":
                if ag.phase == "prepare":
                    alpha = ag.alpha[ag.atk_index]
                    ag.start_point = calc.pred_startpoint(self.prey_positions, self.phis, alpha, ag.atk_distance)
                    d_point = calc.length(ag.position - ag.start_point)
                    attacks = np.array([ag.phase == "attack" for ag in self.agents if ag.type == "pred"])
                    if d_point < p.prepare_point_radius and not attacks.any():
                        ag.phase = "attack"
                elif ag.phase == "attack":
                    attack_time += 1
                    d_com = calc.length(calc.centerofmass(self.prey_positions) - ag.position)
                    if (d_com < p.atk_point_radius) or (attack_time > p.max_atk_timesteps):
                        ag.phase = "prepare"
                        attack_time = 0
                        if ag.atk_index < (len(ag.alpha) - 1):
                            ag.atk_index += 1
        self.update_agents()

    def simulate(self, time, t_step = 0.01):
        timespan = int(time/t_step)
        for _ in range(timespan):
            self.move_agents(t_step)

    def create_timeseries(self, time, t_step = 0.01, sub = 1):
        """
        simulate all timesteps over time
        creates timeseries object
        """
        timespan = int(time/t_step)
        length = int(np.ceil(timespan/sub))
        preys = np.zeros((self.n_prey,length,4)) #x,y,phi,v
        preds = np.zeros((self.n_pred,length,6)) #x_head, y_head, x_tail, y_tail, phi, v
        for t in range(length):
            #track all prey
            count1 = 0
            for prey in self.prey:
                preys[count1,t] = np.array([prey.position[0], prey.position[1], prey.phi, prey.s])
                count1 += 1
            #track all pred
            count2 = 0
            for pred in self.pred:
                preds[count2,t] = np.array([pred.position[0], pred.position[1], pred.tail_position[0], pred.tail_position[1], pred.phi, pred.s])
                count2 += 1
            for _ in range(sub):
                self.move_agents(t_step)
        return ts.timeseries(preys, preds, t_step)

    def live_simulation(self, time, t_step = 0.01, sub = 15, save = True, path = "animation.mp4"):
        """
        creates a live animation of a simulation run
        """
        #plt.style.use('fivethirtyeight')
        self.move_agents(t_step)
        fig = plt.figure()
        ax = plt.axes()
        mplstyle.use('fast')

        #trace configuration
        trace = p.trace
        last_x = np.zeros((self.n_prey, trace))
        last_y = np.zeros((self.n_prey, trace))
        last_x_pred = np.zeros((self.n_pred, trace))
        last_y_pred = np.zeros((self.n_pred, trace))
        alphas  = np.flip(np.linspace(0.01,1,trace))
        cmap_nointeraction = "binary"
        cmap_interaction = LinearSegmentedColormap.from_list("cmap_interaction", ["white", "blue"])
        cmap_pred_attack = LinearSegmentedColormap.from_list("cmap_interaction", ["white", "green"])
        cmap_pred_prepare = LinearSegmentedColormap.from_list("cmap_interaction", ["white", "red"])
        def animate(i):
            plt.cla()
            #define boundaries
            positions = np.concatenate((self.prey_positions, self.pred_positions, self.tail_positions))
            bounds = calc.plot_boundaries(positions)

            #plot agents and voronoi
            for index, prey in enumerate(self.prey):
                if prey.pred_interaction == True:
                    ax.quiver(prey.position[0], prey.position[1], np.cos(prey.phi), np.sin(prey.phi), color = "blue")
                    # trace
                    if i >= trace:
                        ax.scatter(last_x[index], last_y[index], s = 1, c = alphas, cmap = cmap_interaction, marker = ".")
                        last_x[index] = np.roll(last_x[index], 1)
                        last_y[index] = np.roll(last_y[index], 1)
                        last_x[index][0] = prey.position[0]
                        last_y[index][0] = prey.position[1]
                    else:
                        last_x[index][-(i+1)] = prey.position[0]
                        last_y[index][-(i+1)] = prey.position[1]
                else:
                    ax.quiver(prey.position[0], prey.position[1], np.cos(prey.phi), np.sin(prey.phi))
                    # trace
                    if i >= trace:
                        ax.scatter(last_x[index], last_y[index], s = 1, c = alphas, cmap = cmap_nointeraction, marker = ".")
                        last_x[index] = np.roll(last_x[index], 1)
                        last_y[index] = np.roll(last_y[index], 1)
                        last_x[index][0] = prey.position[0]
                        last_y[index][0] = prey.position[1]
                    else:
                        last_x[index][-(i+1)] = prey.position[0]
                        last_y[index][-(i+1)] = prey.position[1]
            if self.voronoi and (self.voronoi_object != 0):
                voronoi_plot_2d(self.voronoi_object, ax = ax, show_points = False, show_vertices = False)
            for index, pred in enumerate(self.pred):
                if pred.phase == "prepare":
                    ax.plot([pred.tail_position[0], pred.position[0]], [pred.tail_position[1], pred.position[1]], "r")
                    ax.quiver(pred.position[0], pred.position[1], np.cos(pred.phi), np.sin(pred.phi), color = "red")
                    if i >= trace:
                        ax.scatter(last_x_pred[index], last_y_pred[index], s = 1, c = alphas, cmap = cmap_pred_prepare, marker = ".")
                        last_x_pred[index] = np.roll(last_x_pred[index], 1)
                        last_y_pred[index] = np.roll(last_y_pred[index], 1)
                        last_x_pred[index][0] = pred.tail_position[0]
                        last_y_pred[index][0] = pred.tail_position[1]
                    else:
                        last_x_pred[index][-(i+1)] = pred.tail_position[0]
                        last_y_pred[index][-(i+1)] = pred.tail_position[1]
                elif pred.phase == "attack":
                    ax.plot([pred.tail_position[0], pred.position[0]], [pred.tail_position[1], pred.position[1]], "g")
                    ax.quiver(pred.position[0], pred.position[1], np.cos(pred.phi), np.sin(pred.phi), color = "green")
                    if i >= trace:
                        ax.scatter(last_x_pred[index], last_y_pred[index], s = 1, c = alphas, cmap = cmap_pred_attack, marker = ".")
                        last_x_pred[index] = np.roll(last_x_pred[index], 1)
                        last_y_pred[index] = np.roll(last_y_pred[index], 1)
                        last_x_pred[index][0] = pred.tail_position[0]
                        last_y_pred[index][0] = pred.tail_position[1]
                    else:
                        last_x_pred[index][-(i+1)] = pred.tail_position[0]
                        last_y_pred[index][-(i+1)] = pred.tail_position[1]
            #customize axes
            ax.set_xlim(bounds[0]-15,bounds[1]+15)
            ax.set_ylim(bounds[2]-15,bounds[3]+15)
            ax.set_aspect("equal")
            ax.axis('off')
            for _ in range(sub):
                self.move_agents(t_step)
        ani = animation.FuncAnimation(fig, animate, interval = 10, frames = int(time/(sub*t_step)))
        if save == True:
            frames = int(np.around((1/t_step)/sub))
            ani.save(path, fps = frames)
        plt.show()

class agent():
    """
    agent in the model
    """
    def __init__(self, index, position, phi, s):
        self.index = index
        self.position = position
        self.phi = phi
        self.s = s


class prey(agent):
    """
    prey agent
    """
    def __init__(self, index, position, phi, s, A = 0.1, s0 = 8, sigma = 0.02,
                 con_force = calc.preyprey_force, mu_con = [0.2,5,1.5], a_con = [-0.15,-0.15,-0.15], r_con = [160,20,40],
                 interaction_con = "nnn", n_nearest_con = 5, ran_con = 40,
                 pred_force = calc.predprey_force, mu_pred = [10,7], a_pred = [-0.15,-0.15], r_pred = [50,70],
                 interaction_pred = "all", n_nearest_pred = 3, ran_pred = 70,
                 near_predator = "no_interaction"):
        super().__init__(index, position, phi, s)
        self.type = "prey"

        #inidvidual parameters
        self.A = A #stubborness
        self.s0 = s0 #prefered speed
        self.sigma = sigma

        #conspecific interaction parameters
        self.near_predator = near_predator
        self.con_force = con_force
        self.mu_con = np.array(mu_con)
        self.a_con = np.full(3, a_con)
        self.r_con = np.array(r_con)
        if interaction_con == "all":
            self.int_function_con = calc.all_interactions
        elif interaction_con == "nnn":
            self.int_function_con = calc.nnn_interactions
            self.n_nearest_con = n_nearest_con
        elif interaction_con == "range":
            self.int_function_con = calc.range_interactions
            self.ran_con = ran_con
        elif interaction_con == "voronoi":
            self.int_function_con = calc.voronoi_interactions

        #predator interaction parameters
        self.pred_force = pred_force
        self.mu_pred = np.array(mu_pred)
        self.a_pred = np.full(3, a_pred)
        self.r_pred =  np.array(r_pred)
        if interaction_pred == "all":
            self.int_function_pred = calc.all_interactions_pred
        elif interaction_pred == "nnn":
            self.int_function_pred = calc.nnn_interactions
            self.n_nearest_pred = n_nearest_pred
        elif interaction_pred == "range":
            self.int_function_pred = calc.range_interactions_pred
            self.ran_pred = ran_pred
        elif interaction_pred == "voronoi":
            self.int_function_pred = calc.voronoi_interactions
        self.pred_interaction = False 

    def move(self, prey_positions, prey_phis, pred_positions, tail_positions, t_step, voronoi_regions = 0):
        """
        move agent one timestep

        different phases
        """
        #con forces
        if self.int_function_con == calc.voronoi_interactions:
            con_interactions = self.int_function_con(self.position, prey_positions, voronoi_regions)
        else:
            con_interactions = self.int_function_con(self.position, prey_positions)
        positions_con = np.array([prey_positions[i] for i in con_interactions])
        phis_con = np.array([prey_phis[i] for i in con_interactions])
        

        if self.near_predator == "no interaction":
            if self.pred_interaction: 
                con_force = self.con_force(self.position, self.phi, positions_con, phis_con, mu_con = [0,0,0], a_con = self.a_con, r_con = self.r_con, voronoi = (self.int_function_con == calc.voronoi_interactions))
            else:
                con_force = self.con_force(self.position, self.phi, positions_con, phis_con, mu_con = self.mu_con.astype("float64"), a_con = self.a_con, r_con = self.r_con, voronoi = (self.int_function_con == calc.voronoi_interactions))
        else:
            con_force = self.con_force(self.position, self.phi, positions_con, phis_con, mu_con = self.mu_con.astype("float64"), a_con = self.a_con, r_con = self.r_con, voronoi = (self.int_function_con == calc.voronoi_interactions))

        #pred forces
        pred_interactions = self.int_function_pred(self.position, pred_positions, tail_positions)
        positions_pred = np.array([pred_positions[i] for i in pred_interactions])
        pred_force = self.pred_force(self.position, positions_pred, tail_positions, self.mu_pred.astype("float64"), self.a_pred, self.r_pred)
        if (pred_force != 0.).any():
            self.pred_interaction = True
        else:
            self.pred_interaction = False
        new_r, new_phi, new_s = calc.move_prey(force = pred_force + con_force, t_step = t_step,
                                               position = self.position, phi = self.phi, s = self.s, A = self.A, s0 = self.s0, sigma = self.sigma)
        self.phi = new_phi
        self.s = new_s
        self.position = new_r


class pred(agent):
    """
    predator agent
    """
    def __init__(self, index, position, phi, s, length, sigma = 0.02, 
                 alpha = np.pi, beta = 0, atk_distance = 150, 
                 mu_con_pred = 0, mu_prey = 0.35, a_con_pred = -0.5,
                 a_prey = -0.15, r_con_pred = 20, r_prey = 1000,
                 con_pred_force = calc.predpred_force,
                 atk_force = calc.preypred_force,
                 pos_force = calc.pos_force):
        super().__init__(index, position, phi, s)

        #individual parameters
        self.type = "pred"
        self.length = length
        self.tail_position = np.array([self.position[0] - self.length*np.cos(self.phi),
                                       self.position[1] - self.length*np.sin(self.phi)])
        self.sigma = sigma
        self.phase = "prepare"
    
        #conspecific interactions
        self.con_function = con_pred_force
        self.r_con_pred = r_con_pred
        self.a_con_pred = a_con_pred
        self.mu_con_pred = mu_con_pred

        #prey interactions
        self.atk_function = atk_force
        self.pos_function = pos_force
        self.mu_prey = mu_prey
        self.a_prey = a_prey
        self.r_prey = r_prey
        if isinstance(alpha, (float, int)):
            self.alpha = [alpha] # attack angle
        else:
            self.alpha = alpha
        self.beta = beta # offset angle
        self.atk_distance = atk_distance # minimum attack distance to school
        self.start_point = np.array([0,0]) #next start point of attack
        self.atk_index = 0

    def move(self, prey_positions, prey_phis, pred_positions, tail_positions, t_step, voronoi_regions = 0):
        if self.phase == "attack":
            atk_force = self.atk_function(self.position, prey_positions, mu_prey = self.mu_prey, angle = self.beta) # force for attacking prey
            con_force =  self.con_function(self.position, pred_positions, mu_con_pred = 0, r_con_pred = self.r_con_pred) # force between predators
            total_force = atk_force + con_force
        elif self.phase == "prepare":
            pos_force = self.pos_function(self.position, self.start_point, self.mu_prey, prey_positions) #force to get into position (attraction to position, alignment to attack angle, repulsion from school to certain distance)
            con_force = self.con_function(self.position, pred_positions, mu_con_pred = self.mu_con_pred, r_con_pred = self.r_con_pred) # force between predators
            total_force = pos_force + con_force

        new_r, new_phi, new_s = calc.move_pred(force = total_force, t_step = t_step,
                                               position = self.position, phi = self.phi, s = self.s, sigma = self.sigma)
        self.phi = new_phi
        self.s = new_s
        self.position = new_r
        self.tail_position = calc.new_tail(new_r, new_phi, self.length)