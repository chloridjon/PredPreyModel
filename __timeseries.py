#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 12:26:55 2022

@author: root
"""

from numba import jit
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import normal
import pyinform as pin
import pandas as pd
from matplotlib.offsetbox import AnchoredText
from scipy.optimize import curve_fit
import __entropy_estimators3 as ee
from importlib import reload
import matplotlib.style as mplstyle
import __calculations as calc
from scipy.spatial import KDTree



class timeseries():
    def __init__(self, preys, preds, t_step):
        self.preys = preys
        self.preds = preds
        self.t_step = t_step

        self.n_prey = len(preys)
        self.n_pred = len(preds)
        self.length = len(preys[0])

        self.x_prey = preys[:,:,0]
        self.y_prey = preys[:,:,1]
        self.phi_prey = preys[:,:,2] #polar orientation angle
        self.s_prey = preys[:,:,3]

        self.x_head = preds[:,:,0]
        self.y_head = preds[:,:,1]
        self.x_tail = preds[:,:,2]
        self.y_tail = preds[:,:,3]
        self.phi_pred = preds[:,:,4] #polar orientation angle
        self.s_pred = preds[:,:,5]

    def fleeangles(self, pred_index = 0):
        fleeangles = np.zeros((self.n_prey, self.length))

        r = self.preys[:,:,0:2]
        phi = self.phi_prey

        r_pred = self.preds[pred_index,:,0:2]
        phi_pred = self.phi_pred[pred_index]

        for i in range(self.n_prey):
            R = r[i,:] - r_pred[i]
            angles = np.arctan2(R[:,1], R[:,0])
            fleeangles[i] = np.pi - (phi[i] - phi_pred) + angles
            for j in range(self.length):
                if fleeangles[i][j] > np.pi:
                    fleeangles[i][j] -= 2*np.pi
        return fleeangles

    def as_table(self):
        data_preys = np.array(np.split(self.preys, self.n_prey, axis = 0))[:,0]
        if self.n_pred > 0:
            data_preds = np.array(np.split(self.preds, self.n_pred, axis = 0))[:,0]
            data = np.concatenate((np.concatenate(data_preys, axis = 1),
                                np.concatenate(data_preds, axis = 1)), axis = 1)
        else:
            data = np.concatenate(data_preys, axis = 1)
        names = []
        for i in range(self.n_prey):
            names.append(f"x{i}")
            names.append(f"y{i}")
            names.append(f"phi{i}")
            names.append(f"s{i}")
        for pred in range(self.n_pred):
            names.append(f"xp{i}_head")
            names.append(f"yp{i}_head")
            names.append(f"xp{i}_tail")
            names.append(f"y{i}_tail")
            names.append(f"phi{i}")
            names.append(f"s{i}")
        df = pd.DataFrame(data, columns = names)
        self.table = df

    def to_csv(self, path):
        if not self.table:
            self.as_table()
        self.table.to_csv(path)
    
    def to_h5(self, path, key):
        if not self.table:
            self.as_table()
        self.table.to_hdf(path, key = key)

    def plot_trajectories(self, savefig = False, path = "trajectory.pdf"):
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        plt.plot(self.x_prey[0], self.y_prey[0], label = "prey")
        plt.plot(self.x_head[0], self.y_head[0], "--", label = "predator")
        plt.plot(self.x_prey[0][0], self.y_prey[0][0], "r+", label = "prey start")
        plt.plot(self.x_head[0][0], self.y_head[0][0], "r^", label = "predator start")

        T = int(len(self.x_prey[0])/2)
        D_pred = np.max(self.x_head[0,T:]) - np.min(self.x_head[0,T:])
        D_prey = np.max(self.x_prey[0,T:]) - np.min(self.x_prey[0,T:])

        plt.legend([f"Prey, turnradius = {np.round(D_prey)}", f"Predator, turnradius = {np.round(D_pred)}", "Prey start", "Predator Start"], loc='lower right', bbox_to_anchor=(1, 0))
        if savefig:
            plt.savefig(path)
        plt.show()

    def fast_animation(self, path = "test.mp4", sub = 10, see_pred = False):
        mplstyle.use('fast')
        fig, ax = plt.subplots()

        u_prey = np.cos(self.phi_prey)
        v_prey = np.sin(self.phi_prey)
        u_pred = np.cos(self.phi_pred)
        v_pred = np.sin(self.phi_pred)

        if not see_pred:
            x_min = self.x_prey.min() - 15
            x_max = self.x_prey.max() + 15
            y_min = self.y_prey.min() - 15
            y_max = self.y_prey.max() + 15
        else:
            x_min = np.array([self.x_prey.min(), self.x_head.min(), self.x_tail.min()]).min() - 15
            x_max = np.array([self.x_prey.max(), self.x_head.max(), self.x_tail.max()]).max() + 15
            y_min = np.array([self.y_prey.min(), self.y_head.min(), self.y_tail.min()]).min() - 15
            y_max = np.array([self.y_prey.max(), self.y_head.max(), self.y_tail.max()]).max() + 15

        def update(num):
            num = int(num*sub)
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect("equal")
            ax.quiver(self.x_prey[:,num], self.y_prey[:,num], u_prey[:,num], v_prey[:,num])
            ax.quiver(self.x_head[:,num], self.y_head[:,num], u_pred[:,num], v_pred[:,num], color = "red")
            ax.plot(self.x_tail[:,num], self.y_tail[:,num], "r.")
                
        Q_ani = animation.FuncAnimation(fig, update, interval = 1, frames = int(np.ceil(self.length/sub)))
         

        FFMpegWriter = animation.writers['ffmpeg']
        frames = int(np.around((1/self.t_step)/sub))
        writer = FFMpegWriter(fps=frames)
        Q_ani.save(path, writer=writer, dpi=75)
        plt.show() 
                  
    def transfer_entropy(self, mode = "global", history_length = 2):
        dphi = []
        for column in self.values:
            if column[0:4] == "dphi":
                dphi.append(np.array(self.values.iloc[:][column]))
        n_fish = len(dphi)
        time = len(dphi[0]) 
        
        if mode == "global":
            TE = np.zeros((n_fish, n_fish))
            if self.disc != "continuos":
                for i in range(n_fish):
                    for j in range(n_fish):
                        if i != j:
                            TE[i][j] = pin.transfer_entropy(dphi[i], dphi[j], k=history_length)
            elif self.disc == "continous":
                for i in range(n_fish):
                    for j in range(n_fish):
                        if i != j:
                            k = int(history_length/2)
                            x = dphi[i][k:]
                            y = dphi[j][:-k]
                            z = np.zeros(len(x))
                            z[1:] = x[:-1]
                            x=np.reshape(x,(-1,1))
                            y=np.reshape(y,(-1,1))
                            z=np.reshape(z,(-1,1))
                            TE[i][j] = ee.cmi(x,y,z)
                            
        elif mode == "local":
            TE = np.zeros((n_fish, n_fish, time - history_length))
            if self.disc != "continuos":
                for i in range(n_fish):
                    for j in range(n_fish):
                        if i != j:
                            TE[i][j] = pin.transfer_entropy(dphi[i], dphi[j], k=history_length, local=True)[0]
            elif self.disc == "continuos":
                print("Not possible with continuous values (now)")
                
        return TE
    
    def active_information(self, mode = "global", sub = 1, history_length = 2):
        dphi = []
        for column in self.values:
            if column[0:4] == "dphi":
                dphi.append(np.array(self.values.iloc[:][column]))
        n_fish = len(dphi)
        time = len(dphi[0]) 
        
        if mode == "global":
            AIS = np.zeros(n_fish)
            if self.disc != "continuous":
                for i in range(n_fish):
                    AIS[i] = pin.active_info(dphi[i][::sub], k=history_length) 
            elif self.disc == "continuous":
                for i in range(n_fish):
                    k = int(history_length/2)
                    x = dphi[i][1:]
                    z = np.zeros(len(x))
                    z[1:] = x[:-1]
                    x=np.reshape(x,(-1,1))
                    z=np.reshape(z,(-1,1))
                    AIS[i] = ee.mi(x,z)
                    
        elif mode == "local":
            AIS = np.zeros((n_fish, time - history_length))
            if self.disc != "continuos":
                for i in range(n_fish):
                    AIS[i] = pin.active_info(dphi[i], k=history_length, local=True)[0]
            else:
                print("Not possible with continuous values (now)")
        return AIS

    def average_polarization(self, sub = 10):
        P_ges = 0
        v = np.array([np.cos(self.phi_prey), np.sin(self.phi_prey)])
        v = np.transpose(v, axes = (1,2,0))[:,::sub]
        time = len(v[0])
        for i in range(time):
            avg_v = np.sum(v[:,i], axis = 0)
            V = calc.length(avg_v)
            P = V/self.n_prey
            P_ges += P/time
        return P_ges
    
    def average_distance_to_pred(self, sub = 10):
        D_ges = 0
        time = int(np.ceil(self.length/sub))
        for i in range(time):
            i = sub*i
            for j in range(self.n_pred):
                x_dis = self.x_prey[:,i] - self.x_head[j,i]
                y_dis = self.y_prey[:,i] - self.y_head[j,i]
                D = np.sqrt(x_dis**2 + y_dis**2)
                D_ges += np.mean(D)/self.n_pred
        D_ges = D_ges/time
        return D_ges
                    
    def average_interindividual_distance(self, sub = 10):
        time = int(np.ceil(self.length/sub))
        D_ges = 0
        for i in range(time):
            index = sub*i
            x = self.x_prey[:,index]
            y = self.y_prey[:,index]
            total_count = self.n_prey*(self.n_prey - 1)/2
            for i in range(1, self.n_prey):
                for j in range(i):
                    D = np.array([x[i]-x[j], y[i]-y[j]])
                    D_ges += calc.length(D)/total_count          
        D_ges = D_ges/time
        return D_ges
    
    def average_nearest_neighbor_distance(self, sub = 10):
        len = int(np.ceil(self.length/sub))
        for i in range(len):
            index = sub * i
            x = self.x_prey[:,index]
            y = self.y_prey[:,index]
            xy = np.transpose(np.vstack((x,y)))
            Tree = KDTree(xy)
            D, i = Tree.query(xy, k=2)
            D = D[np.nonzero(D)]
        D_ges = np.mean(D)
        return D_ges

    
    def simple_interaction_matrix(self, interaction_threshold = 50):
        D = self.average_distance()
        int_matrix = np.zeros((self.n_prey, self.n_prey))
        for i in range(self.n_prey):
            for j in range(self.n_prey):
                if D[i][j] < 50:
                    int_matrix[i][j] = 1
        return int_matrix
   