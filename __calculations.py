import numpy as np
import pandas as pd
from scipy.spatial import Voronoi, voronoi_plot_2d
from numba import jit
from numpy.random import normal
import time
import matplotlib.pyplot as plt


@jit(nopython = True, fastmath = True)
def pred_distance(r_pred, positions):
    pos = np.array(positions)
    r = pos - np.array(r_pred)
    d = np.linalg.norm(r, axis = 1)
    d_min = np.min(d)
    return d_min

@jit(nopython = True, fastmath = True)
def centerofmass(positions):
    r_com = np.array([0.,0.])
    N = len(positions)
    for r in positions:
        r_com += r/N
    return r_com

@jit(nopython = True, fastmath = True)
def pred_startpoint(positions, phis, alpha = 0, atk_distance = 100):
    r_com = np.array([0.,0.])
    v_com = np.array([0.,0.])
    N = len(positions)
    for r in positions:
        r_com += r/N
    for phi in phis:
        v_com += np.array([np.cos(phi), np.sin(phi)])/N
    phi_com = np.arctan2(v_com[1], v_com[0])
    angle = phi_com + alpha
    R = np.array([np.cos(angle), np.sin(angle)])
    start_point = r_com + R*atk_distance
    return start_point

def predator_start(back_pos, orientation, distance = 70, sigma = 0.5):
    phi = np.arctan2(orientation[1], orientation[0]) + sigma*np.random.uniform(-1,1)
    ori = np.array([np.cos(phi), np.sin(phi)])
    r_pred = back_pos - (distance * ori)
    return r_pred
    
def random_positions(N, d, sigma = 0.2, mid_point = np.array([0,0])):
    positions = np.zeros((N,2))
    positions[0] = mid_point + np.array([sigma*d*np.random.uniform(), sigma*d*np.random.uniform()])

    N_fit = 3 # number of particles that fit on current circle
    N_cap = N_fit #number of particles that fit on all previous and the current circle
    circle = 1 #circle
    N_circ = 0 #number of agents on current circle

    for i in range(N-1):
        index = i+1
        if i == N_cap:
            circle += 1
            N_fit = int(2*np.pi*circle)
            N_cap += N_fit
            N_circ = 0
        positions[index] = mid_point + np.array([d * circle * np.cos(2*np.pi*N_circ/N_fit) + sigma*d*np.random.uniform(),
                                                 d * circle * np.sin(2*np.pi*N_circ/N_fit) + sigma*d*np.random.uniform()])
        N_circ += 1
    
    return positions

@jit(nopython = True, fastmath = True)
def plot_boundaries(positions):
    """
    set of positions with N*[x,y]
    """
    ran = range(len(positions))
    xs = [positions[i,0] for i in ran]
    ys = [positions[i,1] for i in ran]
    return (min(xs), max(xs), min(ys), max(ys))

def mov_avg(array, interval_size = 50):
    sma = pd.Series(array).rolling(interval_size).sum()/interval_size
    return np.array(sma)

@jit(nopython = True, fastmath = True)
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

@jit(nopython = True, fastmath = True)
def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    angle = np.arctan2(v1[1], v1[0]) - np.arctan2(v2[1], v2[0])
    if angle > np.pi:
        angle = angle - 2*np.pi
    return angle

@jit(nopython = True, fastmath = True)    
def get_avg_series(phi_series):
    avg_series = np.zeros(len(phi_series[0]))
    n = len(phi_series)
    for series in phi_series:
        avg_series += np.array(series)/n
    
    return avg_series

@jit(nopython = True, fastmath = True)
def get_dphi_series(phi, sub):
    dphi = np.zeros(len(phi))
    for i in range(len(dphi)-sub):
        if phi[i] > phi[i+sub]:
            dphi[i] = 1
            
    return dphi

@jit(nopython = True, fastmath = True)
def sigmoid(alpha, d, r):
    S = 0.5 * (np.tanh(alpha*(d - r)) + 1)
    return S

@jit(nopython = True, fastmath = True)
def length(v):
    L = np.linalg.norm(v)
    return L

@jit(nopython = True, fastmath = True)
def all_interactions(r_i, positions):
    """
    just returns indices
    """
    n = len(positions)
    ag_indices = np.array([i for i in range(n) if (positions[i] != r_i).any()])
    return ag_indices


@jit(nopython = True, fastmath = True)
def nnn_interactions(r_i, positions, N_nearest = 8):
    """
    i : agent
    j : agents including agent
    """
    n = len(positions) - 1
    ag_indices = np.array([i for i in range(n+1) if (positions[i] != r_i).any()])
    if N_nearest < n+2:
        D = np.zeros(n)
        for j in range(n):
            r = np.abs(r_i - positions[j])
            D[j] = 0.96*max(r) + 0.4*min(r) #faster approximation of distance (4% Error)
        Idx = np.argsort(D)[:N_nearest]
        indices = np.array([ag_indices[i] for i in Idx])
    else:
        indices = ag_indices
    return indices

@jit(nopython = True, fastmath = True)
def range_interactions(r_i, positions, ran = 40):
    n = len(positions) - 1
    ag_indices = np.array([i for i in range(n+1) if (positions[i] != r_i).any()])
    D = np.zeros(n)
    counter = 0
    for j in range(n):
        r = np.abs(r_i - positions[j])
        D[j] = 0.96*max(r) + 0.4*min(r) #faster approximation of distance (4% Error)
        if D[j] > ran:
            D[j] = 0
            counter += 1
    Idx = np.argsort(D)[counter:]
    indices = [ag_indices[i] for i in Idx]
    return indices 

def voronoi(positions):
    if len(positions) >= 4:
        V = Voronoi(positions)
    else:
        V = 0
    return V
    
def voronoi_regions(V):
    if V == 0:
        point_regions = 0
    else:
        indices = V.point_region
        regions = V.regions
        point_regions = [regions[i] for i in indices]
    return point_regions

def voronoi_adj(point_regions, ag_index):
    if point_regions == 0:
        indices = 0
    else:
        ag_vertices = [p for p in point_regions[ag_index] if p != -1]
        indices = [i for i in range(len(point_regions)) if (not set(point_regions[i]).isdisjoint(ag_vertices) and i != ag_index)]
    return indices

def voronoi_interactions(r_i, positions, voronoi_reg):
    if type(voronoi_regions) == "int":
        point_regions = voronoi_regions(positions)
    else:
        point_regions = voronoi_reg
    ag_index = np.where(positions == r_i)[0][0]
    indices = voronoi_adj(point_regions, ag_index)
    if indices == 0:
        indices = all_interactions(r_i, positions)
    return indices

@jit(nopython = True, fastmath = True)
def all_interactions_pred(r_i, pred_positions, tail_positions):
    """
    just returns indices
    """
    n = len(pred_positions)
    pred_indices = np.array([i for i in range(n)])
    return pred_indices

@jit(nopython = True, fastmath = True)
def range_interactions_pred(r_i, pred_positions, tail_positions, ran = 50):
    n = len(pred_positions)
    pred_indices = np.array([i for i in range(n)])

    D = np.zeros(n)
    D_tail = np.zeros(n)
    counter = 0
    for i in range(n):
        r_head = np.abs(r_i - pred_positions[i])
        r_tail = np.abs(r_i - tail_positions[i])
        D[i] = 0.96*max(r_head) + 0.4*min(r_head) #faster approximation of distance (4% Error)
        D_tail[i] = 0.96*max(r_tail) + 0.4*min(r_tail)
        if D[i] > 50 and D_tail[i] > 70:
            D[i] = 0
            counter += 1
    Idx = np.argsort(D)[counter:]
    indices = [pred_indices[i] for i in Idx]

    return indices

@jit(nopython = True, fastmath = True)
def preyprey_force(r_i, phi_i, positions, phis, mu_con, a_con, r_con, voronoi = False):
    """
    forces of prey on prey
    """
    #set up forces
    F_att = np.array([0.,0.])
    F_rep = np.array([0.,0.])
    F_alg = np.array([0.,0.])
    for j in range(len(phis)):
        #vectors between agents
        r_j = positions[j]
        phi_j = phis[j]
        r_ij = r_j - r_i
        v_ij = np.array([np.cos(phi_j), np.sin(phi_j)]) - np.array([np.cos(phi_i), np.sin(phi_i)])
        dis = length(r_ij)

        #calc forces
        if voronoi:
            F_att += mu_con[0] * sigmoid(a_con[0], dis, r_con[0]) * r_ij/dis
            F_rep -= mu_con[1] * sigmoid(a_con[1], dis, r_con[1]) * r_ij/dis
            F_alg += mu_con[2] * v_ij
        else:
            F_att += mu_con[0] * sigmoid(a_con[0], dis, r_con[0]) * r_ij/dis
            F_rep -= mu_con[1] * sigmoid(a_con[1], dis, r_con[1]) * r_ij/dis
            F_alg += mu_con[2] * sigmoid(a_con[2], dis, r_con[2]) * v_ij
    
    F_i = F_att + F_rep + F_alg

    return F_i

@jit(nopython = True, fastmath = True)
def predprey_force(r_i, pred_positions, tail_positions, mu_pred, a_pred, r_pred):
    """
    forces of pred on prey
    """
    F_att = np.array([0.,0.])
    F_rep = np.array([0.,0.])

    for j in range(len(pred_positions)):
        r_headi = pred_positions[j] - r_i
        r_taili = tail_positions[j] - r_i
        dis_head = length(r_headi)
        dis_tail = length(r_taili)

        sig = 5
        if dis_head <= r_pred[0] + sig:
            F_rep -= mu_pred[0] * sigmoid(a_pred[0], dis_head, r_pred[0])* r_headi/dis_head
        if dis_tail <= r_pred[1] + sig:
            F_att += mu_pred[1] * sigmoid(a_pred[1], dis_tail, r_pred[1]) * r_taili/dis_tail

    F_i = F_att + F_rep
    return F_i

@jit(nopython = True, fastmath = True)
def pos_force(r_i, start_point, mu, positions):
    R = start_point - r_i
    D = length(R)
    F_att = mu * (R/D) # Attraction to Attack Point

    r_com = centerofmass(positions)
    r_comi = r_com - r_i 
    r_rep = length(r_com - start_point)
    F_rep = - mu * sigmoid(-0.5, length(r_comi), r_rep) * r_comi/length(r_comi)

    return F_att + F_rep

@jit(nopython = True, fastmath = True)
def fleeangle_only(r_i, pred_positions, tail_positions, mu_pred, a_pred, r_pred):
    theta = np.pi/3
    F_i = np.array([0.,0.])

    if len(pred_positions) > 0:
        for j in range(len(pred_positions)):
            #pred orientation
            v_pred = pred_positions[j] - tail_positions[j]
            phi_pred = np.arctan2(v_pred[1], v_pred[0])
            
            #polar angle relative to predator
            r = r_i - pred_positions[j]
            angle = np.arctan2(r[1], r[0])
            if 0 < angle < np.pi:
                fleeangle = theta
            else:
                fleeangle = -theta
            
            phi = phi_pred + fleeangle
            F_i += mu_pred[0]*np.array([np.cos(phi), np.sin(phi)])
    else:
        pass
    return F_i
            


@jit(nopython = True, fastmath = True)
def preypred_force(r_pred, positions, mu_prey, angle = 0):
    N = len(positions)
    r_com = np.array([0.,0.])
    d_closest = 100000000
    for j in range(N):
        r_i = positions[j]
        r_com = r_com + r_i/N
        r_ip = r_pred - r_i
        d_ip = np.linalg.norm(r_ip)
        if d_ip < d_closest:
            d_closest = d_ip
    r_comp = r_com - r_pred
    d_com = np.linalg.norm(r_comp)

    r = np.array([r_comp[0]*np.cos(angle) - r_comp[1]*np.sin(angle), r_comp[0]*np.sin(angle) + r_comp[1]*np.cos(angle)])
    F_att = mu_prey * (r/d_com)
    
    return F_att

@jit(nopython = True, fastmath = True)
def predpred_force(r_i, pred_positions, mu_con_pred, r_con_pred):
    N = len(pred_positions)
    F_rep = np.array([0.,0.])
    
    return F_rep

@jit(nopython = True, fastmath = True)
def move_prey(force, t_step, position, phi, s, A, s0, sigma):
    v_i = np.array([np.cos(phi), np.sin(phi)])
    u_i = np.array([-np.sin(phi), np.cos(phi)])
    F_phi = np.dot(force, u_i)
    F_vel = np.dot(force, v_i)
    dphi = t_step * F_phi
    phi_between = angle_between(force, v_i)
    if np.absolute(dphi) > np.absolute(phi_between):
        dphi = phi_between
    ds = t_step * F_vel + A*(s0 - s)

    new_phi = phi + dphi + sigma*np.sqrt(t_step)*normal()
    new_s = s + ds + sigma*np.sqrt(t_step)*normal()
    new_r = position + t_step * new_s * np.array([np.cos(new_phi),np.sin(new_phi)])
    return new_r, new_phi, new_s

@jit(nopython = True, fastmath = True)
def move_pred(force, t_step, position, phi, s, sigma):
    u_i = np.array([-np.sin(phi), np.cos(phi)])
    F_phi = np.dot(force, u_i)
    dphi = t_step * F_phi

    new_phi = phi + dphi + sigma*np.sqrt(t_step)*normal()
    new_s = s
    new_r = position + t_step * new_s * np.array([np.cos(new_phi),np.sin(new_phi)])
    return new_r, new_phi, new_s

@jit(nopython = True, fastmath = True)
def new_tail(r, phi, length):
    new_tail = np.array([r[0] - length*np.cos(phi),
                        r[1] - length*np.sin(phi)])
    return new_tail






            





        
            