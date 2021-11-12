#!/usr/bin/env python3

import numpy as np
import numpy.fft
import scipy.signal
import scipy.optimize

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import collections  as mc
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import h5py

import sys
import os

plt.rcParams.update({'font.size': 9})
plt.rcParams["font.family"] = "Arial"
plt.tight_layout()

def plot_trajectory(state, t_start = 0, t_end = 1000, directory = None):
    q0, q1 = state[2][t_start:t_end], state[3][t_start:t_end]
    #q0, q1 = q0 - np.pi, q1 - np.pi
    num = q0.shape[0]

    t = np.linspace(t_start, t_end, num = num)

    joint_1_positions_t = list(zip(np.cos(q0) + t             , np.sin(q0)             ))
    joint_2_positions_t = list(zip(np.cos(q0) + np.cos(q1) + t, np.sin(q0) + np.sin(q1)))
    joint_0_positions_t = list(zip(t, [0.0] * num))

    lines = list(zip(joint_0_positions_t, joint_1_positions_t, joint_2_positions_t))

    lc = mc.LineCollection(lines, colors = 'black')
    fig, ax0 = plt.subplots(figsize = (8,1.6))    
    ax0.add_collection(lc)
    ax0.autoscale()
    ax0.set_aspect(1)
    ax0.scatter(t,  [0.0] * num, c = 'black')
    ax0.set_xticks(np.floor(t[::10]))
    ax0.set_xticklabels(np.floor(np.array(t[::10] / 10)).astype(int))
    ax0.set_xlabel("Time [s]")
    ax0.set_yticks([])

    #fig.tight_layout() 
    fig.savefig(directory + 'trajectory_2.png', dpi = 400)

    plt.show()

def make_segments(a, b, offset):
    points = np.array([a + offset, b]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def plot_selectivity(state, state_neurons, t_start = 0, t_end = 1000, directory = None):
    fig, ax = plt.subplots(1, 1, figsize = (4,2)) #2.7 2
    q0 = state[2][t_start:t_end]
    q1 = state[3][t_start:t_end]
    q0_d = state[0][t_start:t_end]
    q1_d = state[1][t_start:t_end]
    num = q0.shape[0]

    V_neurons = state_neurons[:,t_start:t_end]
    
    cmap = plt.get_cmap("plasma")
    #norm = BoundaryNorm([-1, -0.5, 0.5, 1], cmap.N)

    plot = None

    for neuron in range(8):
        offset = - 0.3 * neuron #+ (0.001 * np.linspace(t_start, t_end, num = num))
        offset_d = (2*neuron) #+ (0.001 * np.linspace(t_start, t_end, num = num))

        x = np.cos(q0) + np.cos(q1)
        y = np.sin(q0) + np.sin(q1)

        #segments = make_segments(q0, q1, offset)
        segments = make_segments(x, y, offset)
        segments_d = make_segments(q0_d, q1_d, offset_d)


        lc = LineCollection(segments, cmap=cmap)
        lc.set_array(np.array(V_neurons[neuron][:-1]))
        lc.set_linewidth(1)
        lc.set_clim(vmin = 0.4, vmax = 0.8)

        #lc_d = LineCollection(segments_d, cmap=cmap)
        #lc_d.set_array(np.array(V_neurons[neuron][:-1]))
        #lc_d.set_linewidth(1)
        #lc_d.set_clim(vmin = 0.2, vmax = 0.8)

        ax.add_collection(lc)
        ax.autoscale()
        #ax[0].scatter(0, 0 + neuron, c = 'black')
        #ax[1].add_collection(lc_d)
        #ax[1].autoscale()
        #ax[1].scatter(0, 0 + 2*neuron, c = 'black')

        plot = lc

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(plot, cax=cax, label = "")



    scalebar = AnchoredSizeBar(ax.transData,
                           1, '1', 'lower left', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=0)
    ax.add_artist(scalebar)

    scalebar = AnchoredSizeBar(ax.transData,
                           0, '', 'lower left', 
                           pad=0.1,
                           color='black',
                           frameon=False,
                           size_vertical=1)
    ax.add_artist(scalebar)

    
    ax.set_aspect(1)    

    
    #scalebar = AnchoredSizeBar(ax[1].transData,
    #                       0, '1 rad/s', 'lower left', 
    #                       pad=0.1,
    #                       color='black',
    #                       frameon=False,
    #                       size_vertical=1)
    #ax[1].add_artist(scalebar)


    #ax.set_xlabel('$\\theta_1$ [rad]')
    #ax.set_ylabel('$\\theta_2$ [rad]')
    ax.set_xlabel('$x$, Neurons')
    ax.set_ylabel("")

    ax.set_xticks([])
    ax.set_yticks([])


    #ax[1].set_xlabel('$\dot{\\theta}_1$ [rad/s]')
    #ax[1].set_ylabel('$\dot{\\theta}_2$ [rad/s]')
    #ax[1].set_yticks([])

    fig.tight_layout()

    fig.savefig(directory + 'trajectory.png', dpi = 400)

    #plt.show()

def plot_cycle(state, t_start = 0, t_end = 1000, directory = None):
    q0, q1 = state[2][t_start:t_end], state[3][t_start:t_end]

    x = np.cos(q0) + np.cos(q1)
    y = np.sin(q0) + np.sin(q1)

    fig, ax = plt.subplots(figsize = (1.6,1.6))

    ax.plot(x, y)

    ax.set_aspect(1)    

    plt.show()

def get_states(directory):
    files = os.listdir(directory)
    states_mech = []
    states_neurons = []
    for filename in files:    
        if not "h5" in filename:
            continue
        f_h5 = h5py.File(directory + filename, 'r')
        state_mech = np.array(f_h5['mech_system_state']).T
        state_neurons = np.array(f_h5['V_neurons']).T
        states_mech.append(state_mech)
        states_neurons.append(state_neurons)
    return states_mech, states_neurons

def main():
    print(sys.argv)
   
    directory = sys.argv[1]
    directory_pre_learn  = directory + "/pre_learn/"
    directory_post_learn = directory + "/post_learn/"

    #_ = get_states(directory_pre_learn)
    states_mech, states_neurons = get_states(directory_post_learn)

    #print("Number of pre learn tests:", len(pre_learn_states))
    #print("Number of post learn tests:", len(post_learn_states))

    plotdir = directory + '/plots/'
    try:
        os.mkdir(plotdir)
    except:
        pass

    i = 1
    plot_trajectory(states_mech[i], t_start = 410, t_end = 440, directory = plotdir)
    #plot_selectivity(states_mech[i], states_neurons[i], t_start = 800, t_end = 840, directory = plotdir)
    plot_selectivity(states_mech[i], states_neurons[i], t_start = 421, t_end = 440, directory = plotdir)
    #plot_cycle(states_mech[i], t_start = 421, t_end = 440, directory = plotdir)


if __name__ == "__main__":
    main()
