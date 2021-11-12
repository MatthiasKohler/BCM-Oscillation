#!/usr/bin/env python3

import numpy as np
import numpy.fft
import scipy.signal
import scipy.optimize

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import h5py

import sys
import os


plt.rcParams.update({'font.size': 9})
plt.rcParams["font.family"] = "Arial"
plt.tight_layout()

sides = ['top', 'right', 'left']


def get_states(directory):
    files = os.listdir(directory)
    states = []
    for filename in files:    
        if not "h5" in filename:
            continue
        f_h5 = h5py.File(directory + filename, 'r')
        state = np.array(f_h5['mech_system_state']).T
        states.append(state)
    return states

def plot_raw_data(state, t_start = 0, t_end = None, label = "", directory = None):
    fig, ax = plt.subplots(figsize = (8,0.8))

    #ax.set_yticks([])
    scale = 0.5 #Double Pendulum

    #x1, x2 = state[0][t_start:t_end], state[1][t_start:t_end]
    x1, x2 = state[2][t_start:t_end], state[3][t_start:t_end]

    t = np.array(list(range(t_start, t_end)))
    ax.plot(0.1 * t, x1, color = 'grey')
    ax.plot(0.1 * t, x2, color = 'black')

    if directory is not None:
        #fig.tight_layout()
        fig.subplots_adjust(bottom=0.4) 
        fig.savefig(directory + "raw_data_" + label + ".png", dpi = 400)
    else:
        plt.show()


def main():
    print(sys.argv)
   
    directory = sys.argv[1]
    directory_pre_learn  = directory + "/pre_learn/"
    directory_post_learn = directory + "/post_learn/"

    pre_learn_states = get_states(directory_pre_learn)
    post_learn_states = get_states(directory_post_learn)

    print("Number of pre learn tests:", len(pre_learn_states))
    print("Number of post learn tests:", len(post_learn_states))

    plotdir = directory + '/plots/'
    try:
        os.mkdir(plotdir)
    except:
        pass

    i = 1
    plot_raw_data(pre_learn_states[i], label = "before" + str(i), t_end = 1000, directory = plotdir)
    plot_raw_data(post_learn_states[i], label = "after" + str(i), t_end = 1000, directory = plotdir)


if __name__ == "__main__":
    main()
