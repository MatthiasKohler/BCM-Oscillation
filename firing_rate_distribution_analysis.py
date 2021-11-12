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

#import  matplotlib.font_manager
#flist = matplotlib.font_manager.get_fontconfig_fonts()
#print(flist)
#names = [matplotlib.font_manager.FontProperties(fname=fname).get_name() for fname in flist]
#print(names)

sides = ['bottom', 'top', 'right', 'left']

def hide_spines(ax):
    for s in sides:
        ax.spines[s].set_visible(False)

def get_states(directory):
    files = os.listdir(directory)
    states = []
    for filename in files:    
        f_h5 = h5py.File(directory + filename, 'r')
        state = np.array(f_h5['V_neurons']).T
        states.append(state)
    return states


def analysis3(pre_learn, post_learn, directory = None):
    n_neurons = pre_learn[0].shape[0]
    print(n_neurons)
    fig, ax = plt.subplots(n_neurons, figsize = (2.7, 2))
    
    means_pre = []
    means_post = []

    for neuron in range(n_neurons):
        ax[neuron].set_xlim(0, 0.8)
        #ax[neuron].set_xlim(0.2, 1)
        ax[neuron].set_ylim(-20, 500)

        ax[neuron].set_yticks([])
        if neuron < n_neurons - 1:
            ax[neuron].set_xticks([])

        #hide_spines(ax[neuron]) 

        rates_pre = np.array([])
        rates_post = np.array([])
        


        for rate in pre_learn:
            rates_pre = np.append(rates_pre, rate[neuron])
        for rate in post_learn:
            rates_post = np.append(rates_post, rate[neuron])

        
        mean_rate_pre = np.mean(rates_pre)
        mean_rate_post = np.mean(rates_post)
        print(mean_rate_post)

        means_pre.append(mean_rate_pre)
        means_post.append(mean_rate_post)

        ax[neuron].hist(rates_pre, bins = 1000, color = 'green', alpha = 0.5)
        ax[neuron].hist(rates_post, bins = 1000, color = 'red', alpha = 0.5)
        ax[neuron].axvline(mean_rate_pre, lw = 1, color = 'green')
        ax[neuron].axvline(mean_rate_post, lw = 1, color = 'red')
        #ax[neuron].axvline(np.median(rates_post), lw = 1, color = 'pink')

        median = np.median(rates_post)
        print(median, 'median')

        l = len(rates_post[rates_post < median] )
        print(l)


    
    if directory is not None:
        fig.savefig(directory + "firing_rate_distribution" + ".png", dpi = 400)
    else:
        plt.show()

    print("Pre: Mean ", np.mean(means_pre), "SD", np.std(means_pre))
    print("Post: Mean ",np.mean(means_post), "SD", np.std(means_post))

def main():
    print(sys.argv)
   
    directory = sys.argv[1]
    directory_pre_learn  = directory + "/pre_learn/"
    directory_post_learn = directory + "/post_learn/"

    pre_learn = get_states(directory_pre_learn)
    post_learn = get_states(directory_post_learn)

    print("Number of pre learn tests:", len(pre_learn))
    print("Number of post learn tests:", len(post_learn))

    plotdir = directory + '/plots/'
    try:
        os.mkdir(plotdir)
    except:
        pass
    analysis3(pre_learn, post_learn, directory = plotdir)

if __name__ == "__main__":
    main()
