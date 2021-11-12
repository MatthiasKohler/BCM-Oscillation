#!/usr/bin/env python3

import numpy as np
import numpy.fft
import scipy.signal
import scipy.optimize

import matplotlib.pyplot as plt
import h5py

import sys
import os

#10 2
scale = 0.1

plt.rcParams.update({'font.size': 9})
plt.rcParams["font.family"] = "Arial"

#plt.tight_layout(pad = -3.0)

sides = ['top', 'right', 'left']

def jitter(x, l = 0.1):
    return x + np.random.uniform(- l / 2, l / 2, len(x))

def hide_spines(ax):
    for s in sides:
        ax.spines[s].set_visible(False)

def autocorrelation(series):
    series_fft =  np.fft.fft(series)
    autocor = np.fft.ifft(series_fft * np.conj(series_fft))
    return np.real(autocor)

def frequency_analysis(state):
    autocor = autocorrelation(state)
    localmaxima = scipy.signal.argrelextrema(autocor, np.greater)[0]
    interval = localmaxima[0] if len(localmaxima) > 0 else 0
    #print(interval, autocor[localmaxima[0]])
    
    #plt.plot(autocor, c = 'g')
    #plt.plot(state, c = 'b')
    #plt.show()
    
    return interval, (autocor[localmaxima[0]] if len(localmaxima) > 0 else 0)

def amplitude_analysis(states):
    t_0 = 500
    amp1s, amp2s = [], []
    for s in states:
        q0, q1 = s[0], s[1]
        #q0, q1 = s[2], s[3]
        max1 = max(q0[t_0:])
        max2 = max(q1[t_0:])
        min1 = min(q0[t_0:])
        min2 = min(q1[t_0:])
        amp1s.append(max1 - min1)
        amp2s.append(max2 - min2)
    return amp1s, amp2s

def analyze_states(states):
    cor1s = []
    cor2s = []
    cor3s = []
    intervals1 = []
    intervals2 = []
    intervals3 = []


    t_0 = 500

    for state in states:
        q0, q1 = state[0], state[1]    
        #q0, q1 = state[2], state[3]    
        interval1, cor1 = frequency_analysis(q0[t_0:])
        interval2, cor2 = frequency_analysis(q1[t_0:])
        interval3, cor3 = frequency_analysis(q0[t_0:] * state[1][t_0:])
        cor1s.append(cor1)
        cor2s.append(cor2)
        cor3s.append(cor3)
        intervals1.append(interval1)
        intervals2.append(interval2)
        intervals3.append(interval3)

    return cor1s, cor2s, cor3s, intervals1, intervals2, intervals3


#def analyze_states2(states):
#    for state in states:
#        crosscor = scipy.signal.correlate(state[0][200:], state[1][200:])
#        plt.plot(crosscor[:800])
#        plt.plot(state[0][200:], color = 'red')
#        plt.plot(state[1][200:], color = 'red')
#        plt.show()

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


def analysis1(pre_learn_states, post_learn_states, plotdir):
    cor1s_pre, cor2s_pre, cor3s_pre, intervals1_pre, intervals2_pre, intervals3_pre = \
        analyze_states(pre_learn_states)
    cor1s_post, cor2s_post, cor3s_post, intervals1_post, intervals2_post, intervals3_post = \
        analyze_states(post_learn_states)

    amp1s_pre, amp2s_pre = amplitude_analysis(pre_learn_states)
    amp1s_post, amp2s_post = amplitude_analysis(post_learn_states)

    amp1s_pre, amp2s_pre, amp1s_post, amp2s_post = \
        map(lambda x: jitter(x, l = 0.01), [amp1s_pre, amp2s_pre, amp1s_post, amp2s_post])

    fig, (ax1, ax2) = plt.subplots(1, 2) 
    fig.set_size_inches(3,3/2)
    #fig.tight_layout(pad = 1.0)
    #fig.tight_layout()
    #fig.subplots_adjust(top=0.2, bottom = 0.2)


    l2 = 1 / scale
    ax1.scatter(scale * jitter(intervals1_pre, l2), 
                scale * jitter(intervals2_pre, l2), color = 'green', s = 1)
    ax1.scatter(scale * jitter(intervals1_post, l2), 
                scale * jitter(intervals2_post, l2), color = 'red', s = 1)
    ax1.set_title("Period [s]", fontsize=9)
    #ax2.set_xlim(-0.1, 3)
    #ax2.set_ylim(-0.1, 3)

    ax2.scatter(amp1s_pre, amp2s_pre, color = 'green', s = 1)
    ax2.scatter(amp1s_post, amp2s_post, color = 'red', s = 1)
    ax2.set_title("Amplitude [rad]", fontsize=9)

#    ax3.scatter(jitter(cor1s_pre), jitter(cor2s_pre), color = "green", s = 1)
#    ax3.scatter(jitter(cor1s_post), jitter(cor2s_post), color = "red", s = 1)
#    ax3.set_title("Autocorrelation", fontsize=9)


    fig.tight_layout()
    fig.savefig(plotdir + 'frequency.png', dpi = 500)

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
    analysis1(pre_learn_states, post_learn_states, plotdir)



if __name__ == "__main__":
    main()
