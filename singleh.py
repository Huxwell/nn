from itertools import izip
import os
import random
import numpy as np
from PIL import Image as Im

rand = lambda: random.random() - 0.5  # from -0.5 to 0.5)
signum = lambda (x): 1 if x >= 0 else -1
sigmoid = lambda (x): 1.0 / (1.0 + np.exp(-x))
grad_sigmoid = lambda (x): sigmoid(x) * (1 - sigmoid(x))

class NN(object):
    ws_h = [] # 2 dims
    ws_o = [] # 2 dims

    def __init__(self, n_i=77, n_o=10, n_h=44, learn=0.2):
        self.learn = learn
        self.n_i = n_i
        self.n_o = n_o
        self.n_h = n_h
        self.ws_h = [[rand() for i in xrange(n_i)] for h in xrange(n_h)]
        self.ws_o = [[rand() for i in xrange(n_h)] for h in xrange(n_o)]

    def forward(self, xes, y):
        self.hidd = [sum([x * w for x, w in izip(xes,n)]) for n in self.ws_h]
        self.outs = [sum([sigmoid(h) * w for h, w in izip(self.hidd,n)]) for n in self.ws_o]
        print "len ws_h " + str(len(self.ws_o))
        print "len ws_h[0] = len(n) " + str(len(self.ws_o[0]))
        print "len xes " + str(len(self.hidd))
        print self.hidd
        print self.outs
        prop_o = [1 if i==y else -1 for i in xrange(10)]
        print "proper outs: " + str(prop_o)
        err_o = [(proper_out - sigmoid(out)) * grad_sigmoid(out) for out, proper_out in izip(self.outs, prop_o)]
        print err_o
        ws_o_trans = zip(*self.ws_o)
        err_h = [grad_sigmoid(h) * sum([e * w for e, w in izip(err_o,ws_o_trans[h_idx])]) for h_idx, h in enumerate(self.hidd)]
        print err_h


"""
class Neuron(object):
    signal = None
    weights = []
    sum = None #net

    def __init__(self, no_outs, last=False):
        self.weights = rand() if(last) else [rand() for n in xrange(no_outs)]
    # every neutron get no_output synapses,  hidden layers has same no of neurons that output layer


class NN(object):
    weights = []

    #one hidden layer, no_hidd is for neurons in hidden no number of hidden layers
    def __init__(self, no_ins=77, no_outs=10, no_hidd=44, learning=0.2):
        self.learning = learning
        self.no_ins = no_ins
        self.no_outs = no_outs
        self.no_hidd = no_hidd

        self.weights.append([rand() for n in xrange(no_hidd)])

        self.neurons.append([Neuron(no_outs) for n in xrange(no_ins)])  # input layer
        for l in xrange(no_hidd):
            self.neurons.append([Neuron(no_outs) for n in xrange(no_outs)])
        self.neurons.append([Neuron(no_outs, True) for n in xrange(no_outs)])  # output layer

    def clean_neurons(self):
        for l in self.neurons:
            for n in l:
                n.sum = 0
                n.signals = [0 for i in xrange(self.no_outs)] if l != self.neurons[-1] else 0

    def forward(self, dig_data):
        self.clean_neurons()
        # put a digit in input layer
        for n, d in izip(self.neurons[0], dig_data):
            n.sum = d

        for l_id, l in enumerate(self.neurons):
            for n_id, n in enumerate(self.neurons[l_id]):
                if l_id == len(self.neurons) - 1:
                    print "n.weights: " + str(n.weights)
                    print "n.sum: " + str(n.sum)
                    print "sum * weights: " + str(n.sum * n.weights)
                    print "signals: " + str(self.neurons[l_id][n_id].signals)
                    self.neurons[l_id][n_id].signals = n.sum * n.weights
                    print self.neurons[l_id][n_id].signals
                else:
                    for o in xrange(self.no_outs):
                        self.neurons[l_id][n_id].signals[o] = n.sum * n.weights[o]
                        self.neurons[l_id+1][o].sum += self.neurons[l_id][n_id].signals[o]

    def epoch(self, dig_data, dig_no):
        # self.outs =
        self.forward(dig_data)
        print "sums: " + str([n.sum for n in self.neurons[-1]])
        outs = [n.signals for n in self.neurons[-1]]
        print "outs: " + str(outs)
        proper_outs = [1 if i==dig_no else -1 for i in xrange(10)]
        print "proper outs: " + str(proper_outs)
        errors = [ (proper_out - out) /2 for out, proper_out in izip(outs, proper_outs)]
        print "errors: " + str(errors)
        self.update_weights(errors)
    def update_weights(self, errors):
        print "output layer weights:" + str([n.weights for n in self.neurons[-1]])
        for n_idx, n in enumerate(self.neurons[-1]):
            self.neurons[-1][n_idx].weights = self.neurons[-1][n_idx].weights + self.neurons[-1][n_idx].signals * errors[n_idx] * self.learning

        #for l_idx, l in enumerate(reversed(self.neurons)):

        #self.weights[-1] = [[w[0] + o * e * self.learning] for w, o, e in izip(self.weights[-1],self.outs, errors)]  # last layer
        print "output layer weights:" + str([n.weights for n in self.neurons[-1]])
"""
def ldigits():
    return [[int(p[0] + p[1] + p[2] <= 127 * 3) for p in list(Im.open("data/" + f).getdata())] for f in
            os.listdir("data") if ".bmp" in f]