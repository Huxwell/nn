from itertools import izip
import os
import random
import numpy as np
from PIL import Image as Im

rand = lambda: random.random() - 0.5  # from -0.5 to 0.5)
sigmoid = lambda (x): 1.0 / (1.0 + np.exp(-x))
grad_sigmoid = lambda (x): sigmoid(x) * (1 - sigmoid(x))

class NN(object):
    ws_h = [] # 2 dims
    ws_o = [] # 2 dims

    def __init__(self, n_i=77, n_o=10, n_h=44, learn=0.2):
        self.learn = learn
        self.ws_h = [[rand() for i in xrange(n_i)] for h in xrange(n_h)]
        self.ws_o = [[rand() for i in xrange(n_h)] for h in xrange(n_o)]

    def epoch(self, xes, y):
        hidd = [sum([x * w for x, w in izip(xes,n)]) for n in self.ws_h]
        outs = [sigmoid(sum([sigmoid(h) * w for h, w in izip(hidd,n)])) for n in self.ws_o]
        prop_o = [1 if i==y else 0 for i in xrange(10)]
        err_o = [(proper_out - sigmoid(out)) * grad_sigmoid(out) for out, proper_out in izip(outs, prop_o)]
        ws_o_trans = zip(*self.ws_o)
        err_h = [grad_sigmoid(h) * sum([e * w for e, w in izip(err_o,ws_o_trans[h_idx])]) for h_idx, h in enumerate(hidd)]
        self.ws_o = [[w + self.learn * e * sigmoid(net) for w, net in izip(w_n,hidd)] for w_n, e in izip(self.ws_o,err_o)]
        self.ws_h = [[w + self.learn * e * sigmoid(x) for w, x in izip(w_n,xes)] for w_n, e in izip(self.ws_h,err_h)]
        print [round(f,2) for f in outs]
        return outs

def ldigits():
    return [[int(p[0] + p[1] + p[2] <= 127 * 3) for p in list(Im.open("data/" + f).getdata())] for f in
            os.listdir("data") if ".bmp" in f]