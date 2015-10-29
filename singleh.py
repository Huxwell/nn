from itertools import izip
import os
import random
import numpy as np
from PIL import Image as Im
import cv2


rand = lambda: random.random() - 0.5  # from -0.5 to 0.5)
sigmoid = lambda (x): 1.0 / (1.0 + np.exp(-x))
grad_sigmoid = lambda (x): sigmoid(x) * (1 - sigmoid(x))


class NN(object):
    ws_hidden = [] # 2 dims
    ws_out = [] # 2 dims

    def __init__(self, n_i=77, n_o=10, n_h=44, learn=0.1):
        self.learn = learn
        self.ws_hidden = [[rand() for i in xrange(n_i)] for h in xrange(n_h)]
        self.ws_out = [[rand() for i in xrange(n_h)] for h in xrange(n_o)]

    def epoch(self, xes, y):
        net_hidden = [sum([x * w for x, w in izip(xes,n)]) for n in self.ws_hidden]
        net_out = [sum([sigmoid(h) * w for h, w in izip(net_hidden,ws)]) for ws in self.ws_out]
        outs = [sigmoid(net) for net in net_out]

        prop_outs = [1 if i==y else 0 for i in xrange(10)]
        # print prop_outs
        # print net_out
        err_o = [(proper_out - sigmoid(net)) * grad_sigmoid(net) for net, proper_out in izip(net_out, prop_outs)]
        # print err_o
        self.error = sum([e**2 for e in err_o]) *0.5
        print self.error
        ws_out_trans = zip(*self.ws_out)
        err_h = [grad_sigmoid(net) * sum([e * w for e, w in izip(err_o, ws)]) for ws, net in izip(ws_out_trans, net_hidden)]
        self.ws_out = [[w + self.learn * e * sigmoid(net) for w, net in izip(w_n, net_hidden)] for w_n, e in izip(self.ws_out, err_o)]
        self.ws_hidden = [[w + self.learn * e * x for w, x in izip(w_n, xes)] for w_n, e in izip(self.ws_hidden, err_h)]
        # print [round(f,2) for f in outs]

        return outs


def ldigits():
    return [[int(p[0] + p[1] + p[2] <= 127 * 3) for p in list(Im.open("data/" + f).getdata())] for f in
            os.listdir("data") if ".bmp" in f]