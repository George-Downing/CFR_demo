import math
import numpy as np


# distribution arithmetics
def distrib_encode(F):
    N, D = F.shape
    f = np.concatenate([F, np.ones([N, 1])], axis=1)
    for d in range(D, 0, -1):
        f[:, d] = f[:, d] - f[:, d-1]
    return f

def distrib_decode(f):
    N, D = f.shape
    D = D - 1
    F = np.copy(f[:, 0:D])
    for d in range(1, D):
        F[:, d] += F[:, d-1]
    return F

def rand_F(N, D):
    F = np.random.rand(N, D)
    F.sort(axis=1)
    return F

def rand_f(N, D):
    if D == -1:
        return np.zeros([N, 0])
    F = rand_F(N, D)
    f = distrib_encode(F)
    return f

# combination arithmetics
class CombinationIncrementer(object):
    def __init__(self):
        self.overflow = False
    
    def __call__(self, n, v):
        k = len(v)
        p = k-1
        while p >= 0:
            if v[p] - p == n - k:
                p -= 1
            else:
                break
        if p >= 0:
            v[p:k] = np.arange(v[p] + 1, v[p] + (k-p) + 1)
            self.overflow = False
        else:
            v[:] = np.arange(k)
            self.overflow = True
        return

comb_incr = CombinationIncrementer()

def comb_enum(n, v):
    m = len(v)
    y = math.comb(n, m)
    for k in range(m):
        y -= math.comb(n - v[k] - 1, m - k)
    y -= 1
    return y

def comb_encode(n, No):
    v = np.arange(n)
    ## to be continued... 2022-07-22
    return


def shadow_length_div(N, D):
    Q = np.zeros_like(N)
    Q = N / D
    Q[D <= 0] = np.inf
    Q[N < 0] = 0
    return Q


def smoothmax(v, corner=1.00):  # p.s.: this scalar is also potential function of vector field softmax
    vm = v.max(axis=0)
    w = (v - vm) / corner
    return np.log(np.sum(np.exp(w), axis=0)) * corner + vm

def softmax(v, corner=1.00):
    vm = v.max(axis=0)
    w = (v - vm) / corner
    return np.exp(w) / np.sum(np.exp(w), axis=0) * corner
