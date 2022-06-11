import numpy as np
from scipy import sparse

def L2_h(t_z, T):
    return 2

def ADT_h(t_z, T):
    return (T - t_z)/T

def ADT2_h(t_z, T):
    return ((T - t_z)/T)**2

def ADiT_h(t_z, T):
    return 1/t_z

def ADiT2_h(t_z, T):
    return 1/(t_z*t_z)

################################

def L2_after(x, P_full, m_p):
    P = P_full[0]
    xv = sparse.csr_matrix(P.shape)
    for x_i in x:
        xv[0, x_i] = m_p
    return (sparse.linalg.norm(P - xv)/m_p)**2

def Z_minus(x, P_full, m_p):
    P = P_full[0]

    xv = sparse.csr_matrix(P.shape)
    for x_i in x:
        xv[0, x_i] = 1

    return -(P*xv.T).sum()

################################

def L2(G, x, Y, s):
    return len(x ^ Y.keys())

def Ld(G, x, Y, s):
    k = 3
    count = 0
    for node in x ^ Y.keys():
        if G.dist(node, s) <= k:
            count += 1
    return count

def Le(G, x, Y, s):
    loss = 0.0
    for node in x ^ Y.keys():
        loss += np.exp(-G.dist(node, s))
    return loss

def Ll(G, x, Y, s):
    loss = 0.0
    for node in x ^ Y.keys():
        loss += 1.0/G.dist(node, s)
    return loss

def Lm(G, x, Y, s):
    if x == Y.keys():
        return 0.0
    return 1.0

def first_miss_loss(G, x, samples, s, weights = None):
    min_steps = []
    if not weights is None:
        for sample, weight in zip(samples, weights):
            for i in range(len(sample)):
                if not sample[i] in x:
                    min_steps += [weight*i]
                    break
    else:
        for sample in samples:
            for i in range(len(sample)):
                if not sample[i] in x:
                    min_steps += [i]
                    break
    return -np.mean(min_steps)

def avg_deviation_time(G, x, samples, s):
    avg_steps = list()
    T_l = len(samples[0])
    for sample in samples:
        avg_steps.append(np.mean([T_l - i*(not sample[i] in x) for i in range(T_l)]))
        #for i in range(T):
        #    if not sample[i] in x:
        #        diff_steps.append(T - i)
        #avg_steps.append(np.mean(diff_steps))
    return np.mean(avg_steps)

def avg_matching_time(G, x, samples, s):
    avg_match = list()
    T_l = len(samples[0])
    for sample in samples:
        #same_steps = list()
        #T = len(sample)
        #for i in range(T):
        #    if sample[i] in x:
        #        same_steps.append(i)
        #avg_match.append(np.mean(same_steps))
        avg_match.append(np.mean([i*(sample[i] in x) for i in range(T_l)]))
    return -np.mean(avg_match)

def distance_loss(G, x, samples, s):
    return sum([G.dist(i, s) for i in x])

def min_dist(G, x, samples, s):
    return -1*min([G.dist(i, s) for i in set(G.graph.nodes()) - set(x)])

def max_dist(G, x, samples, s):
    return max([G.dist(i, s) for i in x])

##############################################################

def loss_creator(loss_func):
    def loss_wrapper(G, x, samples_Y, ratios, s):
        return sum([loss_func(G, x, Y, s)*ratio for Y, ratio in zip(samples_Y, ratios)])
    return loss_wrapper
