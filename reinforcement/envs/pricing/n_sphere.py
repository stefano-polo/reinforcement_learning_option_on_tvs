import numpy as np

def n_sphere_to_cartesian(radius, angles):
    """Function that takes n-spherical coordinates and convert into n-cartesian coordinates
       angles: the n-2 values between [0,\pi) and last one between [0,2\pi)
    """
    a = np.concatenate((np.array([2*np.pi]), angles))
    si = np.sin(a)
    si[0] = 1
    si = np.cumprod(si)
    co = np.cos(a)
    co = np.roll(co, -1)
    return si*co*radius


def sign_renormalization(vec, sum_plus, sum_minus):
    """renormalize positive and negative elements of an array separately if necessary such 
    that sum of positive is equal to sum_plus and the abs sum of negative is equal to sum_neg"""
    pos = vec>=0
    neg = (1-vec)>0
    p_s = np.sum(vec[pos])
    n_s = abs(np.sum(vec[neg]))
    if p_s > sum_plus:
        vec[pos] = (vec[pos]/p_s)*sum_plus
    if n_s > sum_minus:
        vec[neg] = (vec[neg]/n_s)*sum_minus
    p_s = np.sum(vec[pos])
    n_s = abs(np.sum(vec[neg]))
    return vec