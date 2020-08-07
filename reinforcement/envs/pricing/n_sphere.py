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