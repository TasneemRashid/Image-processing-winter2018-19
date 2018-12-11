import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# reading an intensity image
img = plt.imread('./resources/bauckhage.jpg')

# calculate the coefficients <standard_deviation = sigma>


def coefficients(standard_deviation):
    # we can also replace standard_deviation with "sigma" to make it short!
    # considering input values from Project definition
    coefs = {'alpha1': 1.6800, 'alpha2': -0.6803,
             'beta1': 3.7350, 'beta2': -0.2598,
             'landa1': 1.7830, 'landa2': 1.7230,
             'omega1': 0.6318, 'omega2': 1.9970}

    pos_a0 = coefs['alpha1'] + coefs['alpha2']

    pos_a1 = (np.exp(-coefs['landa2']/standard_deviation)*(coefs['beta2'] *
              np.sin(coefs['omega2']/standard_deviation)-(coefs['alpha2'] +
              2*coefs['alpha1'])*np.cos(coefs['omega2']/standard_deviation))) +
    (np.exp(-coefs['landa1']/standard_deviation)*(coefs['beta1'] *
     np.sin(coefs['omega1']/standard_deviation)-(2*coefs['alpha2'] +
     coefs['alpha1'])*np.cos(coefs['omega1']/standard_deviation)))

    pos_a2 = 2*np.exp(-1*(coefs['landa1'] +
                      coefs['landa2'])/standard_deviation) *
                     ((coefs['alpha1']+coefs['alpha2']) *
                      np.cos(coefs['omega2']/standard_deviation) *
                      np.cos(coefs['omega1']/standard_deviation) -
                      np.cos(coefs['omega2']/standard_deviation) *
                      coefs['beta1']*np.sin(coefs['omega1']/standard_deviation) -
                      np.cos(coefs['omega1']/standard_deviation)*coefs['beta2'] *
                      np.sin(coefs['omega2']/standard_deviation)) +
                      coefs['alpha2']*np.exp(-2*(coefs['landa1'])/standard_deviation) +
                      coefs['alpha1']*np.exp(-2*(coefs['landa2'])/standard_deviation)

    pos_a3 = np.exp(-1*(coefs['landa2']+2*coefs['landa1'])/standard_deviation) *
            (coefs['beta2']*np.sin(coefs['omega2']/standard_deviation) -
            coefs['alpha1']*np.cos(coefs['omega2']/standard_deviation)) +
            np.exp(-1*(coefs['landa1']+2*coefs['landa2'])/standard_deviation) *
            (coefs['beta1']*np.sin(coefs['omega1']/standard_deviation) -
            coefs['alpha1']*np.cos(coefs['omega1']/standard_deviation))

    b1 = (-2) * np.exp(-coefs['landa2']/standard_deviation) *
                np.cos(coefs['omega2']/standard_deviation)-(2) *
                np.exp(-coefs['landa1']/standard_deviation) *
                np.cos(coefs['omega1']/standard_deviation)

    b2 = (4) * np.cos(coefs['omega2']/standard_deviation) *
    np.cos(coefs['omega1']/standard_deviation) *
    np.exp(-(coefs['landa1']+coefs['landa2'])/standard_deviation) +
    np.exp(-2*coefs['landa2']/standard_deviation) +
    np.exp(-2*coefs['landa1']/standard_deviation)

    b3 = (-2) * np.cos(coefs['omega1']/standard_deviation) *
    np.exp(-(coefs['landa1']+(2*coefs['landa2']))/standard_deviation) -
    (2) * np.cos(coefs['omega2']/standard_deviation) *
    np.exp(-(coefs['landa2']+(2*coefs['landa1']))/standard_deviation)

    b4 = np.exp(-((2*coefs['landa1'])+(2*coefs['landa2']))/standard_deviation)

    neg_a1 = pos_a1 - (b1*pos_a0)
    neg_a2 = pos_a2 - (b2*pos_a0)
    neg_a3 = pos_a3 - (b3*pos_a0)
    neg_a4 = -b4 * pos_a0


# since all the neg_b s equal to pos_b s only one is considered here
final_coefs = [pos_a0, pos_a1, pos_a2, pos_a3, b1, b2, b3, b4, neg_a1, neg_a2,
               neg_a3, neg_a4]
return final_coefs


# building the causal part of the main recursive Gaussian filter

# building the anti-causal part of the main recursive Gaussian filter

# combining the causal and anti-causal parts to finally have the filter


""" experimenting the application of different valuses for parameter (sigma)
with our approach """


""" experimenting the application of different valuses for parameter (sigma)
with the built-in function """
