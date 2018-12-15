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

    pos_a1 = (np.exp(-coefs['landa2']/standard_deviation)*(coefs['beta2']*np.sin(coefs['omega2']/standard_deviation)-(coefs['alpha2']+2*coefs['alpha1'])*np.cos(coefs['omega2']/standard_deviation)))+(np.exp(-coefs['landa1']/standard_deviation)*(coefs['beta1']*np.sin(coefs['omega1']/standard_deviation)-(2*coefs['alpha2']+coefs['alpha1'])*np.cos(coefs['omega1']/standard_deviation)))

    pos_a2 = 2*np.exp(-1*(coefs['landa1']+coefs['landa2'])/standard_deviation)*((coefs['alpha1']+coefs['alpha2']) * np.cos(coefs['omega2']/standard_deviation)*np.cos(coefs['omega1']/standard_deviation)-np.cos(coefs['omega2']/standard_deviation)*coefs['beta1']*np.sin(coefs['omega1']/standard_deviation)-np.cos(coefs['omega1']/standard_deviation)*coefs['beta2']*np.sin(coefs['omega2']/standard_deviation))+coefs['alpha2']*np.exp(-2*(coefs['landa1'])/standard_deviation)+coefs['alpha1']*np.exp(-2*(coefs['landa2'])/standard_deviation)

    pos_a3 = np.exp(-1*(coefs['landa2']+2*coefs['landa1'])/standard_deviation)*(coefs['beta2']*np.sin(coefs['omega2']/standard_deviation)-coefs['alpha1']*np.cos(coefs['omega2']/standard_deviation))+np.exp(-1*(coefs['landa1']+2*coefs['landa2'])/standard_deviation)*(coefs['beta1']*np.sin(coefs['omega1']/standard_deviation)-coefs['alpha1']*np.cos(coefs['omega1']/standard_deviation))

    b1 = (-2)*np.exp(-coefs['landa2']/standard_deviation)*np.cos(coefs['omega2']/standard_deviation)-(2)*np.exp(-coefs['landa1']/standard_deviation)*np.cos(coefs['omega1']/standard_deviation)

    b2 = (4)*np.cos(coefs['omega2']/standard_deviation)*np.cos(coefs['omega1']/standard_deviation)*np.exp(-(coefs['landa1']+coefs['landa2'])/standard_deviation)+np.exp(-2*coefs['landa2']/standard_deviation)+np.exp(-2*coefs['landa1']/standard_deviation)

    b3 = (-2)*np.cos(coefs['omega1']/standard_deviation)*np.exp(-(coefs['landa1']+(2*coefs['landa2']))/standard_deviation)-(2)*np.cos(coefs['omega2']/standard_deviation)*np.exp(-(coefs['landa2']+(2*coefs['landa1']))/standard_deviation)

    b4 = np.exp(-((2*coefs['landa1'])+(2*coefs['landa2']))/standard_deviation)

    neg_a1 = pos_a1 - (b1*pos_a0)
    neg_a2 = pos_a2 - (b2*pos_a0)
    neg_a3 = pos_a3 - (b3*pos_a0)
    neg_a4 = -b4 * pos_a0
# since all the neg_b s equal to pos_b s only one is considered here
    final_coefs = [pos_a0, pos_a1, pos_a2, pos_a3, b1, b2, b3, b4, neg_a1, neg_a2, neg_a3, neg_a4]
    return final_coefs


# building the causal part of the main recursive Gaussian filter


def causal_filter(image, coefficients):
    """ we use image, as for x but 3 elements should be concatenated to the
    beginning of the image array to prevent negative indexing """
    # initializing y to recursively/ gradually fill it in
    y = np.zeros((img.shape[0],img.shape[1]), dtype=float)
    y[3] = coefficients[0]*img[0]
    for i in range(len(img)):
        """ in the original formula we consider the frist summation result
        as zigma_1 and the second as zigma_2 """
        zigma_1 = 0
        zigma_2 = 0

        if (i in range(4)):
            continue
        for j in range(4):
            zigma_1 = zigma_1+coefficients[j]*img[i-j]
        for j in range(1, 5):
            zigma_2 = zigma_2+coefficients[3+j]*y[i-j]
        y[i] = zigma_1-zigma_2

    return y


# building the anti-causal part of the main recursive Gaussian filter

def anti_causal_filter(image, coefficients):
    y = np.zeros((img.shape[0],img.shape[1]), dtype=float)
    y[4] = coefficients[0]*img[0]
    for i in range(len(img)):
        """ in the original formula we consider the frist summation result
        as zigma_1 and the second as zigma_2 """
        zigma_1 = 0
        zigma_2 = 0

        if (i in range(len(img)-4,len(img))):
            continue
        for j in range(1, 5):
            zigma_1 = zigma_1+coefficients[j+7]*img[i+j]
        for j in range(1, 5):
            zigma_2 = zigma_2+coefficients[3+j]*y[i+j]
        y[i] = zigma_1-zigma_2

    return y
    
    
    

""" combining the causal and anti-causal parts to finally have the filter and
normalizing the output"""

def final_filter (sigma, causal, antiCausal):
    y = np.zeros((img.shape[0],img.shape[1]), dtype=float)
    for i in range(causal.shape[0]):
        for j in range(causal.shape[1]):
            y[i][j] = (causal[i][j] + antiCausal[i][j])/(sigma * np.sqrt(2 * np.pi))
    
    return y

""" experimenting the application of different valuses for parameter (sigma)
with our approach """

for sd in range(1,10):
    standardDeviation = sd
    coeffs = coefficients (standardDeviation)
    causal = causal_filter(img, coeffs)
    anti_causal = anti_causal_filter(img,coeffs)
    final = final_filter(standardDeviation,causal,anti_causal)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', interpolation='nearest') 
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(final, cmap='gray', interpolation='nearest') 
    plt.title('Filtered image')
    plt.xticks([])
    plt.yticks([])
    plt.show()

""" experimenting the application of different valuses for parameter (sigma)
with the built-in function """

from scipy.ndimage.filters import gaussian_filter

print('-------------------------------------')
for sd in range(1,10):
    
    final = gaussian_filter(img,sd)

    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray', interpolation='nearest') 
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(final, cmap='gray', interpolation='nearest') 
    plt.title('Filtered image')
    plt.xticks([])
    plt.yticks([])
    plt.show()

