import numpy as np
import scipy as sc
import matplotlib.pyplot as plt

# reshaping the list to prepare it for depicting
def reshape_img(list_img):
    return np.reshape(np.asarray(list_img), (256, -1))


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

    pos_a3 = np.exp(-1*(coefs['landa2']+2*coefs['landa1'])/standard_deviation)*(coefs['beta2']*np.sin(coefs['omega2']/standard_deviation)-coefs['alpha2']*np.cos(coefs['omega2']/standard_deviation))+np.exp(-1*(coefs['landa1']+2*coefs['landa2'])/standard_deviation)*(coefs['beta1']*np.sin(coefs['omega1']/standard_deviation)-coefs['alpha1']*np.cos(coefs['omega1']/standard_deviation))

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
    """ 1) we use image (as a one-dimensional array), as for x but 3 elements
          should be concatenated to the beginning of the image array to prevent negative
          indexing.
        2) like mentioned in the task we flatten the image array to work on a
            1-d array"""

    image = image.flatten()
    img = 3*[0]+image.tolist()
    # initializing y to recursively/ gradually fill it in
    y = [0 for item in range(len(img))]
    y[3] = coefficients[0]*img[0]
    for n in range(4, len(img)):
        """ in the original formula we consider the frist summation result
        as zigma_1 and the second as zigma_2 """
        zigma_1 = 0
        zigma_2 = 0

        # if (i in range(4)):
        #     continue
        for m in range(4):
            zigma_1 = zigma_1+coefficients[m]*img[n-m]
        for m in range(1, 5):
            zigma_2 = zigma_2+coefficients[3+m]*y[n-m]
        y[n] = zigma_1-zigma_2

    return y[3:]


# building the anti-causal part of the main recursive Gaussian filter

def anti_causal_filter(image, coefficients):
    image = image.flatten()
    img = image.tolist()+3*[0]

    y = [0 for item in range(len(img))]
    # calculating the last element to initaite the recursive equation
    y[len(img)-5] = coefficients[8]*img[len(img)-4]

    # current elements depend on next elements! so we fill the list from end to the beginning
    for n in reversed(range(0, len(img)-6)):
        """ in the original formula we consider the frist summation result
        as zigma_1 and the second as zigma_2 """
        zigma_1 = 0
        zigma_2 = 0

        # if (n in range(len(img)-4,len(img))):
        #     continue
        for m in range(1, 5):
            zigma_1 = zigma_1+coefficients[m+7]*img[n+m]
        for m in range(1, 5):
            zigma_2 = zigma_2+coefficients[3+m]*y[n+m]
        y[n] = zigma_1-zigma_2

    return y[:len(y)-3]



""" combining the causal and anti-causal parts to finally have the filter and
normalizing the output"""

def final_filter(sigma, causal, antiCausal):
    y = []
    img_length = len(causal)
    for n in range(0, img_length):
            y.append((causal[n] + antiCausal[n])/(sigma * np.sqrt(2 * np.pi)))

    # as of the last part, it's time to normalize after applying the Gaussian distribution
    max_value = max(y)
    min_value = min(y)
    for n in range(0, img_length):
        y[n] = 255*(y[n]-min_value)/(max_value-min_value)

    return y

""" experimenting the application of different valuses for parameter (sigma)
with our approach """

org_img = plt.imread('./resources/bauckhage.jpg')

for standardDeviation in range(1, 10):
    coeffs = coefficients(standardDeviation)
    causal = causal_filter(org_img, coeffs)
    anti_causal = anti_causal_filter(org_img, coeffs)
    final = final_filter(standardDeviation, causal, anti_causal)

    plt.subplot(1, 2, 1)
    plt.imshow(org_img, cmap='gray', interpolation='nearest')
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(reshape_img(final), cmap='gray', interpolation='nearest')
    plt.title('Filtered image-Our version')
    plt.xticks([])
    plt.yticks([])
    plt.show()

""" experimenting the application of different valuses for parameter (sigma)
with the built-in function """

from scipy.ndimage.filters import gaussian_filter

print('-------------------------------------')
for sd in range(1,10):

    final = gaussian_filter(org_img,sd)

    plt.subplot(1, 2, 1)
    plt.imshow(org_img, cmap='gray', interpolation='nearest')
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(final, cmap='gray', interpolation='nearest')
    plt.title('Filtered image-Built in')
    plt.xticks([])
    plt.yticks([])
    plt.show()
