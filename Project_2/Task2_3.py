import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter

built_in_average_time = []
costum_average_time = []

org_img = plt.imread('./resources/bauckhage.jpg')
#org_img = plt.imread('./resources/clock.jpg')
img = np.zeros((org_img.shape[0]+8,org_img.shape[1]+8),dtype = float)
for i in range(4,img.shape[0]-4):
    for j in range(4,img.shape[1]-4):
        img[i][j] = org_img[i-4][j-4]


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
def causal_filter_row_based(img, coefficients):
    # initializing y to recursively/ gradually fill it in
    y = np.zeros((img.shape[0],img.shape[1]),dtype = float)

    #row-wise causal
    for row in range(y.shape[0]):
        y[row][3] = coefficients[0]*img[row][3]
        for n in range(4, len(img[row])):
            """ in the original formula we consider the frist summation result
            as zigma_1 and the second as zigma_2 """
            zigma_1 = 0
            zigma_2 = 0

            for m in range(4):
                zigma_1 = zigma_1+coefficients[m]*img[row][n-m]
            for m in range(1, 5):
                zigma_2 = zigma_2+coefficients[3+m]*y[row][n-m]
            y[row][n] = zigma_1-zigma_2

    return y

def causal_filter_col_based(img, coefficients):
    #column-wise causal
    y = np.zeros((img.shape[0],img.shape[1]),dtype = float)
    for col in range(y.shape[1]):
        y[3][col] = coefficients[0]*img[3][col]
        for n in range(4, len(img[col])):
            """ in the original formula we consider the frist summation result
            as zigma_1 and the second as zigma_2 """
            zigma_1 = 0
            zigma_2 = 0

            for m in range(4):
                zigma_1 = zigma_1+coefficients[m]*img[n-m][col]
            for m in range(1, 5):
                zigma_2 = zigma_2+coefficients[3+m]*y[n-m][col]
            y[n][col] = zigma_1-zigma_2
    return y

def causal_filter(img, coefficients):
    """ 1) we use image (as a one-dimensional array), as for x but 3 elements
          should be concatenated to the beginning of the image array to prevent negative
          indexing.
        2) like mentioned in the task we flatten the image array to work on a
            1-d array"""
    y = causal_filter_row_based(img, coefficients)
    plt.imshow(y, cmap='gray', interpolation='nearest')
    plt.show()
    y = causal_filter_col_based(y, coefficients)
    plt.imshow(y, cmap='gray', interpolation='nearest')
    plt.show()
    return y



# building the anti-causal part of the main recursive Gaussian filter
def anti_causal_filter_row_based(img, coefficients):
    #row-wise anti-causal
    y = np.zeros((img.shape[0],img.shape[1]),dtype = float)
    for row in range(y.shape[0]):
        # calculating the last element to initaite the recursive equation
        y[row][len(img)-6] = coefficients[8]*img[row][len(img)-5]
        # current elements depend on next elements! so we fill the list from end to the beginning
        for n in reversed(range(0, len(img[row])-6)):
            """ in the original formula we consider the frist summation result
            as zigma_1 and the second as zigma_2 """
            zigma_1 = 0
            zigma_2 = 0

            for m in range(1, 5):
                zigma_1 = zigma_1+coefficients[m+7]*img[row][n+m]
            for m in range(1, 5):
                zigma_2 = zigma_2+coefficients[3+m]*y[row][n+m]
            y[row][n] = zigma_1-zigma_2
    return y

def anti_causal_filter_col_based(img, coefficients):
    # initializing y to recursively/ gradually fill it in
     y = np.zeros((img.shape[0],img.shape[1]),dtype = float)
     #column-wise anti-causal
     for col in range(y.shape[1]):
         # calculating the last element to initaite the recursive equation
         y[len(img)-6][col] = coefficients[8]*img[len(img)-5][col]
         # current elements depend on next elements! so we fill the list from end to the beginning
         for n in reversed(range(0, len(img[col])-6)):
             """ in the original formula we consider the frist summation result
             as zigma_1 and the second as zigma_2 """
             zigma_1 = 0
             zigma_2 = 0

             for m in range(1, 5):
                 zigma_1 = zigma_1+coefficients[m+7]*img[n+m][col]
             for m in range(1, 5):
                 zigma_2 = zigma_2+coefficients[3+m]*y[n+m][col]
             y[n][col] = zigma_1-zigma_2
     return y

def anti_causal_filter(img, coefficients):

    y = anti_causal_filter_row_based(img, coefficients)
    plt.imshow(y, cmap='gray', interpolation='nearest')
    plt.show()
    y = anti_causal_filter_col_based(y, coefficients)
    plt.imshow(y, cmap='gray', interpolation='nearest')
    plt.show()
    return y
    


""" combining the causal and anti-causal parts to finally have the filter and
normalizing the output"""

def final_filter(sigma, img):
    y = np.zeros((img.shape[0],img.shape[1]),dtype = float)
    z = np.zeros((y.shape[0],y.shape[1]),dtype = float)
    coeffs = coefficients(sigma)

    causal = causal_filter_row_based(img, coeffs)
    antiCausal = anti_causal_filter_row_based(img, coeffs)
    for n in range(0, img.shape[0]):
        for m in range(0, img.shape[1]):
            y[n][m] = (causal[n][m] + antiCausal[n][m])/(sigma * np.sqrt(2 * np.pi))

    causal = causal_filter_col_based(y, coeffs)
    antiCausal = anti_causal_filter_col_based(y, coeffs)
    for n in range(0, y.shape[0]):
        for m in range(0, y.shape[1]):
            z[n][m] = (causal[n][m] + antiCausal[n][m])/(sigma * np.sqrt(2 * np.pi))
    return z


for standardDeviation in range(5,6):

    """ experimenting the application of different valuses for parameter (sigma)
    with our approach """

    start_time = time.time()
    final_custom = final_filter(standardDeviation, img)
    end_time = time.time()
    costum_average_time.append(end_time-start_time)

    """ experimenting the application of different valuses for parameter (sigma)
    with the built-in function """

    start_time = time.time()
    final_builtIn = gaussian_filter(org_img,standardDeviation)
    end_time = time.time()
    built_in_average_time.append(end_time-start_time)

    plt.subplot(1, 2, 1)
    plt.imshow(final_builtIn, cmap='gray', interpolation='nearest')
    plt.title('Filtered image-Built in')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 2, 2)
    plt.imshow(final_custom, cmap='gray', interpolation='nearest')
    plt.title('Filtered image-Our version')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    print()



print('Average time of our implementation: ', sum(costum_average_time)/float(len(costum_average_time)))
print('Average time of built-in function: ', sum(built_in_average_time)/float(len(built_in_average_time)))
