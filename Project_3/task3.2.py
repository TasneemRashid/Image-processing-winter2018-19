import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

""" image warping is all about manipulating coordinates, not the intensity
values of the images"""


def waves_effect_warping(img, amp, x_freq, y_freq, phi, oblique):

    if oblique:
        padding = amp
    elif 0 < x_freq <= 1/2:
        padding = [(amp, 0), (0, 0)]
    elif -1/2 <= x_freq < 0:
        padding = [(0, amp), (0, 0)]
    else:
        padding = [(amp, amp), (0, 0)]

    img = np.pad(img, padding, 'constant', constant_values=0)

    height, width = img.shape
    print(height, width)
    # getting coordinate matrices (grid) from coordinate vectors
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    # getting stack arrays in sequence vertically

    """we use pad fnction to pad the numpy array (image) with zeros wherever
    needed (as the amplitude will chain the dimension of the image)"""

    """def r4(r, rmax):
        return 1. - np.tanh(r/rmax)

    C = mu-xx  # vectors pointing to mu
    r = np.sqrt(np.sum(C**2, axis=0)) # distances to mu
    d = r4(r, rmax)
    X += C * d"""

    """we need to consider two cases: first the wave is applied on
    both axes which makes it oblique or diagonal, second it is applied only
    only towards the axes"""

    if not oblique:
        yw = amp * np.sin(2 * np.pi * xx * x_freq * 1 / width - phi)
        yt = yy + yw
        X = np.vstack((yt.flatten(), xx.flatten()))
    else:
        xw = amp * np.sin(2 * np.pi * yy * y_freq * 1 / height - phi)
        xt = xx + xw
        yw = amp * np.sin(2 * np.pi * xt * x_freq * 1 / width - phi)
        yt = yy + yw
        X = np.vstack((yt.flatten(), xt.flatten()))

    h = ndimage.map_coordinates(img, X, order=3)
    h = h.reshape(height, width)
    plt.imshow(h, cmap=plt.cm.gray)
    plt.show()


# initializing image
img = plt.imread('./resources/bauckhage.jpg')
waves_effect_warping(img, 80, 1/2, 0, 0, 0)
waves_effect_warping(img, 100, 1, 0, 0, 0)
waves_effect_warping(img, 10, 2, 2, np.pi/2, 1)
waves_effect_warping(img, 10, 4, 6, 0, 1)
waves_effect_warping(img, 15, 9, 1, np.pi, 1)
