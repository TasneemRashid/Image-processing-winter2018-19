import cv2
import numpy as np
import scipy.ndimage as ndImage
from matplotlib import pyplot as plt

def conversion(image,sigma):

    newImg = image.astype(np.float64)  # ensure the image is Float

    sqrt = np.sqrt(((newImg.shape[0] / 2.0) ** 2.0) + ((newImg.shape[1] / 2.0) ** 2.0))  # square root from the center (entire dimension of the image)

    polarImg = cv2.linearPolar(newImg,(newImg.shape[0]/2, newImg.shape[1]/2), sqrt, cv2.WARP_FILL_OUTLIERS) # polar image conversion

    oneDGaussian = ndImage.gaussian_filter1d(polarImg, sigma=sigma, axis=0, order=0, mode='wrap', cval=0.0,truncate=4.0) # Gaussian 1D filter

    cartesianImg = cv2.linearPolar(oneDGaussian, (newImg.shape[0] / 2, newImg.shape[1] / 2), sqrt, cv2.WARP_INVERSE_MAP) # Back to cartesian system

    return polarImg,oneDGaussian,cartesianImg




def showImage(image,polarImg,oneDGaussian,cartesianImg,pos,sigma):
    fig=plt.figure()
    fig.suptitle('Sigma '+str(sigma),fontsize=16)
    plt.subplot(pos,4,1),plt.imshow(image,cmap = 'gray'),plt.title("x, y image"), plt.xticks([]),plt.yticks([])
    plt.subplot(pos,4,2),plt.imshow(polarImg.astype(np.uint8),cmap = 'gray'),plt.title("r, φ image"), plt.xticks([]),plt.yticks([])
    plt.subplot(pos,4,3),plt.imshow(oneDGaussian.astype(np.uint8),cmap = 'gray'),plt.title("φ-axis blurred"), plt.xticks([]),plt.yticks([])
    plt.subplot(pos,4,4),plt.imshow(cartesianImg.astype(np.uint8),cmap = 'gray'),plt.title("new x, y image"), plt.xticks([]),plt.yticks([])



image = cv2.imread("clock.jpg")  # read the image
a,b,c = conversion(image,9)
showImage(image,a,b,c,1,9)
a,b,c = conversion(image,13)
showImage(image,a,b,c,2,13)
plt.show()
