import matplotlib.pyplot as plt
import numpy as np
from skimage import io

#reading the image
clock = io.imread('./resources/clock.jpg')

#plotting the image
plt.subplot(1, 2, 1)
plt.imshow(clock, cmap='gray', interpolation='nearest') 
plt.title('Original image')
plt.xticks([])
plt.yticks([])

####################################################################
# Function to convert the Image
####################################################################
def img_converter(img,rmin,rmax):
    newImg = np.copy(img)
    for i in range(clock.shape[0]):
        for j in range(clock.shape[1]):
            a = np.array((i,j))
            b = np.array((clock.shape[0]/2,clock.shape[1]/2))
            if rmin <= np.linalg.norm(a-b) <= rmax:
                newImg[i][j] = 0
    return newImg

#building the converted version
converted_image = img_converter(clock,30,77)

#plotting the converted image
plt.subplot(1, 2, 2)
plt.imshow(converted_image, cmap='gray', interpolation='nearest') 
plt.title('Converted image')
plt.xticks([])
plt.yticks([])

#showing all images
plt.show()