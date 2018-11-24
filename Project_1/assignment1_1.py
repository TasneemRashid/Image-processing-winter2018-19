import matplotlib.pyplot as plt
import numpy as np
from skimage import io

#reading the image
clock = io.imread('./resources/clock.jpg')

#plotting the image
plt.subplot(1, 3, 1)
plt.imshow(clock, cmap='gray', interpolation='nearest') 
plt.title('Original image')
plt.xticks([])
plt.yticks([])

####################################################################
# Function to convert the Image (origin: center)
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

####################################################################
# Same function with user-defined origin
####################################################################
def img_convert_origin(img,rmin,rmax,x,y):
    newImg = np.copy(img)
    for i in range(clock.shape[0]):
        for j in range(clock.shape[1]):
            a = np.array((i,j))
            b = np.array(((clock.shape[0]/2) - y,(clock.shape[1]/2 + x)))
            if rmin <= np.linalg.norm(a-b) <= rmax:
                newImg[i][j] = 0
    return newImg

#building the converted version (origin: center)
converted_image = img_converter(clock,30,77)

#plotting the converted image
plt.subplot(1, 3, 2)
plt.imshow(converted_image, cmap='gray', interpolation='nearest') 
plt.title('Converted image')
plt.xticks([])
plt.yticks([])

#building the converted version with user-defined origin
converted_image_origin = img_convert_origin(clock,30,77,-30,60)

#plotting the converted image
plt.subplot(1, 3, 3)
plt.imshow(converted_image_origin, cmap='gray', interpolation='nearest') 
plt.title('User-defined origin')
plt.xticks([])
plt.yticks([])

#showing all images
plt.show()