import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.ndimage as ndImage

'''
The concept of this problem was taken from the following paper:
    https://suw.biblos.pk.edu.pl/downloadResource&mId=1626326
'''

'''
Pull warp should have been the best solution to implement the concepts (transformation 
of the pixels). 
The concept of pull warp was to do a reverse calculation of the pixels for destination 
image [as shown in lecture 19]. And then apply interpolation for the pixel intensity.
However, there is one problem for this implementation:
        For the reflection angle, it depends on original image's row value. So for pull
        method, it can't set the value of refAngle.
'''
# pull warp
'''
# reading input image
img = cv2.imread('bauckhage.jpg')

# initialization of output image
img_output = np.zeros(img.shape, dtype=img.dtype)

rows = img.shape[0]  # number of rows in the source image
cols = img.shape[1]  # number of col in the source image

radius = 5 # radius
refAngle = 45 # reflection angle (assumption?)

for y in range(rows):
    for x in range(cols):

        # following formulas were derived from the mentioned paper 
        u = (radius * math.cos(refAngle)) - ((x - (radius * math.cos(refAngle)))/math.cos(2*refAngle))

        v = y - ((radius * math.cos(refAngle) - u) * math.sin(2 * refAngle))
        
        img_output[y,x] =  ndImage(img, v, u) #resample
'''



#push warp
'''
As in the pull warp reflection angle depends on the rows of original image tried push warp.

Problem with this push warp is 
        We are working with sin and cos which leads to floating points.
        Whereas we can't have fractions for pixel point. Temporarily absolute values 
        were taken.
        However, some points are out of the size of the original image for which the 
        output image is black in most parts. 
        This is why we need pull warping. But pull warp has some issues as well.      
'''
img = cv2.imread('bauckhage.jpg', cv2.IMREAD_GRAYSCALE)
rows = img.shape[0]  # number of rows in the source image
cols = img.shape[1]  # number of col in the source image
img_output = np.zeros(img.shape, dtype=img.dtype)

for v in range(rows):
    for u in range(cols):
        radius = 5
        # refAngle = np.arcsin(j / radius)
        refAngle = (v / radius)

        x = abs(int(radius * math.cos(refAngle) + (radius * math.cos(refAngle) - u) * math.cos(2*refAngle)))

        y = abs(int(v + (radius * math.cos(refAngle) - u) * math.sin(2 * refAngle)))

        if x < cols & y < rows:
            print(y,x,u,v)
            img_output[y,x] = img[v, u]

        '''
        else:
            img_output[y,x] = 0
        '''


# image display
fig=plt.figure()
fig.suptitle('task 3.3',fontsize=16)
plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray'),plt.title("Input image"), plt.xticks([]),plt.yticks([])
plt.subplot(1,2,2),plt.imshow(img_output,cmap = 'gray'),plt.title("Output image"), plt.xticks([]),plt.yticks([])
plt.show()



