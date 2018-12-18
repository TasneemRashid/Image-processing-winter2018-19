import cv2
from matplotlib import pyplot as plt
import numpy as np

img1 = cv2.imread('bauckhage.jpg',0)
img2 = cv2.imread('clock.jpg',0)

def gradientImage(img, dx, dy, ksize):
    deriv_filter = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=True) #Normalize the kernel
    return cv2.sepFilter2D(img, -1, deriv_filter[0], deriv_filter[1])            #Get 2D

position = 1 
fig = plt.figure()
fig.suptitle('Image Gradients',fontsize=16)
for i in [3,5,9,15]:
    plt.subplot(2,4,position),plt.imshow(gradientImage(img1,1,0,i),cmap = 'gray'),plt.title('Size: '+ str(i)+ ', X'), plt.xticks([]), plt.yticks([])
    position=position+1  
    plt.subplot(2,4,position),plt.imshow(gradientImage(img1,0,1,i),cmap = 'gray'),plt.title('Size: '+ str(i)+ ', Y'), plt.xticks([]), plt.yticks([])
    position=position+1

def gradientMagnitude(img,i):
    g = np.sqrt(gradientImage(img,1,0,i)**2 + gradientImage(img,0,1,i)**2).astype(np.uint8)
    return g

fig = plt.figure()
fig.suptitle('Gradient Magnitude',fontsize=16)
position = 1
for i in [5,9,15]:
    plt.subplot(2,3,position),plt.imshow(gradientMagnitude(img1,i),cmap = 'gray'),plt.title('Kernel '+str(i)), plt.xticks([]), plt.yticks([])
    position = position+1

for i in [5,9,15]:
    plt.subplot(2,3,position),plt.imshow(gradientMagnitude(img2,i),cmap = 'gray'),plt.title('Kernel '+str(i)), plt.xticks([]), plt.yticks([])
    position = position+1

position = 1 
fig = plt.figure()
fig.suptitle('Other Libraries: Sobel',fontsize=16)
for i in [3,5,9,15]:
    sobelx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=i)             #First order Sobel edge detection
    sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=i)
    plt.subplot(2,4,position),plt.imshow(sobelx,cmap = 'gray'),plt.title('Size: '+ str(i)+ ', X'), plt.xticks([]), plt.yticks([])
    position=position+1  
    plt.subplot(2,4,position),plt.imshow(sobely,cmap = 'gray'),plt.title('Size: '+ str(i)+ ', Y'), plt.xticks([]), plt.yticks([])
    position=position+1
   
position =1
fig = plt.figure()
fig.suptitle('Other Libraries: Laplacian',fontsize=16)    
for i in [3,5,9,15]:
    laplacian = cv2.Laplacian(img1,cv2.CV_64F)                  #2nd order Laplacian edge detection
    plt.subplot(1,4,position),plt.imshow(laplacian,cmap = 'gray'),plt.title('Size: '+str(i)), plt.xticks([]), plt.yticks([])
    position=position + 1 

plt.show()