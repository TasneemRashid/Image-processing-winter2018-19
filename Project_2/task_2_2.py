import cv2
from matplotlib import pyplot as plt
import scipy.ndimage as ndimg

img1 = cv2.imread('bauckhage.jpg',0)
img2 = cv2.imread('clock.jpg',0)

#Loop through the array of Filter Size
position = 1 
fig = plt.figure()
fig.suptitle('Normalized Image Gradients',fontsize=16)


def gradient(img1, dx, dy, ksize):
    deriv_filter = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=True) 
    return cv2.sepFilter2D(img1, -1, deriv_filter[0], deriv_filter[1])

for i in [3,5,9,15]:
    plt.subplot(2,4,position),plt.imshow(gradient(img1,1,0,i),cmap = 'gray'),plt.title('Size: '+ str(i)+ ', X'), plt.xticks([]), plt.yticks([])
    position=position+1  
    plt.subplot(2,4,position),plt.imshow(gradient(img1,0,1,i),cmap = 'gray'),plt.title('Size: '+ str(i)+ ', Y'), plt.xticks([]), plt.yticks([])
    position=position+1

position = 1 
fig = plt.figure()
fig.suptitle('Sobel',fontsize=16)
for i in [3,5,9,15]:
    sobelx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=i)
    sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=i)
    plt.subplot(2,4,position),plt.imshow(sobelx,cmap = 'gray'),plt.title('Size: '+ str(i)+ ', X'), plt.xticks([]), plt.yticks([])
    position=position+1  
    plt.subplot(2,4,position),plt.imshow(sobely,cmap = 'gray'),plt.title('Size: '+ str(i)+ ', Y'), plt.xticks([]), plt.yticks([])
    position=position+1
   
position =1
fig = plt.figure()
fig.suptitle('Laplacian',fontsize=16)    
for i in [3,5,9,15]:
    laplacian = cv2.Laplacian(img1,cv2.CV_64F) #2nd order Laplacian edge detection
    plt.subplot(2,2,position),plt.imshow(laplacian,cmap = 'gray'),plt.title('Size: '), plt.xticks([]), plt.yticks([])
    position=position + 1  

def gradientFunction(img,mode,sigma):
    g = ndimg.filters.gaussian_gradient_magnitude(img, sigma=sigma,mode=mode, cval=0.0)
    return g

fig = plt.figure()
fig.suptitle('Gradient Magnitude',fontsize=16)
position = 1
for i in [5,10,25]:
    plt.subplot(2,3,position),plt.imshow(gradientFunction(img1,'reflect',i),cmap = 'gray'),plt.title('Sigma '+str(i)), plt.xticks([]), plt.yticks([])
    position = position+1

for i in [5,10,25]:
    plt.subplot(2,3,position),plt.imshow(gradientFunction(img2,'reflect',i),cmap = 'gray'),plt.title('Sigma '+str(i)), plt.xticks([]), plt.yticks([])
    position = position+1
plt.show()