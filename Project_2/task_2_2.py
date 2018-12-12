import cv2
from matplotlib import pyplot as plt
import scipy.ndimage as ndimg

img1 = cv2.imread('bauckhage.jpg',0)
img2 = cv2.imread('clock.jpg',0)

#Loop through the array of Filter Size
position = 1 
# Create a figure
plt.figure()
for i in [3,5,9,15]:
    #laplacian = cv2.Laplacian(img1,cv2.CV_64F)
    sobelx = cv2.Sobel(img1,cv2.CV_64F,1,0,ksize=i)
    sobely = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=i)
    plt.subplot(2,4,position),plt.imshow(sobelx,cmap = 'gray')
    plt.title('Size: '+ str(i)+ ', X'), plt.xticks([]), plt.yticks([])
    position=position+1  
    plt.subplot(2,4,position),plt.imshow(sobely,cmap = 'gray')
    plt.title('Size: '+ str(i)+ ', Y'), plt.xticks([]), plt.yticks([])
    position=position+1     

def gradientFunction(img,mode,sigma):
    g = ndimg.filters.gaussian_gradient_magnitude(img, sigma=sigma,mode=mode, cval=0.0)
    return g

plt.figure()
plt.subplot(2,3,1),plt.imshow(gradientFunction(img1,'reflect',5),cmap = 'gray')
plt.title('Sigma 5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,2),plt.imshow(gradientFunction(img1,'reflect',10),cmap = 'gray')
plt.title('Sigma 10'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,3),plt.imshow(gradientFunction(img1,'reflect',25),cmap = 'gray')
plt.title('Sigma 25'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,4),plt.imshow(gradientFunction(img2,'reflect',5),cmap = 'gray')
plt.title('Sigma 5'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,5),plt.imshow(gradientFunction(img2,'reflect',10),cmap = 'gray')
plt.title('Sigma 10'), plt.xticks([]), plt.yticks([])
plt.subplot(2,3,6),plt.imshow(gradientFunction(img2,'reflect',25),cmap = 'gray')
plt.title('Sigma 25'), plt.xticks([]), plt.yticks([])

plt.show()