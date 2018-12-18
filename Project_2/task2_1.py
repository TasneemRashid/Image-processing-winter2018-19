import numpy as np
import scipy.ndimage as ndImage
from scipy.signal import fftconvolve
from matplotlib import pyplot as plt
import cv2
import timeit, functools
from astropy.convolution import Gaussian2DKernel

myImg = cv2.imread('bauckhage.jpg',0)                           #Read image and convert into float
img = np.float32(myImg)                     
cat_img = cv2.imread('cat.png',0)
cat = np.float32(cat_img)
clock_img = cv2.imread('clock.jpg',0)
clock = np.float32(clock_img)

#task 2.1.1 Naive Implementation
fig=plt.figure()
fig.suptitle('Convolution with FFT Approach',fontsize=16)
plt.subplot(1,2,1),plt.imshow(img,cmap = 'gray')
plt.title("Original Image"), plt.xticks([]),plt.yticks([])      #Plot the orginal Image

#Convolution with FFT Approach
def naiveImplementation(ksize):
    t = 1 - np.abs(np.linspace(-1, 1, ksize))
    kernel = t.reshape(ksize, 1) * t.reshape(1, ksize)
    kernel /= kernel.sum()                                      #Kernel should sum to 1           
    image = fftconvolve(img, kernel, mode='same')
    return image

#Run for 6 different filters
position =1
for i in [3,5,9,15,23,33]:
    plt.subplot(2,3,position),plt.imshow(naiveImplementation(i),cmap = 'gray'),plt.title(str(i)+"x"+str(i)), plt.xticks([]),plt.yticks([])
    position =position+1
    
#task 2.1.1 Scipy Implementation   
def sciPyImplementation(ksize):
    sigma = (ksize-1.) / (2*2.575)
    #Call gaussian_filter with parameters
    ndarray = ndImage.gaussian_filter(img,sigma=sigma,order =0,mode='reflect',cval=0.0,truncate=4.0)
    return ndarray

#Run for 6 different filters
position=1
fig=plt.figure()
fig.suptitle('SciPy Library',fontsize=16)
for i in [3,5,9,15,23,33]:                                      #Array of filter size                                    
    plt.subplot(2,3,position),plt.imshow(sciPyImplementation(i),cmap = 'gray'),plt.title(str(i)+ 'x'+str(i)), plt.xticks([]),plt.yticks([])               
    position=position+1

#task 2.1.1 CV2 Implementation
def cv2GaussianBlur(filter_size):
    blurryImage = cv2.GaussianBlur(img,(filter_size,filter_size),0)    #Call GaussianBlue with filter size
    return blurryImage

#Run for 6 different filters
position=1
fig=plt.figure()
fig.suptitle('OpenCV Library',fontsize=16)
for i in [3,5,9,15,23,33]:
    plt.subplot(2,3,position),plt.imshow(cv2GaussianBlur(i),cmap = 'gray'),plt.title(str(i)+ 'x'+str(i)),plt.xticks([]), plt.yticks([]),plt.imshow(cv2GaussianBlur(i), cmap = 'gray')  
    position=position+1                          

#task 2.1.2 
def seperableFilter(img,sigma):
    sigma = 13.                                                  #Calculate weights g
    msize = int(np.ceil(sigma * 2.575) * 5 + 1)
    x = np.arange(msize)
    g = np.exp(-0.5 * ((x-msize/2) / sigma)**2)
    g /= g.sum()
    
    rowConvolution = ndImage.convolve1d(img,g,axis=1,mode='reflect',cval=0.0)   #axis=1 for column   
    columnConvolution = ndImage.convolve1d(rowConvolution,g,axis=0,mode='reflect',cval=0.0) #axis=0 for column      
    #oneDConvRow=ndImage.gaussian_filter1d(rowConvolution,sigma=sigma,axis=1,order =0,mode='reflect',cval = 0.0,truncate =4.0)
    #t = rowConvolution*oneDConvRow
    #oneDConvColumn=ndImage.gaussian_filter1d(columnConvolution,sigma=sigma,axis =0,order =0,mode ='reflect',cval = 0.0,truncate =4.0)
    #h = oneDConvColumn*columnConvolution
    return rowConvolution,columnConvolution

position = 1
fig=plt.figure()
fig.suptitle('Separability',fontsize=16)
for i in [img, cat]: 
    t,h = seperableFilter(i,13)
    plt.subplot(2,3,position),plt.imshow(i,cmap = 'gray')
    position += 1
    plt.title("Original Image "), plt.xticks([]),plt.yticks([])
    plt.subplot(2,3,position),plt.imshow(t,cmap = 'gray')
    position += 1        
    plt.title("Row Convolution "), plt.xticks([]),plt.yticks([]) 
    plt.subplot(2,3,position),plt.imshow(h,cmap = 'gray')
    position += 1     
    plt.title("Column Convolution"), plt.xticks([]),plt.yticks([]) 

#Task 2.1.3
fft_of_img = np.fft.fft2(img)
fShift_of_img = np.fft.fftshift(fft_of_img)                     #move zero frequency to the center of the spectrum
absValue_of_img = np.log10(abs(fShift_of_img))

fig=plt.figure()
fig.suptitle('Convolution in Space Domain',fontsize=16)
plt.subplot(2,3,1),plt.imshow(img,cmap = 'gray')                #Plot the Image 
plt.title("Original Image"), plt.xticks([]),plt.yticks([])
plt.subplot(2,3,2),plt.imshow(absValue_of_img,cmap = 'gray')    #Plot the Image 
plt.title("FFT"), plt.xticks([]),plt.yticks([])

gauss_kernel = Gaussian2DKernel(15)                              #Create 3x3 Kernel 
sz = (256, 256)                                                 #Image output size macthing with the original image
after_x = (sz[0] - gauss_kernel.shape[0])//2                    #Slice the kernel into 4 pieces
before_x = sz[0] - gauss_kernel.shape[0] - after_x
after_y = (sz[1] - gauss_kernel.shape[1])//2
before_y = sz[1] - gauss_kernel.shape[1] - after_y
gauss_kernel = np.pad(gauss_kernel, ((before_x, after_x), (before_y, after_y)), 'constant')
gauss_kernel = np.fft.ifftshift(gauss_kernel)
fft_of_gauss_kernel = np.fft.ifft2(gauss_kernel)
absValue_of_gauss_kernel = np.abs(fft_of_gauss_kernel)
K = fShift_of_img*fft_of_gauss_kernel
K = np.abs(K)

plt.subplot(2,3,3),plt.imshow(gauss_kernel,cmap = 'gray')                #Plot the Image 
plt.title("Padded Filter"), plt.xticks([]),plt.yticks([])
plt.subplot(2,3,4),plt.imshow(absValue_of_gauss_kernel,cmap = 'gray')    #Plot the Image 
plt.title("FFT Filter"), plt.xticks([]),plt.yticks([])
plt.subplot(2,3,5),plt.imshow(K,cmap = 'gray')                           #Plot the Image 
plt.title("Multiplication FFT"), plt.xticks([]),plt.yticks([])

#Task 2.1.4
start = timeit.default_timer()
x = np.array([3, 5, 9, 15, 23])

#Calculate runtime for OpenCV implementation
arrayOne = []    
for i in [3,5,9,15,23]:  
    t = timeit.Timer(functools.partial(cv2GaussianBlur, i)).repeat(1, 1000)
    t = np.array(t)
    arrayOne = np.append(arrayOne,np.mean(t))

#Calculate runtime for SciPy Implementation
arrayTwo = []
for i in [3,5,9,15,23]:
    t = timeit.Timer(functools.partial(sciPyImplementation, i)).repeat(1, 1000)
    t = np.array(t)
    arrayTwo = np.append(arrayTwo,np.mean(t))

#Calculate runtime for Separable 1D Implementation
arrayThree = []
for i in [3,5,9,15,23]:
    t = timeit.Timer(functools.partial(naiveImplementation, i)).repeat(1, 1000)
    t = np.array(t)
    arrayThree = np.append(arrayThree,np.mean(t))

#Calculate runtime for Separable 1D Implementation
'''
arrayFour = []
for i in [3,5,9,15,23]:
    t = timeit.Timer(functools.partial(seperableFilter,i)).repeat(1, 1000)
    t = np.array(t)
    arrayFour = np.append(arrayFour,np.mean(t))
'''    
plt.figure()
plt.plot(x,arrayOne)
plt.plot(x,arrayTwo)
plt.plot(x,arrayThree)
#plt.plot(x,arrayFour)
plt.legend(('OpenCV', 'SciPy','FFT Convolve'), loc='upper right')
plt.ylabel('Run Times')
plt.xlabel('Filter Size')
plt.title('Runtime of different approaches')
plt.show()
stop = timeit.default_timer()
print('Estimated time elapsed : ', (stop - start)/60,'Mins' )
