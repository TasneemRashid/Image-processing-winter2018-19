import numpy as np
import cv2
import sys
import math


def gaussian(x, sigma):  
    return (1.0 / (2 * math.pi * (sigma ** 2))) * math.exp(- (x ** 2) / (2 * sigma ** 2))        #Calculate the Gaussian function


def euclidean_distance(x, y, i, j):                      
    return np.sqrt((x-i)**2 + (y-j)**2)                                                          #Calculate the Euclidean distance


# Calculate the intensity for each of the pixel from the neighbouring pixels
def filter_pixel(source_image, filtered_image, x , y, diameter, rho, sigma):
    length = diameter/2
    filter_sum = 0
    norm_factor = 0
    for i in xrange(diameter):
        for j in xrange(diameter):                                                              #determine the neighboring pixels of the current pixel 
            neighbour_x = x - (length - i)                                                      #based on diameter and length
            neighbour_y = y - (length - j)
            if neighbour_x >= len(source_image):                                                 
                neighbour_x -= len(source_image)
            if neighbour_y >= len(source_image[0]):
                neighbour_y -= len(source_image[0])
            gauss_intensity = gaussian(source_image[neighbour_x][neighbour_y] - source_image[x][y], rho)   #calculate the gaussian for the difference in intensities
            gauss_euclidean = gaussian(euclidean_distance(neighbour_x, neighbour_y, x, y), sigma)          #calculate the gaussian for the euclidean distance
            weight = gauss_intensity * gauss_euclidean                                                     #multiply the gaussians 
            filter_sum += source_image[neighbour_x][neighbour_y] * weight                                  #add the product of gaussians with the intensity of neighbouring pixel
            norm_factor += weight                                                                          #add the calculated weights
    filter_sum = filter_sum / norm_factor                                                                  #normalize the filtered sum
    filtered_image[x][y] = int(round(filter_sum))                                                          #get the filtered intensity for each pixel 


# function to loop over every pixel of the image
def bilateral_filtering(source_image, filter_diameter, rho, sigma):
    filtered_image = np.zeros(source_image.shape)
    for i in range(len(source_image)):
        for j in range(len(source_image[0])):
            filter_pixel(source_image, filtered_image, i, j, filter_diameter, rho, sigma)       #for each of the pixels, calculate the filtered pixel intensity based on neighbouring pixels
    return filtered_image



source_image = cv2.imread('./resources/bauckhage.jpg', 0)                                        #read the image
filtered_image_OpenCV = cv2.bilateralFilter(source_image, 15, 100.0, 10.0)           #filter the image with CV2
cv2.imwrite("filtered_image_opencv.jpg", filtered_image_OpenCV)                      #Get the filtered image
filtered_image_custom = bilateral_filtering(source_image,15, 100.0, 10.0)            #filter the image with custom code
cv2.imwrite("filtered_image_custom.jpg", filtered_image_custom)                      #Get the new filtered image

