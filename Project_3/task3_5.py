import cv2
import numpy as np
from matplotlib import pyplot as plt

src_img = cv2.imread('clock.jpg')                       # read image
rows = src_img.shape[0]                                 # determine number of rows
cols = src_img.shape[1]                                 # determine number of columns
dst_img = cv2.imread('isle.jpg')                        # read image

# display initial images
plt.subplot(141), plt.imshow(src_img, cmap='gray'), plt.title('Image 1')
plt.subplot(142), plt.imshow(dst_img, cmap='gray'), plt.title('Image 2'), plt.axis("off")

points1 = np.float32([[0, 0], [cols - 1, 0], [rows - 1, cols - 1], [0, rows - 1]])          # coordinates of the source image
points2 = np.float32([[215, 56], [365, 10], [364, 296], [218, 258]])         # coordinates of the poster in the destination image

perspective = cv2.getPerspectiveTransform(points1, points2)                  # calculate transformation between source and destination points
warp = cv2.warpPerspective(src_img, perspective, (dst_img.shape[1], dst_img.shape[0]))      # warp source image

cv2.fillConvexPoly(dst_img, points2.astype(int), 0)                          # black out polygonal area in destination image
final_image = dst_img + warp                                                 # add warped source image to destination image

# display images
plt.subplot(143), plt.imshow(warp, cmap='gray'), plt.title('Warp Image'), plt.axis("off")
plt.subplot(144), plt.imshow(final_image, cmap='gray'), plt.title('Augmented Reality'), plt.axis("off")
plt.show()
