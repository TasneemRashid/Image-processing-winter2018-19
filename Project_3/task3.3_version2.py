import math
from cv2 import *
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread("bauckhage.jpg")                                       # read the image
(rows, cols) = (img.shape[0], img.shape[1])                             # number of rows and columns in the source image
r = 0                                                                   # define radius
c = rows // 2                                                           # creates offset
warp = np.zeros([rows, cols, 3], dtype=np.uint8)                        # creates the destination image with the dimension of the source image


def convert(R, b):
    return math.trunc(c * (1 - b/math.pi)), 2*math.trunc(c - R) - 1     # returns indices for the image


for i in range(0, cols):
    for j in range(0, rows):
        b = math.atan2(j - c, i - c)                                    # calculate angle
        R = math.sqrt((j - c)**2 + (i - c)**2)                          # calculate radius from center
        if r <= R <= c:                                                 # check if R is within range
            q, p = convert(R, b)
            warp[j, i] = img[p, q]                                      # assign appropriate pixels of the final image based on the original image


warp = blur(warp, (2, 2))
# display images
plt.subplot(1, 2, 1), plt.imshow(img, cmap='gray'), plt.title("Input image"), plt.axis("off")
plt.subplot(1, 2, 2), plt.imshow(warp, cmap='gray'), plt.title("Output image"), plt.axis("off")
plt.show()
