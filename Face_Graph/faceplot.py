import numpy as np
import scipy
import scipy.signal as sig
import matplotlib.pyplot as plt
import imageio


# With mode="L", we force the image to be parsed in the grayscale.
img = imageio.imread("test10.jpg", pilmode="L")

kernel_x = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

G_x = sig.convolve2d(img, kernel_x, mode='same') 
G_y = sig.convolve2d(img, kernel_y, mode='same') 

# Plot them!
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.imshow((G_x + 255) / 2, cmap='gray'); ax1.set_xlabel("SVC")
ax2.imshow((G_y + 255) / 2, cmap='gray'); ax2.set_xlabel("HOG")
fig.suptitle("alogrithm test")
plt.tight_layout()
plt.show()
