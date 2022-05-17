import skimage.segmentation
from matplotlib import pyplot as plt

img2 = plt.imread("test10.jpg")
segment_mask1 = skimage.segmentation.felzenszwalb(img2, scale=100)
segment_mask2 = skimage.segmentation.felzenszwalb(img2, scale=1000)

fig = plt.figure(figsize=(12, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.imshow(segment_mask1); ax1.set_xlabel("SEGMENT SVC")
ax2.imshow(segment_mask2); ax2.set_xlabel("SEGMENT HOG")
fig.suptitle("face graph based image segment")
plt.tight_layout()
plt.show()