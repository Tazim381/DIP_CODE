import numpy as np
import skimage.io as io
import scipy.ndimage as ndi

img01= io.imread('C:\\Users\\Lab-2\\Desktop\\Dataset\\7.2.01.tiff')
io.imshow(img01)
io.show()

f=np.array([[1,4,1],[4,-20,4],[1,4,1]])

img02= ndi.convolve(img01, f, mode='constant')

io.imshow(img02)
io.show()
