
import skimage.io as io

b=io.imread('C://DIP//Dataset//4.1.02.tiff')
io.imshow(b)
io.show()

print (b[99,199,:])

