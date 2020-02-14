import numpy as np 
import math
from skimage import color, io 
from sklearn.cluster import KMeans
import scipy.misc as smp

width = 0
height = 0

def imgToStr(img):
	global width, height

	rgbImg = io.imread(img)
	labImg = color.rgb2lab(rgbImg)

	height = len(rgbImg)
	width = len(rgbImg[0])

	outList = []
	for i in range(width): #w
		for j in range(height): #h
			outList.append(labImg[i,j,:])
	return outList

def colorLookup(cluster):
	switcher={
		0: [90, 43, 23],
		1: [255, 0, 0],
		2: [0, 255, 0],
		3: [0, 0, 255],
		4: [80, 60, 50]
	}
	return switcher.get(cluster, [0,0,0])

####   begin main   ####

imgList = imgToStr('colors.png')

kmeans = KMeans(n_clusters=5)
kmeans.fit(imgList)

predList = []

for i in range(len(imgList)):
    pred = np.array(imgList[i].astype(float))
    pred = pred.reshape(-1, len(pred))
    predList.append(kmeans.predict(pred))

outImg = np.zeros((width,height,3), dtype=np.uint8) #h,w,[r,g,b]

for i in range(width):	#h
	for j in range(height):#w									
		outImg[i][j] = colorLookup(predList[j + i*width][0])#i*w

img = smp.toimage(outImg)    
img.save('output.png')
img.show()                      