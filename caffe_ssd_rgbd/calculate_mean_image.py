import cv2
import numpy as np
import os

path= ["/home/dhall/data/VOCdevkit/VOC2007_d/JPEGImages/", "/home/dhall/data/VOCdevkit/VOC2012_d/JPEGImages/", "/home/dhall/data/VOCdevkit/rgbd_trees/JPEGImages/"]
path= ["/home/dhall/data/VOCdevkit/rgbd_trees/JPEGImages/"]

img = None
mean_dbgr = np.array([0,0,0,0], dtype=np.float32)
n = 0
for dir_ in path:
	for filename in os.listdir(dir_):
		if n%1000 == 0 and n!=0:
			print "processed {} files...".format(n)
			print mean_dbgr/n
		if filename.endswith(".png"):
			n+=1
			#print filename
			img = cv2.imread(dir_ + filename, -1)
			#print np.mean(np.mean(img, axis=0), axis=0)
			mean_dbgr += np.mean(np.mean(img, axis=0), axis=0)
			#break
		#break

print "total files:", n
print mean_dbgr
print mean_dbgr/n
