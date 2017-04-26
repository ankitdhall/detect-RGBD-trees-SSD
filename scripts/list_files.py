import os
#import cv2

path = '/home/dhall/JPEGImages_VOC2007/'
dir_list = os.listdir(path)
dir_list.sort()
for filename in dir_list:
	if filename.endswith('.jpg'):
		print path + filename
