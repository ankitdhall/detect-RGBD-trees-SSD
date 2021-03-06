import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import pylab

plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Make sure that caffe is on the python path:
caffe_root = '../'  # this file is expected to be in {caffe_root}/examples
import os
os.chdir(caffe_root)
import sys
sys.path.insert(0, 'python')

import caffe
caffe.set_device(0)
caffe.set_mode_gpu()



from google.protobuf import text_format
from caffe.proto import caffe_pb2

# load PASCAL VOC labels
labelmap_file = 'data/dbgr_trees/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames




model_def = 'models/VGGNet/VOC0712/SSD_dbgr_trees_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_dbgr_trees_300x300/VGG_VOC0712_SSD_dbgr_trees_300x300_iter_10000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]

print net.blobs['data'].data.shape
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer = caffe.io.Transformer({'data': (1, 3, 300, 300)})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([84, 87, 84])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

transformer = caffe.io.Transformer({'data_d': (1, 1, 300, 300)})
#transformer.set_transpose('data_d', (2, 0, 1))
transformer.set_mean('data_d', np.array([137])) # mean pixel
transformer.set_raw_scale('data_d', 255)  # the reference model operates on images in [0,255] range instead of [0,1]






# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1,4,image_resize,image_resize)

#image = caffe.io.load_image('examples/images/fish-bike.jpg')

rgb_path = '/home/dhall/data/VOCdevkit/rgbd_trees/RGB/'
depth_path = '/home/dhall/data/VOCdevkit/rgbd_trees/D/'

dir_list_rgb = os.listdir(rgb_path)
dir_list_rgb.sort()

dir_list_d = os.listdir(depth_path)
dir_list_d.sort()

for i in range(len(dir_list_d)):
	filename_rgb = dir_list_rgb[i]
	filename_d = dir_list_d[i]

	if filename_rgb.endswith('.png') and i%20 == 0:
		image_rgb = caffe.io.load_image(rgb_path + filename_rgb)
		image_d = caffe.io.load_image(depth_path + filename_d, color=False)

		print filename_rgb
		#print image_rgb.shape
		#print image_d

		plt.imshow(image_rgb)
		#print image[:,:,0].shape


		transformer = caffe.io.Transformer({'data': (1, 3, 300, 300)})
		transformer.set_transpose('data', (2, 0, 1))
		transformer.set_mean('data', np.array([84, 87, 84])) # mean pixel
		transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
		transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB


		transformed_image_rgb = transformer.preprocess('data', image_rgb)

	
		transformer = caffe.io.Transformer({'data': (1, 1, 300, 300)})
		transformer.set_transpose('data', (2, 0, 1))
		transformer.set_mean('data', np.array([137])) # mean pixel
		transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]

		transformed_image_d = transformer.preprocess('data', image_d)

		#print transformed_image.shape
		#print transformed_image[0,:,:].shape
		#print np.concatenate((transformed_image, np.reshape(transformed_image[0,:,:], (-1, 300, 300))), axis=0)


		net.blobs['data'].data[...] = np.concatenate((transformed_image_d, transformed_image_rgb), axis=0)

		# Forward pass.
		detections = net.forward()['detection_out']

		# Parse the outputs.
		det_label = detections[0,0,:,1]
		det_conf = detections[0,0,:,2]
		det_xmin = detections[0,0,:,3]
		det_ymin = detections[0,0,:,4]
		det_xmax = detections[0,0,:,5]
		det_ymax = detections[0,0,:,6]

		# Get detections with confidence higher than 0.6.
		top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.25]

		top_conf = det_conf[top_indices]
		top_label_indices = det_label[top_indices].tolist()
		top_labels = get_labelname(labelmap, top_label_indices)
		top_xmin = det_xmin[top_indices]
		top_ymin = det_ymin[top_indices]
		top_xmax = det_xmax[top_indices]
		top_ymax = det_ymax[top_indices]






		colors = plt.cm.hsv(np.linspace(0, 1, 22)).tolist()

		plt.imshow(image_rgb)
		currentAxis = plt.gca()

		for i in xrange(top_conf.shape[0]):
			xmin = int(round(top_xmin[i] * image_rgb.shape[1]))
			ymin = int(round(top_ymin[i] * image_rgb.shape[0]))
			xmax = int(round(top_xmax[i] * image_rgb.shape[1]))
			ymax = int(round(top_ymax[i] * image_rgb.shape[0]))
			score = top_conf[i]
			label = int(top_label_indices[i])
			if label!=21:
				continue
			label_name = top_labels[i]
			display_txt = '%s: %.2f'%(label_name, score)
			coords = (xmin, ymin), xmax-xmin+1, ymax-ymin+1
			color = colors[label]
			currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
			currentAxis.text(xmin, ymin, display_txt, bbox={'facecolor':color, 'alpha':0.5})


		#plt.show()
		plt.savefig('detect_' + filename_rgb)
		plt.cla()
		#break
