

import numpy as np
import os
caffe_root = "/home/dhall/code/caffe_ssd/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')
import time
import matplotlib.pyplot as plt
import caffe
from caffe.proto import caffe_pb2
import cv2
import lmdb

lmdb_file = "/home/dhall/data/VOCdevkit/VOC0712/lmdb/VOC0712_test_lmdb"
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

# datum = caffe_pb2.Datum()
# you can change this to caffe_pb2.AnnotatedDatum().
datum = caffe_pb2.AnnotatedDatum()

for key, value in lmdb_cursor:
    print key
    datum.ParseFromString(value)
    print datum.annotation_group
    #label = datum.label
    #data = caffe.io.datum_to_array(datum)
    #im = data.astype(np.uint8)
    #im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
    #print "label ", label

    #plt.imshow(im)
    #plt.show()

"""
for key, value in lmdb_cursor:
    print "here"
    datum.ParseFromString(value)

    label = datum.label
    print label
    data = caffe.io.datum_to_array(datum)

    #CxHxW to HxWxC in cv2
    image = np.transpose(data, (1,2,0))
    image.shape
    #cv2.imshow('cv2', image)
    #cv2.waitKey(1)
    print('{},{}'.format(key, label))
"""
"""
lmdb_env = lmdb.open(lmdb_file)
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()
datum = caffe_pb2.Datum()

for key, value in lmdb_cursor:
    datum.ParseFromString(value)

    label = datum.label
    print label
    data = caffe.io.datum_to_array(datum)
    print data
    im = data.astype(np.uint8)
    im = np.transpose(im, (2, 1, 0)) # original (dim, col, row)
    print "label ", label

    plt.imshow(im)
    plt.show()
"""

