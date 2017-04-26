cd /home/dhall/code/caffe_ssd/caffe
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_300x300_score/solver.prototxt" \
--weights="models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_30000.caffemodel" \
--gpu 0 2>&1 | tee jobs/VGGNet/VOC0712/SSD_300x300_score/VGG_VOC0712_SSD_300x300_test30000.log
