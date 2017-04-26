cd /home/dhall/code/caffe_ssd/caffe
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_RGBD_test_300x300/solver.prototxt" \
--snapshot="models/VGGNet/VOC0712/SSD_RGBD_test_300x300/VGG_VOC0712_SSD_RGBD_test_300x300_iter_65497.solverstate" \
--gpu 3,2,1 2>&1 | tee jobs/VGGNet/VOC0712/SSD_RGBD_test_300x300/VGG_VOC0712_SSD_RGBD_test_300x300.log
