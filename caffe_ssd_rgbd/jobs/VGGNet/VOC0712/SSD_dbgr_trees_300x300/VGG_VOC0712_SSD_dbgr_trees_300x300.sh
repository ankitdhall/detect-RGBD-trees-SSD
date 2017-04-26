cd /home/dhall/code/caffe_ssd/caffe3
./build/tools/caffe train \
--solver="models/VGGNet/VOC0712/SSD_dbgr_trees_300x300/solver.prototxt" \
--snapshot="models/VGGNet/VOC0712/SSD_dbgr_trees_300x300/VGG_VOC0712_SSD_dbgr_trees_300x300_iter_20000.solverstate" \
--gpu 0 2>&1 | tee jobs/VGGNet/VOC0712/SSD_dbgr_trees_300x300/VGG_VOC0712_SSD_dbgr_trees_300x300.log
