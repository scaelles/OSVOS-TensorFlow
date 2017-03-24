# OSVOS: One-Shot Video Object Segmentation

![OSVOS](doc/ims/osvos.png)

OSVOS is a method that tackles the task of semi-supervised video object segmentation. It is based on a fully-convolutional neural network architecture that is able to successively transfer generic semantic information, learned on ImageNet, to the task of foreground segmentation, and finally to learning the appearance of a single annotated object of the test sequence (hence one-shot). Experiments on DAVIS show that OSVOS is faster than currently available techniques and improves the state of the art by a significant margin (79.8% vs 68.0%).

Check our [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/osvos) for additional information.

This TensorFlow code is a posteriori implmentation of OSVOS. The results published in the paper were obtained using the Caffe version that can be found at [OSVOS-caffe](https://github.com/kmaninis/OSVOS-caffe).

Code coming soon!

### Citation:
	@Inproceedings{Cae+17,
	  Title          = {One-Shot Video Object Segmentation},
	  Author         = {S. Caelles and K.K. Maninis and J. Pont-Tuset and L. Leal-Taix\'e and D. Cremers and L. {Van Gool}},
	  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
	  Year           = {2017}
	}
