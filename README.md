# OSVOS: One-Shot Video Object Segmentation

OSVOS is a method that tackles the task of semi-supervised video object segmentation. It is based on a fully-convolutional neural network architecture that is able to successively transfer generic semantic information, learned on ImageNet, to the task of foreground segmentation, and finally to learning the appearance of a single annotated object of the test sequence (hence one-shot).

Check our [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/osvos) for additional information.

This TensorFlow code is a posteriori implmentation of OSVOS. The results published in the paper were obtained using the Caffe version that can be found at [OSVOS-caffe](https://github.com/kmaninis/OSVOS-caffe).

Code coming soon!

### Citation:
    @Inproceedings{Caelles2017,
      author = {S. Caelles and K.-K. Maninis and J. Pont-Tuset and L. Leal-Taix\'e and D. Cremers and L. {Van Gool},
      title = {One-Shot Video Object Segmentation},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year = {2017}
    }
