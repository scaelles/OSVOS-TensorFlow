# OSVOS: One-Shot Video Object Segmentation
Check our [project page](http://www.vision.ee.ethz.ch/~cvlsegmentation/osvos) for additional information.
![OSVOS](doc/ims/osvos.png)

OSVOS is a method that tackles the task of semi-supervised video object segmentation. It is based on a fully-convolutional neural network architecture that is able to successively transfer generic semantic information, learned on ImageNet, to the task of foreground segmentation, and finally to learning the appearance of a single annotated object of the test sequence (hence one-shot). Experiments on DAVIS 2016 show that OSVOS is faster than currently available techniques and improves the state of the art by a significant margin (79.8% vs 68.0%).


This TensorFlow code is a posteriori implementation of OSVOS and it does not contain the boundary snapping branch. The results published in the paper were obtained using the Caffe version that can be found at [OSVOS-caffe](https://github.com/kmaninis/OSVOS-caffe).

#### NEW: PyTorch implementation also available: [OSVOS-PyTorch](https://github.com/kmaninis/OSVOS-PyTorch)!

### Installation:
1. Clone the OSVOS-TensorFlow repository
   ```Shell
   git clone https://github.com/scaelles/OSVOS-TensorFlow.git
   ```
2. Install if necessary the required dependencies:
   
   - Python 2.7, Python 3 (thanks to [@xoltar](https://github.com/xoltar))
   - Tensorflow r1.0 or higher (`pip install tensorflow-gpu`) along with standard [dependencies](https://www.tensorflow.org/install/install_linux)
   - Other python dependencies: PIL (Pillow version), numpy, scipy, matplotlib, six
   
3. Download the parent model from [here](https://data.vision.ee.ethz.ch/csergi/share/OSVOS/OSVOS_parent_model.zip) (55 MB) and unzip it under `models/` (It should create a folder named 'OSVOS_parent').

4. All the steps to re-train OSVOS are provided in this repository. In case you would like to test with the pre-trained models, you can download them from  [here](https://data.vision.ee.ethz.ch/csergi/share/OSVOS/OSVOS_pre-trained_models.zip) (2.2GB) and unzip them under `models/` (It should create a folder for every model).

### Demo online training and testing
1. Edit in file `osvos_demo.py` the 'User defined parameters' (eg. gpu_id, train_model, etc).

2. Run `python osvos_demo.py`.

It is possible to work with all sequences of DAVIS 2016 just by creating a soft link (`ln -s /path/to/DAVIS/  DAVIS`) in the root folder of the project.

### Training the parent network (optional)
1. All the training sequences of DAVIS 2016 are required to train the parent model, thus download it from [here](https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip) if you don't have it. 
2. Place the dataset in this repository or create a soft link to it (`ln -s /path/to/DAVIS/ DAVIS`) if you have it somewhere else.
3. Download the VGG 16 model trained on Imagenet from the TF model zoo from [here](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz).
4. Place the vgg_16.ckpt file inside `models/`.
5. Edit the 'User defined parameters' (eg. gpu_id) in file `osvos_parent_demo.py`.
6. Run `python osvos_parent_demo.py`. This step takes 20 hours to train (Titan-X Pascal), and ~15GB for loading data and online data augmentation. Change dataset.py accordingly, to adjust to a less memory-intensive setup.

Have a happy training!

### Citation:
	@Inproceedings{Cae+17,
	  Title          = {One-Shot Video Object Segmentation},
	  Author         = {S. Caelles and K.K. Maninis and J. Pont-Tuset and L. Leal-Taix\'e and D. Cremers and L. {Van Gool}},
	  Booktitle      = {Computer Vision and Pattern Recognition (CVPR)},
	  Year           = {2017}
	}
If you encounter any problems with the code, want to report bugs, etc. please contact me at scaelles[at]vision[dot]ee[dot]ethz[dot]ch.
