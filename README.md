# Learning Single-Image Depth from Videos using Quality Assessment Networks

Code for reproducing the results in the following paper:

    Learning Single-Image Depth from Videos using Quality Assessment Networks
    Weifeng Chen, Shengyi Qian, Jia Deng
    Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

Please check the [project site](http://www-personal.umich.edu/~wfchen/youtube3d/) for more details.

## Example outputs on the Depth in the Wild (DIW) test set

![qual_outputs](http://www-personal.umich.edu/~wfchen/youtube3d/qual_results_DIW_ResNet.jpg)


## Setup

1. The code is written in `python 2.7.13`, using `pytorch 0.2.0_4`. Please make sure that you install the correct pytorch version as later versions may cause the code to break.

2. Clone this repo.

```bash
git clone git@github.com:princeton-vl/YouTube3D.git
```

3. Download [data_model.tar.gz](https://drive.google.com/open?id=1UM58PEwq3XXZv-cURk42RzEaTcuWF7cV) into path `YouTube3D`, then untar:

```bash
cd YouTube3D
tar -xzvf data_model.tar.gz
```

4. Download and unpack the images from [Depth in the Wild dataset](http://www-personal.umich.edu/~wfchen/depth-in-the-wild/). Edit `DIW_test.csv` under `YouTube3D/data` so that all the image paths are absolute paths.


## Evaluating the pretrained models

To evaluate the pre-trained model `EncDecResNet` trained on `ImageNet + ReDWeb + DIW + YouTube3D` on the DIW dataset, run the following command:

```bash
cd YouTube3D/src 
python test.py -t DIW_test.csv -model exp/YTmixReD_dadlr1e-4_DIW_ReDWebNet_1e-6_bs4/models/model_iter_753000.bin
```

In case you want to get the qualitative outputs, append a `-vis` flag and the qualitative outputs will be in the folder `visualize`:

```bash
mkdir visualize
python test.py -t DIW_test.csv -model exp/YTmixReD_dadlr1e-4_DIW_ReDWebNet_1e-6_bs4/models/model_iter_753000.bin -vis
```

## Contact
Please send any questions or comments to Weifeng Chen at wfchen@umich.edu.
