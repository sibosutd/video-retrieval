# Instruction for feature extraction

Below is the guidance to extract features from two-stream networks.

# Usage Guide

## Prerequisites

There are a few dependencies to run the code. The major libraries we use are

- [Our home-brewed Caffe][caffe]
- [dense_flow][df]

The most straightforward method to install these libraries is to run the `build-all.sh` script.

Besides software, GPU(s) are required for optical flow extraction and model training. Our Caffe modification supports highly efficient parallel training. Just throw in as many GPUs as you like and enjoy.

### Get trained models

We provided the trained model weights in Caffe style, consisting of specifications in Protobuf messages, and model weights.
In the codebase we provide the model spec for UCF101 and HMDB51.
The model weights can be downloaded by running the script

```
bash scripts/get_reference_models.sh
```

## Extract Frames and Optical Flow Images

To run the training and testing, we need to decompose the video into frames. Also the temporal stream networks need optical flow or warped optical flow images for input.
 
These can be achieved with the script `scripts/extract_optical_flow.sh`. The script has three arguments
- `SRC_FOLDER` points to the folder where you put the video dataset
- `OUT_FOLDER` points to the root folder where the extracted frames and optical images will be put in
- `NUM_WORKER` specifies the number of GPU to use in parallel for flow extraction, must be larger than 1

The command for running optical flow extraction is as follows

```
bash scripts/extract_optical_flow.sh SRC_FOLDER OUT_FOLDER NUM_WORKER
```

It will take from several hours to several days to extract optical flows for the whole datasets, depending on the number of GPUs.  

## Feature extraction

### Get reference models

To help reproduce the results reported in the paper, we provide reference models trained by us for instant testing. Please use the following command to get the reference models.

```
bash scripts/get_reference_models.sh
```

### Feature extraction

We provide a Python framework to extract the features, we can use the scripts `extract_feature.py`.

For example, run
```
python tools/extract_feature.py flow FRAME_PATH models/ucf101/tsn_bn_inception_flow_deploy.prototxt models/ucf101_split_1_tsn_flow_reference_bn_inception.caffemodel --save_scores SCORE_FILE
```

where `FRAME_PATH` is the path you extracted the frames of videos to and `SCORE_FILE` is the filename to store the extracted features.

To modify the number of steps or layer name of feature extraction, modify the `extract_feature.py` in `tools` folder.

[ucf101]:http://crcv.ucf.edu/data/UCF101.php
[hmdb51]:http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/
[caffe]:https://github.com/yjxiong/caffe
[df]:https://github.com/yjxiong/dense_flow

