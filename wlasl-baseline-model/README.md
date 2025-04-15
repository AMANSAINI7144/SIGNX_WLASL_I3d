# Detection of American Sign Language (ASL) Gloass from Videos

## Introduction

Requirements : 

* `numpy`
* `torch`
* `torchvision`
* `opencv-contrib-python`
* `matplotlib`

## Framework Summary

We are extending the work done by Dongxu Li in their work at WACV 2020 titled ["Word-Level Deep Sign Language Recognition from Video: A New Large-Scale Dataset
and Methods Comparison"](https://dxli94.github.io/WLASL/). This repo is a copy of Dongxu Li's training tools, fixing some runtime bugs as well as providing the
weights to the network once training is complete.  The original source code for training their network [can be found here](https://drive.google.com/file/d/1vktQxvRHNS9psOQVKx5-dsERlmiYFRXC/view).
The code does not contain the final weights for gloss recognition, so we took it upon ourselves to train the network.  We have trained the 2000 gloss model, where
the 2000 most frequently used glosses found in the dataset. Please note that the trained model is only CNN based.  The paper mentioned that the source code also introduces
a CNN-RNN hybrid which was not available in the original source code for training their models, even though the author proported to make that available.

## Setup - Running the trained model

1. Once you download the videos, clone this repo locally then place the videos inside the `videos` directory and ensure all of the videos are at their top most
level in the directory. That is, copy the videos over ensuring they are all there with no further subdirectories. The directory should look contain files
like `videos/00295.mp4`, `videos/00333.mp4`, etc.

3. Access the `test.ipynb` notebook file which will load in the pretrained weights, set up the model and perform inference on the test dataset.  The decomposition
of what is the training dataset and test dataset is found in the JSON file in the `preprocess` directory: `nslt_2000.json`.

4.  The top-1, top-5 and top-10 accuracy for the test dataset is reported in the notebook which is in alignment with what is reported in Li's paper.

## Summary of changes made to original source

* `nslt_dataset.py`: The video reading function has been made more robust so that if a video is invalid, we exit gracefully
* `nslt_dataset.py`: The `torch.utils.data.Dataset` that constructs the paths for the train and test videos is now saved and pickled so that this does not need to be
built again when performing the train task. This is very computationally intensive.
* There have been debug statements scattered throughout the source to provide more context and progress on where the training and validation of the test dataset is at.
* There were several PyTorch statements that have been deprecated due to the recent versions of PyTorch.  They have been replaced with their accepted equivalents.

