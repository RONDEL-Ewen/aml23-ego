# Our code for the [Multimodal Egocentric Action Recognition](https://docs.google.com/document/d/12SDMbO3MgdawRx1C6XhHFtmAwzCXgpGQTElbGx4Qm78/edit?usp=sharing) project - Advanced Machine Learning (AML) course @PoliTo 2023

Originally forked from [here](https://github.com/EgovisionPolito/aml23-ego).

## Getting started

You can play around with the code on your local machine, and use Google Colab for training on GPUs. 
In all the cases it is necessary to have the reduced version of the dataset where you are running the code. For simplicity, we inserted just the necessary frames at [this link](https://drive.google.com/drive/u/1/folders/1dJOtZ07WovP3YSCRAnU0E4gsfqDzpMVo).

### 1. Local

You can work on your local machine directly, the code which needs to be run does not require heavy computations. 
In order to do so a file with all the requirements for the python environment is provided [here](requirements.yaml), it contains even more packages than the strictly needed ones so if you want to install everything step-by-step just be careful to use pytorch 1.12 and torchvision 0.13. 

### 2. Google Colab

You can also run the code on [Google Colab](https://colab.research.google.com/).

As a reference, `colab_runner.ipynb` provides an example of how to run the different steps in Google Colab.

NOTE: you need to stay connected to the Google Colab interface at all times for your python scripts to keep training.
