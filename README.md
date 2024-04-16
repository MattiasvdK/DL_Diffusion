# Deep Learning Conditional Diffusion Model

## Overview

During this project we will train variations of the U-Net architecture on the image dataset described above. In the training scheme of SD the images are given a noise, iteratively until the image is practically random noise. The model's task is then to predict this noise to find the original image again using the caption of the image. This is also done iteratively, meaning each image provides X training samples, where X is the total amount of noising steps for the images. Since the noise is known the loss is then the mean squared error over the noise. Thus the squared difference between the predicted noise and the ground truth.


## Models
The model are largely based from [U-Net](https://arxiv.org/pdf/1505.04597.pdf) as Diffusion Model.


## Data
These model run from 2014 Version of [COCO-Dataset](https://cocodataset.org/#home)

## Requirements
Run `pip install -r requirements.txt` for the file `requirements.txt` included on the repo.

## Preprocessing and Training

To start training, write to the CLI:

```python3 src/main.py```


## Model Inference and Evaluation

TBD