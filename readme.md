# Evaluating and predicting computing performance of deep neural network models with different backbones on cross-modal computing platforms and inference frameworks
## Dependencies:

 - Python3.6
 - PyTorch
 - Torchvision
 - OpenVINO
 - Ubuntu18.04
 - Google Colab
 - Intel NCS2
 - SPSS
 - OriginLab

## Introduction

We tested nearly 30 typical DNN models and estimated the performance metrics such as the Top-N accuracy, model complexity, memory usage, computational complexity, and inference time. By deeply analyzing the computing performance of DNN models with different backbones on cross-modal computing platforms and inference frameworks can better select network models for deployment on cloud platforms or edge computing devices in practice. Some results are given below, and the specific results can be found in the paper.

## Top-N Accuracy vs. Computational Complexity vs. Model Complexity
There were the top-1 and top-5 accuracy of each DNN using image preprocessing versus floating-point operations required for a single forward pass.The size of each ball corresponded to the model complexity. Different colored balls indicated different models.



## Top-1 Accuracy vs. Model Efficiency

 It represented the top-1 accuracy density of each DNN model, which was utilized to measure the parameter utilization efficiency of the model. Error bar was used to describe the error range of the top-1 accuracy density.

## Inference Time
It recorded the average inference time(in milliseconds) of dozens of DNN models for dealing with a single image after 10 stable runs on the Google Colab cloud computing platform with different batch sizes.


A combination method of cluster analysis and regression analysis from a quantitative perspective was designed to specifically analyze how the inference time varies with batch size in the above Table shown in Figure below.

## Top-1 Accuracy vs. Inference Time
It showed the Top-1 accuracy versus the number of images processed per second (with batch size 1) on the Google Colab computing platform. For a scatter plot of the relationship between Top-1 accuracy and FPS in Intel NCS2, please see the paper.

## Memory Usage vs. Model Complexity
It displayed the relationship between the total memory utilization and the model parameters (i.e. the model complexity) for different DNN models on Colab platform (with batch size 1 ). The straight lines indicated that the more parameters the model has, the more memory it utilizes.

