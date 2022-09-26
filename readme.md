<div align="center">
 
  ![image](https://user-images.githubusercontent.com/101705236/181425227-d0594cb6-81be-489f-84d0-04b190bb742e.png)

</div>

# Evaluating and predicting computing performance of deep neural network models with different backbones on cross-modal computing platforms and inference frameworks

<div align="center">

<b><font size="5">Google Colab</font></b>
    <sup>
      <a href="https://colab.research.google.com/">
        <i><font size="4">HOT</font></i>
      </a>
    </sup>
    &nbsp;&nbsp;&nbsp;&nbsp;
    <b><font size="5">Intel NCS2</font></b>
    <sup>
      <a href="https://www.intel.com">
        <i><font size="4">TRY IT OUT</font></i>
      </a>
    </sup>

  ![](https://img.shields.io/badge/python-3.8-blue)
  [![](https://img.shields.io/badge/pytorch-1.9.0-blue)](https://pytorch.org/)
  [![](https://img.shields.io/badge/torchvision-0.10.0-orange)](https://pypi.org/project/torchvision/)
  ![](https://img.shields.io/badge/ubuntu-18.04-orange)
  [![](https://img.shields.io/badge/originlab-2021-brightgreen)](https://www.originlab.com/)
  [![](https://img.shields.io/badge/spss-25-brightgreen)](https://www.ibm.com/products/spss-statistics)

  [🛠️Installation Dependencies](#Dependencies) |
  [🎤Introduction](#Introduction) |
 
  [👀Top-N Accuracy vs. Computational Complexity vs. Model Complexity](https://github.com/fancywang0704/Models-Comparison) |
  
  [🌊Top-1 Accuracy vs. Model Efficiency](https://github.com/fancywang0704/Models-Comparison) |
  [🚀Inference Time](https://github.com/fancywang0704/Models-Comparison) |
  
  [🤔Top-1 Accuracy vs. Inference Time](https://github.com/fancywang0704/Models-Comparison) |
 
  [🔥Memory Usage vs. Model Complexity](https://github.com/fancywang0704/Models-Comparison)
  
  
</div>

## Dependencies:

 - Python 3.8
 - [PyTorch](https://pytorch.org/) 1.9.0
 - [Torchvision](https://pypi.org/project/torchvision/) 0.10.0
 - [OpenVINO](https://docs.openvino.ai/)
 - Ubuntu 18.04
 - [Google Colab](https://colab.research.google.com/)
 - Intel NCS2
 - [OriginLab](https://www.originlab.com/)
 - [SPSS](https://www.ibm.com/products/spss-statistics)

## Introduction

We tested nearly 30 typical DNN models and estimated the performance metrics such as the Top-N accuracy, model complexity, memory usage, computational complexity, and inference time. By deeply analyzing the computing performance of DNN models with different backbones on cross-modal computing platforms and inference frameworks can better select network models for deployment on cloud platforms or edge computing devices in practice. Some results are given below, and the specific results can be found in the paper.

## Top-N Accuracy vs. Computational Complexity vs. Model Complexity
There were the top-1 and top-5 accuracy of each DNN using image preprocessing versus floating-point operations required for a single forward pass.The size of each ball corresponded to the model complexity. Different colored balls indicated different models.
![图片](https://user-images.githubusercontent.com/101705236/173185572-eeca4cd0-e525-4bae-8382-c9bac81b5956.png)
![图片](https://user-images.githubusercontent.com/101705236/173185583-3fdec737-b3c6-44d4-8db8-e2e2a3248847.png)

## Top-1 Accuracy vs. Model Efficiency
 It represented the top-1 accuracy density of each DNN model, which was utilized to measure the parameter utilization efficiency of the model. Error bar was used to describe the error range of the top-1 accuracy density.
![图片](https://user-images.githubusercontent.com/101705236/173182662-37dec8a7-2cf2-4688-aca0-edcea0dd2f0c.png)


## Inference Time
It recorded the average inference time(in milliseconds) of dozens of DNN models for dealing with a single image after 10 stable runs on the Google Colab cloud computing platform with different batch sizes.

<div align="center">

  ![图片](https://user-images.githubusercontent.com/101705236/173182290-15ea961b-79fe-41e8-a944-2487c13ca6f1.png)

</div>

A combination method of cluster analysis and regression analysis from a quantitative perspective was designed to specifically analyze how the inference time varies with batch size in the above Table shown in Figure below.
![图片](https://user-images.githubusercontent.com/101705236/173182345-7e4e475b-7922-422b-850b-36cafb435ce2.png)

## Top-1 Accuracy vs. Inference Time
It showed the top-1 accuracy versus the number of images processed per second (with batch size 1) on the Google Colab computing platform. For a scatter plot of the relationship between top-1 accuracy and FPS in Intel NCS2, please see the paper.
![图片](https://user-images.githubusercontent.com/101705236/173182463-02d6c3bf-392e-462e-b01c-32992aae2af7.png)

## Memory Usage vs. Model Complexity
It displayed the relationship between the total memory utilization and the model parameters (i.e. the model complexity) for different DNN models on Colab platform (with batch size 1 ). The straight lines indicated that the more parameters the model has, the more memory it utilizes.
![图片](https://user-images.githubusercontent.com/101705236/173182488-4d212d55-03ef-4b10-88f4-e86de6c0bc16.png)


## Citation
If you use our code, please consider cite the following:
- X. Wang, F. Zhao, P. Lin, and Y. Chen, "Evaluating computing performance of deep neural network models with different backbones on IoT-based edge and cloud platforms," Internet of Things, vol. 20, p. 100609, 2022/11/01/ 2022.

      
@article{WANG2022100609,
author = {Xiaoxuan Wang and Feiyu Zhao and Ping Lin and Yongming Chen},
title = {Evaluating computing performance of deep neural network models with different backbones on IoT-based edge and cloud platforms},
journal = {Internet of Things},
volume = {20},
pages = {100609},
year = {2022},
issn = {2542-6605},
doi = {https://doi.org/10.1016/j.iot.2022.100609},
}
     
