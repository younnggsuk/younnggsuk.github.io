---
title: "Multi-scale Context Aggregation by Dilated Convolutions"
key: 20210304
sidebar:
  nav: papers-ko
tags: 논문 ML/DL Semantic&nbspSegmentation
---

<p align="center">
<img src="/assets/multi-scale_context_aggregation_by_dilated_convolutions/dilatednet_title.png" alt="dilatednet">
</p>

이 글은 <a href="https://arxiv.org/abs/1511.07122" target="_blank" rel="noopener noreferrer">Multi-scale Context Aggregation by Dilated Convolutions</a> 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

# Introduction

- **Fully Convolutional Network(FCN)는 image classification을 위한 CNN architecture를 dense prediction으로 repurpose** 하였는데, 여기서 우리는 다음의 몇가지를 생각해볼 수 있다.
  - Dense prediction으로 repurpose하면서 성능이 저하되지는 않았는지?
  - Dense prediction만을 위해 설계된 모듈은 성능을 더 향상시킬 수 있는지?
- 또한, image classification을 위한 네트워크와 dense prediction을 위한 네트워크를 비교해보면, 다음과 같다.
  - Image classification
    - Pooling 및 subsampling을 통해 multi-scale contextual information을 통합
    - 줄어든 resolution에서 global prediction 수행
  - Dense prediction
    - Pooling 및 subsampling을 통해 multi-scale contextual information을 통합
    - **줄어든 resolution을 다시 복원하기 위해 추가로 deconvolution등의 연산을 수행**
    - Full-resolution의 prediction 수행
  - 즉, **dense prediction을 위해서도 multi-scale contextual information이 필요**한데, 이를 추출하는 과정에서 **resolution이 줄어든다는 것이 문제**이다.
- 따라서, 본 논문에서는 다음의 2가지 convolutional network module을 제안한다.
  - **Front-end module**
    - **Dilated convolution**을 적용하여 **Image classification network를 dense prediction을 위한 network로 재설계**
    - Backbone **network 자체**에서 semantic segmentation **성능을 향상**시킴
  - **Context module**
    - **Dilated convolution**을 사용해 **resolution의 손실 없이 multi-scale contextual information을 통합**
    - **기존의 semantic segmentation 네트워크**에 연결해 **추가로 성능을 향상**시킴

# Dilated Convolutions

<img src="/assets/multi-scale_context_aggregation_by_dilated_convolutions/dilated_conv.gif" width="300px"/>
<figcaption><i><a href="https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d" target="_blank" rel="noopener noreferrer">그림 출처 : An Introduction to different Types of Convolutions in Deep Learning</a></i></figcaption>

- **Dilated convolution**은 위 그림과 같이 **kernel 사이사이에 공간을 두고 수행하는 convolution**을 의미한다.
  - kernel 사이의 공간을 조절하는 파라미터를 dilation factor라고 하며 $l$로 나타냄
  - $l$에 따라 $l$-dilated convolution라고 부르며, 위 그림의  경우 $2$-dilated convolution에 해당
  - $1$-dilated convolution은 일반적인 convolution과 동일

![fig_1](/assets/multi-scale_context_aggregation_by_dilated_convolutions/fig_1.png){:width="700px" style="border:1px solid black"}

- *Figure 1*은 dilation factor가 각각 1, 2, 4인 경우의 receptive field를 나타낸 것이며, 빨간색 점이 있는 위치에서만 필터의 파라미터가 존재하는 것이라고 생각하면 된다.
  - *(a)* - receptive field 3$\times$3 
    - input : 3$\times$3, output: 3$\times$3
  - *(b)* - receptive field 7$\times$7
    - input : 7$\times$7, output: 3$\times$3
  - *(c)* - receptive field 15$\times$15
    - input : 15$\times$15, output: 3$\times$3
- 따라서, dilated convolution은 parameter 수는 동일하지만, **receptive field를 크게 가질 수 있다는 장점**이 있다.

# Context Module

- **Context module**은 **multi-scale contextual information을 통합**하여 dense prediction architecture의 성능을 향상시키기 위해 설계되었다.
  - Input과 output의 크기가 동일하므로, 다른 dense prediction architecture들과 연결하여 사용될 수 있음

![table_1](/assets/multi-scale_context_aggregation_by_dilated_convolutions/table_1.png){:width="700px" style="border:1px solid black"}

- 본 논문에서 제안하는 **context module**은 다음의 2가지가 있다. (*Table 1*)
  - **Basic**
    - **모든 convolution layer는 output feature map의 channel이 $C$로 동일**
    - 7번째 layer까지는 각각 1, 1, 2, 4, 8, 16, 1의 dilated factor를 가지는 $3 \times 3$ convolution를 사용
      - 각 convolution 연산 후에는 activation으로 ReLU를 사용 (*Table 1*의 truncation)
    - 마지막 layer는 $1 \times 1$ convolution을 사용
    - Front-end module로부터 $64 \times 64$ 크기의 입력을 받으므로, 6번째 layer 이후부터는 receptive field의 exponential expansion을 수행하지 않음(dilation factor를 더 크게 하지 않았다는 의미)
    - *Table 1*의 Receptive field는 원본 이미지에 대한 값을 의미
      - Ex) 2번째 layer의 경우, 1번째 layer의 output에 대한 receptive field가 $3 \times 3$이므로, 원본 이미지에 대한 receptive field는 $5\times5$
  - **Large**
    - Basic에서 **layer가 깊어질수록 output feature map의 channel을 증가**시킨 형태
      - 7번째 layer까지는 각각 $2C, 2C, 4C, 8C, 16C, 32C, 32C$
      - 마지막 layer는 $C$

- Context module의 initialization에는 <a href="https://arxiv.org/abs/1504.00941" target="_blank" rel="noopener noreferrer"> Le et al.의 논문</a>에서 나온 방법(identity initialization 형태)을 사용하였다. 
  - Xavier와 같은 random initialization 방법이 효과적이지 않았음

# Front-end Module

- **Front-end module은 context module의 앞단에 연결되는 일종의 back-bone network이다.**
  - RGB 이미지를 입력받아서 $C=21$ channel의 feature map을 출력 (context module의 입력이 됨)
- **FCN에서는 VGG-16을 그대로 사용**한 반면, **front-end module에서는 dense prediction을 위한 네트워크로 다음과 같이 수정**하였다.
  - Conv block 4까지는 동일
  - Conv block 4, 5의 **pooling(pool 4, pool 5)을 제거하여 output size가 줄어들지 않게 함**
    - 기존 VGG-16에서 conv block 5의 output size : $16 \times 16$
    - pooling을 제거한 VGG-16에서 conv block 5의 output size : $64 \times 64$
  - Conv block 5의 convolution(conv5_1, conv5_2, conv5_3)을 모두 $2$-dilated convolution으로 변경
  - Conv block 5 뒤의 첫 fully connected layer(fc_1)을 $4$-dilated convolution으로 변경
- 위와 같이 **네트워크를 재설계함으로써, front-end module이 얻게된 이점**은 다음과 같다.
  - Classification을 위한 VGG-16과 동일하게 **pre-trained weight로 초기화 가능**
  - Pooling을 제거하여 high resolution으로 feature map을 출력할 수 있게 됨 (**resolution 손실 감소**)

# Experiments

- 본 논문의 **Front-end module과 context module은 다음의 순서로 학습**하였다. (joint training이 큰 효과가 없었음)
  1. RGB image를 입력으로 front-end module을 학습
  2. 학습된 front-end module의 output feature map을 입력으로 context module을 학습
- 두 모듈의 학습에 대한 자세한 내용은 다음과 같다.
  - Front-end module
    - 학습은 다음의 2가지 stage로 나누어 수행
      - 1st stage
        - PASCAL VOC-2012와 Microsoft COCO의 image를 함께 사용해서 학습
          - VOC의 경우, training set만 사용
          - COCO의 경우, 모든 이미지를 사용 (VOC data에 없는 class는 background로 처리)
        - Optimizer
          - SGD with momentum 0.9
        - Iteration (batch size 14)
          1. 100K with learning rate $10^{-3}$
          2. 40K with learning rate $10^{-4}$
      - 2nd stage
        - PASCAL VOC-2012의 image만을 사용해서 fine-tuning
          - Training set만 사용
        - Optimizer
          - SGD with momentum 0.9
        - Iteration (batch size 14)
          - 50K with learning rate $10^{-5}$
    - Front-end module만을 사용한 경우의 performance (without context module)
      - PASCAL VOC 2012 validation set
        - mean IoU : 69.8%
      - PASCAL VOC 2012 test set
        - mean IoU : 71.3%
  - Context module
    - learning rate $10^{-3}$ 외에는 논문에 자세한 내용이 나와있지 않음

![table_3](/assets/multi-scale_context_aggregation_by_dilated_convolutions/table_3.png){:width="700px" style="border:1px solid black"}

- *Table 3*은 **PASCAL VOC 2012 validation set에서 context module을 추가하였을 때 성능이 어떻게 변하는지를 비교**한 것이다.
  - Front end, Front end + CRF, Front end + RNN은 각각 FCN, DeepLab V1, CRF-RNN에 front-end module을 적용한 네트워크들을 의미
  - 위의 3가지 네트워크 모두에서 **context module을 추가하였을 때, 성능이 향상**됨
  - Large 버전이 Basic 버전보다 약간 성능이 더 좋음

![table_4](/assets/multi-scale_context_aggregation_by_dilated_convolutions/table_4.png){:width="700px" style="border:1px solid black"}

- *Table 4*는 **PASCAL VOC 2012 test set에서 다른 네트워크들과의 성능을 비교**한 것이다.
  - **Context**(Large 버전의 context module을 front end에 적용한 것)**만으로도 DeepLab-CRF-COCO-LargeFOV보다 성능이 좋음**
  - Context에 CRF-RNN을 추가한 네트워크가 가장 성능이 좋음

![fig_3](/assets/multi-scale_context_aggregation_by_dilated_convolutions/fig_3.png){:width="600px" style="border:1px solid black"}

- *Figure 3*은 *Table 3, 4*에서 비교한 네트워크들의 segmentation 결과를 나타낸 것이다.

# Conclusion

- **Dilated convolution은 resolution의 손실 없이 receptive field를 확장**할 수 있으므로, dense prediction에 적절한 convolution 연산이다.
- Dilated convolution을 적용한 **context module을 기존의 semantic segmentation 네트워크에 연결하여 성능을 향상**시킬 수 있다.
- 기존의 semantic segmentation 네트워크에서 사용한 **classification network를 front-end module과 같이 수정하여 성능을 향상**시킬 수 있다.
  - **Image classification을 위한 CNN 구조를 변형했다는 점이 기존 연구들과의 큰 차이점**이며, 새로운 연구의 가능성을 열어줌