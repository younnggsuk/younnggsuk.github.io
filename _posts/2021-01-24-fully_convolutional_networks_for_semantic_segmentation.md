---
title: "Fully Convolutional Networks for Semantic Segmentation"
key: 20210124
sidebar:
  nav: papers-ko
tags: 논문 ML/DL Semantic&nbspSegmentation
---

<p align="center">
<img src="/assets/fully_convolutional_networks_for_semantic_segmentation/fully_convolutional_networks_for_semantic_segmentation.png" alt="fcn">
</p>

이 글은 [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1411.4038) 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

# Abstract

본 논문에서 제안하는 **Fully Convolutional Network(FCN)**에 대해 요약하면 다음과 같다.
- **Convolutional network만으로 구성**된 네트워크이며, **end-to-end로 학습**한다.
- **어떠한 크기의 입력이 들어와도 같은 크기의 출력으로 semantic segmentation을 수행**할 수 있다.
- PASCAL VOC, NYUDv2, SIFT Flow에서 SOTA를 기록하였고 inference 시간은 5배나 빠르다.
- AlexNet, VGGNet과 같은 **classification model을 fully convolutional network로 변환**하고, segmenation task를 수행할 수 있도록 **fine-tuning**하였다.
- **깊은 layer의 semantic information과 얕은 layer의 appearance information을 결합**하여 정확하고 정밀한 segmentation을 수행한다. 

# Introduction

Convolutional networks는 classification뿐만 아니라, object detection, key-point prediction 등의 task에서도 우수한 성능을 보였다.

Segmentation task에도 convolutional network를 적용하려는 여러 연구들이 있었지만, post/pre processing, superpixel, proposal, post-hoc과 같은 방법들을 사용하여 복잡한 구조를 가진다는 단점이 있었다.

본 논문에서 소개하는 **fully convolutional network(FCN)는 단순한 end-to-end 구조로 semantic segmentation task에서 SOTA 기록하며** convnet을 segmentation에 성공적으로 접목시키게 된다.

FCN의 특징을 정리하면 다음과 같다.
- **어떠한 크기의 입력에서도 dense prediction을 수행**할 수 있다.
- **Feedforward computation 및 backpropagation이 whole-image에서 한번에 수행되므로 학습 및 추론에 효율적**이다.
- **pre-trained classification net을 fully convolutional로 변경하여 segmentation을 수행하도록 fine-tuning하는 방식으로 학습**한다.
- **Skip architecture를 통해 semantic information과 appearance information을 결합하여 정확하고 정밀한 semantic segmentation을 수행**한다.
  - Feature hierarchy에서 깊은 layer의 global information은 object가 무엇인지를 알려주고, 얕은 layer의 local information은 object가 어디에 있는지를 알려준다.
  - Skip architecture는 위의 2가지 정보를 결합하여 정확하고 정밀한 semantic segmentation을 수행할 수 있도록 한다.

# Fully convolutional networks

여기서는 4개의 소챕터로 나누어서 각각 아래의 내용에 대해 설명한다.

- Classification net을 fully convolutional net으로 변환해서 coarse output map을 생성하는 방법
- OverFeat 논문에서 제안된 input shifting and output interlacing을 이용한 upsampling에 대한 고찰
- FCN에서 사용한 upsampling 방법인 transposed convolution에 대한 설명
- Patchwise training에 대한 고찰

## Adapting classifiers for dense prediction

AlexNet이나 VGGNet과 같은 **classification network의 마지막 fully connectecd layer를 convolution으로 변경하여 segmentation task에서 얻을 수 있는 이점**은 다음의 2가지가 있다.

### Arbitrary-sized input

![fig_2](/assets/fully_convolutional_networks_for_semantic_segmentation/fig_2.png){:width="600px" style="border:1px solid black"}

Classification net의 마지막 fully connected layer는 입력받을 수 있는 이미지의 크기(입력 차원)가 고정되어 있으며, 공간 정보를 보존하지 못한다는 문제가 있다. 이를 **convolutional network로 변경하면, 어떠한 크기의 입력이라도 출력으로 classification map을 생성할 수 있고, 이와 동시에 공간 정보를 보존할 수 있다.**
- Fully connected layer는 Width $\times$ Height $\times$ Channel 차원의 입력을 받으므로, Width나 Height가 변하면 입력 차원이 변하게 되어 네트워크 구조를 변경해야 한다.
  - `torch.nn.Linear(channel*height*width, output_dim)`
- 반면, Convolutional layer는 Channel 차원의 입력을 받으므로, Width나 Height가 변해도 네트워크 구조를 변경하지 않아도 된다.
  - `torch.nn.Conv2d(input_channel, output_channel)`

### Computational efficiency

![fig_1](/assets/fully_convolutional_networks_for_semantic_segmentation/fig_1.png){:width="600px" style="border:1px solid black"}

출력으로 얻은 classification map의 모든 output cell은 ground truth와 바로 비교할 수 있기 때문에, **forward 및 backward pass가 straightforward로 이루어지며, 이는 계산에 효율적**이다.

다음은 논문에서 제시하는 AlexNet에서 fully connected와 fully convolution에서의 forward 및 backward pass 계산 속도를 실험한 결과이며, fully convolution이 더욱 빠른 것을 볼 수 있다.
- 227 $\times$ 227 크기의 이미지를 입력으로 1개의 classification score를 출력하는 경우
  - forward pass : 1.2ms
  - backward pass : 2.4ms
- 500 $\times$ 500 크기의 이미지를 입력으로 10$\times$10 grid의 score를 출력하는 경우
  - forward pass : 22ms
  - backward pass : 37ms

## Shift-and-stitch is filter rarefaction

Pixelwise prediction을 위해서는 fully convolution을 통해 출력된 coarse output map을 입력 이미지의 크기로 upsample해주는 방법이 필요하다. 
여기서는 [OverFeat](https://arxiv.org/abs/1312.6229) 논문에서 제안된 **shift-and-stitch trick을 통해 output size를 유지하는 방법을 왜 사용하지 않았는지에 대해 설명**한다.

<figure>
  <img src="/assets/fully_convolutional_networks_for_semantic_segmentation/shift_and_stitch.png" alt="shift_and_stitch" width="600px"/>
  <figcaption><i><a href="https://stackoverflow.com/questions/40690951/how-does-shift-and-stitch-in-a-fully-convolutional-network-work">그림 출처 : https://stackoverflow.com/questions/40690951/how-does-shift-and-stitch-in-a-fully-convolutional-network-work</a></i></figcaption>
</figure>

위 그림은 shift-and-stitch trick이 어떻게 입력과 동일한 크기의 출력을 유지하는지를 보여준다.

Maxpooling layer에서 5$\times$5 크기의 입력에 zero-padding 1, stride 2가 적용된다고 하면, 위 그림 우측 상단의 빨간색 grid로만 이루어진 3$\times$3 크기의 출력이 나올 것이다. (5$\times$5 크기의 입력이 3$\times$3 크기로 downsample된 것) 여기에 다음과 같이 shift-and-stitch trick을 적용하면, output 크기가 downsample되지 않도록 할 수 있다.
1. 좌측 상단을 (0,0)이라고 할 때, (0,0), (1,0), (0,1), (1,1)의 위치에서 동일하게 2x2 filter로 maxpooling을 적용한다. (각 위치에 대한 filter의 색깔은 빨간색, 노란색, 초록색, 파란색)
2. 1에서의 과정을 통해 출력된 (우측 상단의) 4개의 3$\times$3 크기의 결과들을 조합하면, (우측 하단의) 5$\times$5 크기의 출력을 만들어 낼 수 있다.(padding은 제외하고, 각 grid들을 조합한 결과)

위와 같은 trick을 사용하면 output size를 input과 동일하게 유지할 수 있지만, **본 논문에서는 다음과 같은 trade off가 있다고 주장**하며 이 방법을 사용하지 않는다.
- Shift-and-stitch trick을 통해 output의 크기를 유지하는 방법은 **receptive field의 크기를 줄이지 않으면서 output을 dense하게 만들어주지만, filter는 원래의 방법보다 세밀한 정보를 보지 못한다.**

## Upsampling is backwards strided convolution

**FCN에서는 upsampling 방법으로 아래 그림과 같이 동작하는 Transposed Convolution 방법을 사용**한다.  
(논문에서는 Deconvolution이라고 표기하였지만 수학적으로 엄밀하게 따졌을 때 Convolution의 역연산은 아니므로, Transposed Convolution이라고 부르는 것이 더 타당하다. [[여기에 대한 자세한 설명은 여기를 참고]](https://www.slideshare.net/ssuserb208cc1/transposed-convolution))

<figure>
  <img src="/assets/fully_convolutional_networks_for_semantic_segmentation/transposed_convolution.gif" alt="transposed_convolution"/>
  <figcaption><i><a href="https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d">그림 출처 : An Introduction to different Types of Convolutions in Deep Learning</a></i></figcaption>
</figure>

**Transposed convolution은** (downsampling하는 convolution에서 kernel을 통해 연산하는 것처럼) **upsampling하는 convolution의 kernel과의 연산을 통해 수행한다.** (이때, Upsampling에서의 kernel도 backprop을 통해 학습되는 파라미터이다.)

아래 그림은 stride는 1이고 2$\times$2 크기의 kernel이 있을 때, Transpose Convolution으로 2$\times$2 크기의 입력을 3$\times$3 크기의 Output으로 출력하는 예이다. 

![transposed_convolution_hand_written](/assets/fully_convolutional_networks_for_semantic_segmentation/transposed_convolution_hand_written.png){:width="600px" style="border:1px solid black"}

## Patchwise training is loss sampling

<figure>
  <img src="/assets/fully_convolutional_networks_for_semantic_segmentation/patch_wise.png" alt="patch_wise" width="300px"/>
  <figcaption><i><a href="https://arxiv.org/abs/1504.07947">그림 출처 : Patch-based Convolutional Neural Network for Whole Slide Tissue Image
Classification</a></i></figcaption>
</figure>

**Patchwise training이란, 전체 이미지 대신 관심 객체(objects of interest)를 둘러싼 영역의 이미지에서 random하게 추출한 patch를 입력으로 학습하는 방법**이다.

Patchwise training은 (whole image를 학습하는) fully convolutional training과 비교해서 다음과 같은 장단점이 있다.
- 장점
  - class imbalance 문제가 있을 때, sampling을 통해 balance를 맞춰줄 수 있다.
  - patch 단위로 학습하기 때문에, 입력 이미지의 공간적 상관관계(spatial correlation)를 줄여준다.
- 단점
  - whole image로 학습하는 방법보다 속도가 느리다.

또한, patchwise training과 fully convolutional training에서의 output을 계산하는 방법을 비교하면 다음과 같다.

- Patchwise training은 whole image에서 subimage를 crop하고, 이 subimage에 대해 독립적으로 forward pass를 수행하여 output을 계산
- Fully convolutional training은 whole image에서 forward pass를 수행하여 모든 subimage에서의 output을 계산

따라서, 본 논문에서는 fully convolutional 방법의 **whole image에서 계산된 모든 subimage의 output중에서 몇개의 output만을 사용해서 backward pass를 수행하면 patchwise training에서의 장점은 취하면서 더욱 빠르게 학습을 할 수 있다**고 주장한다. (즉, 전체 output에서 backward pass를 수행할 output을 sampling하는 것이므로, 논문에서는 loss sampling이라고 불렀음)

위의 내용들을 정리하면, Patchwise training에서의 장점은 fully convolutional training에서 다음과 같이 얻을 수 있다.
- **Class imbalance 문제는 loss에 weight를 추가**하면 된다.
- **Spatial correlation은 최종 output map에서 loss sampling**을 하면 된다.

# Segmentation Architecture

여기서는 3개의 소챕터로 나누어서 각각 아래의 내용에 대해 설명한다.

- Classifier를 fully convolutional로 변경한 후, segmentation을 수행하도록 fine-tuning한 결과
- Skip architecture를 적용한 결과
- 모델 학습에 사용된 방법들에 대한 설명

## From classifier to dense FCN

아래 표는 ImageNet dataset에서 학습된 3가지 **classification network(AlexNet, VGG16, GoogLeNet)를 fully convolutional로 변경하고 segmentation을 수행하도록 fine-tuning하여 PASCAL VOC 2011 segmentation challenge에서 비교**한 것이다.
- 마지막 classifier layer는 fully convolutional layer로 변경
  - transposed convolutional layer로 upsampling을 수행
- 21 channel을 반환하는 1$\times$1 convolutional layer을 추가
  - 배경을 포함한 PASCAL VOC dataset의 클래스 수가 총 21개라서 21개의 channel(score)을 반환

![table_1](/assets/fully_convolutional_networks_for_semantic_segmentation/table_1.png){:width="600px" style="border:1px solid black"}

여기서 주목할 것은, 위 결과가 learning rate을 고정시킨채로 학습시킨 결과이며 성능을 더 개선시킬 수 있음에도 불구하고 VGG16의 경우 이미 validation에서 56.0의 mean IU로 SOTA를 기록하였다는 것이다. 

따라서, **classification net을 segmentation으로 fine-tuning하는 것이 아주 reasonable하다**는 것을 알 수 있다.

## Combining what and where

앞의 챕터에서 fine-tuning한 fully convolutionalized classifier가 우수한 성능을 보이는 것을 이미 확인하였지만, 한번의 transposed convolution을 통해 32배로 upsampling을 수행한 결과는 여전히 coarse하다.

따라서, 본 논문에서는 **"Skip Architecture"**라고 부르는 아래의 과정을 통해 fine layer와 coarse layer를 결합하여 보다 정교한 prediction을 수행할 수 있도록 하였다. (FCN-8s까지만 있는 이유는 더 낮은 layer(pool2)를 사용하면서부터는 오히려 성능이 나빠졌기 때문)

- **FCN-32s : pool5를 통과한 출력 (1$\times$1)을 32의 stride로 upsample하여 32$\times$32 크기의 결과를 출력**
- **FCN-16s : 아래의 두 출력을 더한 결과 (2$\times$2)에 16의 stride로 upsample하여 32$\times$32 크기의 결과를 출력**
  - pool5를 통과한 출력 (1$\times$1)을 2의 stride로 upsample : **2$\times$2**
  - pool4를 통과한 출력 (2$\times$2)에 1$\times$1 convolution을 통과시킨 출력 : **2$\times$2**
- **FCN-8s : 아래의 두 출력을 더한 결과 (4$\times$4)에 8의 stride로 upsample하여 32$\times$32 크기의 결과를 출력**
  - 아래의 두 출력을 더한 결과 (2$\times$2)에 2의 stride로 upsample : **4$\times$4**
    - pool5를 통과한 출력 (1$\times$1)을 2의 stride로 upsample : **2$\times$2**
    - pool4를 통과한 출력 (2$\times$2)에 1$\times$1 convolution을 통과시킨 출력 : **2$\times$2**
  - pool3를 통과한 출력 (4$\times$4)에 1$\times$1 convolution을 통과시킨 출력 : **4$\times$4**

![fig_3](/assets/fully_convolutional_networks_for_semantic_segmentation/fig_3.png){:style="border:1px solid black"}

아래의 그림 및 표는 Skip Architecture를 적용한 결과(FCN-32s, FCN-16s, FCN-8s)를 비교한 것이다.
- 그림에서, **FCN-32s에서 FCN-8s로 갈수록 segmentation 결과가 점점 정교**해지는 것을 알 수 있다.
- 표에서, **FCN-32s에서 FCN-8s로 갈수록 성능이 증가**하는 것을 알 수 있다.

![fig_4](/assets/fully_convolutional_networks_for_semantic_segmentation/fig_4.png){:width="600px" style="border:1px solid black"}

![table_2](/assets/fully_convolutional_networks_for_semantic_segmentation/table_2.png){:width="600px" style="border:1px solid black"}

## Experimental framework

### Optimization

학습에 사용된 파라미터는 다음과 같으며, 이들 중에서 learning rate 외에는 모델에 큰 영향을 미치지 않았다.

- SGD momentum을 사용
  - momentum : 0.9
- mini-batch size : 20
- learning rate (bias에는 2배)
  - FCN-AlexNet : $10^{-3}$로 고정
  - FCN-VGG16 : $10^{-4}$로 고정
  - FCN-GoogLeNet : $10^{-5}$로 고정
- weight decay : $5^{-4}$ 또는 $2^{-4}$
- dropout : Classifier에서 사용된 위치에 그대로 사용

### Fine-tuning

**모든 layer에서 backpropagation이 되도록 Fine-tuning을 수행**하였으며, classifier에만 fine-tuning을 수행한 경우(Table 2의 FCN-32s-fixed)에는 70% 정도의 성능밖에 나오지 않았다.

### Patch Sampling

![fig_5](/assets/fully_convolutional_networks_for_semantic_segmentation/fig_5.png){:width="600px" style="border:1px solid black"}

위 그림은 patchwise sampling을 통한 학습이 수렴을 가속하는지를 분석한 결과이다. (여기서의 patch sampling은 "Patchwise training is loss sampling" 챕터에서 설명한 whole-image에서의 loss sampling을 의미)
- **Patchwise sampling이 수렴 속도에 큰 영향을 미치지 않음** (왼쪽 그래프)
- **Whole-image를 이용한 학습한 경우, patchwise에 비해 속도가 더욱 빠르다.** (오른쪽 그래프)

따라서, 본 논문에서는 sampling을 수행하지 않고 whole image로 학습하였다.

### Class Balancing

Fully convolutional training의 경우, loss에 weight를 추가하는 방법으로 class의 balance를 맞춰줄 수 있지만, 학습에 사용한 데이터에 큰 불균형이 없어서 balancing은 불필요하였다.

### Dense Prediction

네트워크에서 upsampling을 수행하는 transposed convolutional layer에 대한 세부적인 내용은 다음과 같다.
- **마지막**의 transposed convolutional layer
  - **bilinear interpolation으로 고정**
- **중간**의 transposed convolutional layer
  - **초기화 시에는 bilinear interpolation을 사용하고, 그 이후부터는 학습되도록 설정**

# Results

본 논문에서는 Fully Convolutional Network(FCN)의 semantic segmentation 성능을 PASCAL VOC, NYUDv2, SIFT Flow에서 평가하였다.

### Metrics

본 논문에서는 다음의 4가지 metric을 통해 성능을 평가하였다.

- **pixel accuracy**
  - $\sum_i n_{ii} / \sum_i t_i$ : class $i$로 예측한 픽셀 수 / class $i$의 전체 픽셀 수 
- **mean accuracy**
  - $(1/n_{cl}) \sum_i n_{ii} / t_i$ : pixel accuracy를 전체 클래스 수로 평균낸 것
- **mean IU**
  - $(1/n_{cl}) \sum_i n_{ii} / \big(t_i + \sum_j n_{ji} - n_{ii}\big)$ : IU를 전체 클래스 수로 평균낸 것
- **frequency weighted IU**
  - $(\sum_k t_k)^{-1} \sum_i t_i n_{ii} / \big(t_i + \sum_j n_{ji} - n_{ii}\big)$ : 클래스 불균형을 완화하기 위해 class의 frequency가 적용된 weighted mean IU

위의 4가지 지표에서 사용된 각 항에 대한 설명은 다음과 같다.
- IoU(IU)  
![IU](/assets/fully_convolutional_networks_for_semantic_segmentation/IU.png){:width="250px" style="border:1px solid black"}
- $n_{ii}$ : class $j$로 예측된 class $i$를 가지는 픽셀의 수
- $n_{cl}$ : 전체 클래스의 수
- $t_i = \sum_j n_{ij}$ : class $i$의 총 픽셀 수

### PASCAL VOC

![table_3](/assets/fully_convolutional_networks_for_semantic_segmentation/table_3.png){:width="600px" style="border:1px solid black"}

![fig_6](/assets/fully_convolutional_networks_for_semantic_segmentation/fig_6.png){:width="600px" style="border:1px solid black"}

### NYUDv2

![table_4](/assets/fully_convolutional_networks_for_semantic_segmentation/table_4.png){:width="600px" style="border:1px solid black"}

### SIFT Flow

![table_5](/assets/fully_convolutional_networks_for_semantic_segmentation/table_5.png){:width="600px" style="border:1px solid black"}

# Conclusion

본 논문에서 소개하는 Fully convolutional network를 정리하면 다음과 같다.
- **classification network를 segmentation에 적용**
- **다른 resolution을 가진 layer들을 결합하여 성능을 향상**
- **end-to-end로 모델 구조를 간소화**
- **빠른 학습 및 추론 속도**