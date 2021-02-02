---
title: "Learning Deconvolution Network for Semantic Segmentation"
key: 20210203
sidebar:
  nav: papers-ko
tags: 논문 ML/DL Semantic&nbspSegmentation
---

<p align="center">
<img src="/assets/learning_deconvolution_network_for_semantic_segmentation/deconvnet_title.png" alt="deconvnet">
</p>

이 글은 [Learning Deconvolution Network for Semantic Segmentation](https://arxiv.org/abs/1505.04366) 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

# Abstract

- Deconvolution Network(DeconvNet)는 **deconvolution** 및 **unpooling** layer로 구성된다.
- 입력 이미지에 대해서 여러 **proposal**들을 생성하고, 이들을 입력으로 네트워크를 통과시켜 나온 결과들을 모두 합쳐서 최종 segmentation map을 생성한다.
- 다음의 2가지 방법으로 Fully convolutional network(FCN)에서의 한계들을 보완하여 **detailed structure** 및 **multiple scale**의 object들을 다룰 수 있게 되었다.
  - Deep deconvolution network : detailed structure
  - Proposal-wise prediction : multiple scale objects
- PASCAL VOC 2012 dataset에서 가장 높은 accuracy 72.5%를 기록하였다.

# Introduction

![fig_1](/assets/learning_deconvolution_network_for_semantic_segmentation/fig_1.png)

- **FCN**은 CNN을 통과하여 나온 coarse label map에서 bilinear interpolation으로 구현된 **단순한 deconvolution**을 수행하므로, 다음과 같은 **한계점**들이 있다.
  - 고정된 크기의 receptive field로 인해 **single scale의 object밖에 다루지 못한다.**
    - Receptive field보다 아주 **큰 object**의 경우, *Figure 1(a)*와 같이 **일관되지 못한 결과**가 나타난다.
    - Receptive field보다 아주 **작은 object**, *Figure 1(b)*와 같이 **무시되거나 배경으로 분류**된다.
  - Deconvolution 과정이 너무 simple해서 **detailed structure를 얻지 못한다.**
- 본 논문의 주된 **contribution**을 요약하면 다음과 같다.
  - **Multi-layer deconvolution network**를 학습한다.
    - Detailed structure 문제를 개선
  - Proposal을 생성하고 학습한 네트워크에 통과시켜 **instance-wise segmentation**을 수행한다. 그리고 결과들을 모두 합쳐서 semantic segmentation을 수행한다.
    - Single scale밖에 다루지 못하는 문제를 개선
  - **FCN과 앙상블**하여 PASCAL VOC 2012 dataset에서 가장 높은 accuracy를 기록하였다
    - FCN과 DeconvNet은 서로 heterogeneous 및 complementary한 특성을 지님

# System Architecture

- 여기서는 DeconvNet의 architecture에 대해서 설명한다.

## Architecture

![fig_2](/assets/learning_deconvolution_network_for_semantic_segmentation/fig_2.png)

- *Figure 2*는 DeconvNet의 전체 구조를 보여준다.
  - **Convolution network는 feature extractor**로서 입력 이미지를 다차원의 feature representation으로 변환한다.
    - VGG 16을 사용하였으며, 마지막 fully connected layer는 class-specific projection을 수행하도록 2개의 1x1 convolution layer로 변경하였다.
    - Convolution, pooling, relu로 구성된다.
  - **Deconvolution network는 shape generator**로서 feature representation으로부터 segmentation을 생성한다.
    - Convolution network의 mirrored version이다.
    - Deconvolution, unpooling, relu로 구성된다.
  - **최종 출력**은 입력 이미지와 동일한 크기의 **probability map**이다.
    - 각 pixel의 element는 각기 다른 클래스에 대한 확률값들을 가진다. 즉, 21개의 class라면 1 픽셀에 21개의 element가 존재하는 것이다.

## Deconvolution Network for Segmentation

### Unpooling

![unpooling](/assets/learning_deconvolution_network_for_semantic_segmentation/unpooling.png)

- Convolution network에서의 pooling을 역으로 다시 수행하여 원본 크기의 activation을 재구성한다.
- **Max pooling에서 최댓값으로 활성화 되었던 위치를 저장**하여 unpooling에 사용한다.

### Deconvolution

![deconvolution](/assets/learning_deconvolution_network_for_semantic_segmentation/deconvolution.png)

- FCN과 같이 본 논문에서도 deconvolution으로 표기하였지만, 더 정확히는 **transposed convolution**이다.
- Convolution과 반대로 single input activation(sparse)에서 multiple output activation(dense)을 생성한다.

### Analysis of Deconvolution Network

![fig_4](/assets/learning_deconvolution_network_for_semantic_segmentation/fig_4.png)

- Deconvolutional layer의 activation을 visualize해보면, object의 structure를 coarse에서 fine의 단계로 재구성한다는 것을 알 수 있다. (**hierarchical structure of deconvolutional layers**)
  - **하위 layer**의 filter는 **전체적인 구성(위치, 모양 및 영역)**을 생성한다.
  - **상위 layer**에서는 더 **복잡한 패턴**을 생성한다.
- Unpooling과 deconvolution은 segmentation을 생성하는데 서로 다른 역할을 한다.
  - **Unpooling**은 max pooling에서 가장 크게 활성화된 위치 정보를 사용해서 원본 이미지의 위치를 추적하며 **example-specific**한 structure를 capture한다.
    - *Figure 4의 (c), (e), (g), (i)*에서 **detailed structure**를 높은 해상도로 재구성하는 것을 확인할 수 있다.
  - **Deconvolutional layer**의 학습된 filter는 **class-specific**한 shape를 capture한다.
    - *Figure 4의 (b), (d), (f), (h), (j)*에서 **class와 연관된 영역의 activation은 더 강해지고 다른 영역의 노이즈들은 줄어드는 것**을 확인할 수 있다.

![fig_5](/assets/learning_deconvolution_network_for_semantic_segmentation/fig_5.png)

- *Figure 5*는 DeconvNet이 unpooling과 deconvolution을 통해 **FCN보다 더욱 정확하고(precise) 세밀한(dense) activation을 생성**할 수 있다는 것을 보여준다.

## System Overview

- DeconvNet은 **instance-wise segmentation**을 통해 semantic segmentation을 수행한다.
  - **입력 이미지로부터** object를 잠재적으로 포함하는(potentially containing objects) **proposal들을 추출**한다. (논문에서는 각각의 proposal 또는 sub-image를 instance라고 부름)
  - 위의 **proposal들을 입력으로 받아 네트워크가 출력한 결과들을 원래 이미지 공간에 합치는 방식**으로 전체 이미지에 대한 semantic segmentation을 수행한다.
  - **여러 scale의 object**를 효과적으로 다룰 수 있고, object의 **fine detail**까지 인식할 수 있다.
  - Sub-image에서 prediction을 수행하므로 search space가 줄어들게 되고, 이는 training complexity를 완화시켜주며 training에 필요한 메모리 요구량을 줄여준다.

# Training

- PASCAL dataset은 training 및 validation에 총 12031개의 이미지로 구성되는데, 이는 DeconvNet을 학습하기에 부족하다.
- 여기서는 적은 데이터로 깊은 네트워크를 학습하기 위해 사용한 방법들에 대해서 설명한다.

## Batch Normalization

- 모든 convolutional 및 deconvolutional layer의 output에 **batch normalization layer를 추가**하였다.

## Two-stage Training

- 학습 데이터 수에 비해 semantic segmentation을 수행하는 공간(이미지의 영역을 의미하는것 같음)이 아주 크기 때문에, 쉬운 데이터로 먼저 학습한 다음 어려운 데이터로 fine-tuning하는 **two-stage training** 방법을 사용하였다.
  - First stage (**쉬운 데이터**로 학습)
    - Ground-truth annotation을 사용해 **object가 정중앙에 위치**하도록 bounding box를 생성하고 이를 crop하는 방식으로 학습 데이터(object instance)를 구성하여 학습한다.
    - **Object의 위치 및 크기 변화를 제한**함으로서 search space를 줄일 수 있기 때문에, 학습이 더 쉬워지고 **적은 데이터로도 네트워크를 훈련**시킬 수 있게 된다.
  - Second stage (**어려운 데이터**로 학습)
    - **[Edge Boxes](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/09/ZitnickDollarECCV14edgeBoxes.pdf)**에서 제안한 방법을 통해 **proposal들을 생성**하는 방식으로 학습 데이터를 구성하여 학습한다.
    - 이때, 실제 정답에서의 segmentation(ground-truth segmentation)과 잘 겹치는 proposal들만을 사용한다. (IoU가 0.5이상인 proposal들)
    - **Object의 위치 및 크기가 많이 달라지기** 때문에 학습은 더 어려워지지만, proposal의 misalignment에 대해 **네트워크가 더욱 robust하도록** 만들어준다.

# Inference

- DeconvNet은 입력 이미지에서 proposal을 생성하고 각각에 대해 동작한 결과들을 결합해서 전체 이미지에 대한 최종 segmenation map을 생성한다.
- 여기서는 DeconvNet의 동작 과정에 대해 보다 자세히 설명한다.

## Aggregating Instance-wise Segmentation Maps

- Object에 대한 misalignment나 cluttered background로 인해 발생하는 몇몇 proposal에 대한 부정확한 prediction으로 인한 **noise는 aggregation 과정에서 완화**할 수 있다.
  - 여러 proposal(instance)로부터 생성된 output score map들에 **pixel-wise maximum 또는 average**를 취하는 방식으로 aggregation을 수행한다.
- Aggregation을 수행한 결과에 **softmax** 함수를 적용하면 **pixel-wise class conditional probability map**을 얻을 수 있고, 여기에 마지막으로 **[fully-connected CRF](https://arxiv.org/abs/1210.5644)**를 적용하면 **최종 pixel-wise label map**을 얻을 수 있다.

## Ensemble with FCN

- **FCN은 DeconvNet과 complementary**한 특성을 지니므로 앙상블을 사용해서 성능을 향상시켰다.
  - DeconvNet
    - Object의 **fine-detail**을 capture하는데 적절하다.
    - Instance-wise prediction을 통해 **다양한 scale의 object**를 다룰 수 있다.
  - FCN
    - Object의 **overall shape**를 추출하는데 좋은 성능을 보인다.
    - Coarse scale map을 사용하므로 **이미지의 context**를 capture 할 수 있다.

# Experiments

- 여기서는 상세한 구현 및 실험 과정에 대해 설명한다.

## Implementation Details

#### Network Configuration

![table_2](/assets/learning_deconvolution_network_for_semantic_segmentation/table_2.png)

- DeconvNet의 네트워크 구성은 *Table 2*와 같다.
  - 여기서 **fc6, fc7**은 fully connected가 아니라 **1x1 convolution layer**이다.

#### Dataset

- PASCAL VOC 2012 segmentation dataset만을 사용해 training 및 test를 하였다.
  - 학습에는 training, validation 이미지를 모두 사용하였다.
  - Evaluation에는 test 이미지를 사용하였다.

#### Training Data Construction

- Two-stage training의 각 stage 별로 training dataset을 분리하여 사용하였다.
  - First stage
    - Training image의 각 object별로 object의 크기와 딱맞는 bounding box를 그린 후, local context를 포함시키기 위해 bounding box의 크기를 1.2배로 키워주고 해당 영역을 crop하여 training data를 구성한다.
    - Crop한 영역에서 object는 항상 중앙에 위치하고, 나머지 픽셀들은 모두 background이다.
  - Second stage
    - [Edge Boxes](https://www.microsoft.com/en-us/research/wp-content/uploads/2014/09/ZitnickDollarECCV14edgeBoxes.pdf)에서 제안한 방법으로 instance(proposal)를 생성하여 training data를 구성한다. (이때는 한 instance에 여러 class label이 있을 수 있다.)
    - First stage에서와 동일한 방법으로 local context를 포함시켜준다.
  - 위의 두 dataset에서 데이터 수가 부족한 class들에 대해서는 데이터를 중복해서 사용하는 방식으로 balance를 맞춰주었다.
  - Data augmentation 방법으로는 각 instance(proposal)를 250 $\times$ 250 크기로 변환한 다음 224 $\times$ 224 크기로 randomly crop하는 방식을 사용하였다.

#### Optimization

- SGD with momentum을 사용하였다.
  - initial learning rate : 0.01
  - momentum : 0.9
  - weight decay : 0.0005
- Weight initialization
  - Convolution network의 weight는 ImageNet에서 학습된 VGG16의 weight로 초기화하였다.
  - Deconvolution network의 weight는 zero-mean gaussian으로 초기화하였다.
- Batch Normalization을 사용하여 drop-out은 사용하지 않았다.
- Validation accuracy가 개선되지 않을 때마다 learning rate을 감소시키며 학습을 진행하였다.
- First, second stage 각각에 mini-batch size 64로 약 20K, 40K의 iteration에서 네트워크가 수렴하였다.

#### Inference

- 각 testing image에 대해서 Edge Boxes에서 제안한 방법으로 약 2000개의 object proposal을 생성한 후, 이들 중 objectness score가 가장 높은 50개의 proposal을 선정한다.
- 그리고 이들에 대해 instance-wise prediction을 수행하고, aggregation을 통해 전체 이미지에 대한 pixel-wise class conditional probability map을 생성한다.

## Evaluation on Pascal VOC

![table_1](/assets/learning_deconvolution_network_for_semantic_segmentation/table_1.png)

- *Table 1*은 PASCAL VOC 2012 test set에서 다른 모델들과의 성능을 비교한 것이다.
- FCN-8s와의 앙상블 및 CRF를 적용한 모델(EDeconvNet + CRF)이 가장 높은 성능을 기록하였다.
  - [CRF](https://arxiv.org/abs/1210.5644)은 약 1%의 성능을 향상시켜 주었다.
  - FCN-8s와의 **앙상블(EDeconvNet)은** 단일 모델(FCN8s, DeconvNet)의 **성능을 크게 향상**시켜 주었다.

![fig_6](/assets/learning_deconvolution_network_for_semantic_segmentation/fig_6.png)

- *Figure 6*는 **instance-wise prediction이 정확한 segmentation에 도움이 된다**는 것을 보여준다. (proposal의 수가 증가할수록 더 세세한 구조를 인식할 수 있음)

![fig_7](/assets/learning_deconvolution_network_for_semantic_segmentation/fig_7.png)

- *Figure 7*은 DeconvNet, FCN, EDeconvNet, EDeconvNet + CRF를 각각 비교한 것이다.
  - **FCN**은 **아주 크거나 작은 object**에서 성능이 좋지 못하다. (*Figure 7(a)*)
  - **DeconvNet**은 FCN보다 **fine segmentation**을 생성할 수 있고 **multi-scale object**를 다룰 수 있지만, **가끔 noisy한 결과**(*Figure 7(b)*)를 보인다.
  - **EDeconvNet**부터는 **FCN과 DeconvNet보다 좋은 성능**을 보인다. (*Figure 7(a), 7(b)*)
  - FCN, DeconvNet에서 정확하지 못한 prediction이 있는 경우에도, EDeconvNet에서 좋은 성능을 보이는 경우도 있었다. (*Figure 7(c)*)
  - CRF를 추가한 경우 성능이 향상되기는 하지만, 눈에 띄는 정도는 아니다.

# Conclusion

- DeconvNet은 **deconvolution** 연산을 통해 object의 **coarse-to-fine structure를 재구성**함으로써 세세하고 정확한(**dense and precise**) segmentation을 수행할 수 있다.
- DeconvNet의 **instance-wise prediction**은 FCN의 fixed-size receptive field의 한계를 극복하여 **다양한 scale의 object**를 다룰 수 있게 해준다.
- **DeconvNet은 FCN과 complementary**한 특성이 있어서 두 모델을 앙상블한 결과 성능이 크게 향상되었다.
- PASCAL VOC 2012에서 SOTA의 성능을 기록하였다.