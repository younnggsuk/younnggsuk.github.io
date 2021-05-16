---
title: "Deep Residual Learning for Image Recognition"
key: 20210515
sidebar:
  nav: papers-ko
tags: Papers Image&nbspClassification
---

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/resnet_title.png" alt="resnet">
</p>

이 글은 <a href="https://arxiv.org/abs/1512.03385" target="_blank" rel="noopener noreferrer">Deep Residual Learning for Image Recognition</a> 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

# Abstract

- 본 논문에서는 residual learning framework를 제안한다.
- Residual learning은 네트워크의 수렴을 도와주고, degradation 문제를 해결하여 아주 깊은 네트워크라도 네트워크를 깊게 쌓을수록 더 높은 accuracy를 얻을 수 있게 해준다.
- Residual learning을 사용한 네트워크 ResNet은 ILSVRC 2015 classification task에서 1위를 차지하였고, ILSVRC 2015의 detection 및 localization, 그리고 COCO 2015의 detection 및 segmentation에서도 1위를 차지하며 다른 visual recognition task에도 좋은 일반화 성능을 보였다.

# Introduction

- 기존의 많은 연구들에서는 깊은 네트워크일수록 좋은 성능을 보였고, 이로 인해 네트워크를 깊게 쌓으며 다음과 같은 문제들이 나타나게 되었다.
  1. Vanishing / exploding gradients
  2. Degradation problem
- 첫번째 vanishing / exploding gradients 문제는 normalized initilaization 및 batch normalization으로 네트워크가 수렴하도록 해결할 수 있었지만, 두번째 degradation 문제는 적절한 솔루션이 없었다.

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_1.png" alt="fig_1" width="600px" style="border:1px solid black">
</p>

- *Figure 1*은 degradation problem을 보여준다.
  - Degradation problem은 깊은 네트워크가 얕은 네트워크보다 성능이 저하되는 문제를 의미하며, 이는 깊은 네트워크가 수렴하면서 발생하는 문제이다. (수렴이 되지 않는 문제가 아님)
  - 또한, *Figure 1*의 두 그래프에서 training 및 test error 모두 깊은 네트워크가 얕은 네트워크보다 좋지 못하므로 이 문제는 overfitting으로 일어난 것이 아님을 알 수 있다.

- 본 논문에서는 degradation problem의 해결을 위해 deep residual learning framework를 제안하였고, 이 문제를 해결하였다.

# Deep Residual Learning

## Residual learning

<figure align="center">
  <p align="center">
  <img src="/assets/deep_residual_learning_for_image_recognition/residual_block.png" alt="residual_block" border="1px solid black" width="700px">
  </p>
  <figcaption><i><a href="https://web.eecs.umich.edu/~justincj/teaching/eecs498/FA2019/schedule.html" target="_blank" rel="noopener noreferrer">그림 출처 : University of Michigan - EECS 498-007 / 598-005: Deep Learning for Computer Vision (2019) - Lecture 8 slide</a></i></figcaption>
</figure>

- 위 그림에서 왼쪽(plain block)은 기존 네트워크들에서 사용하던 구조이고, 오른쪽은 본 논문에서 제안하는 residual block의 구조이다.
- Plain block에서 $\mathbf{x}$는 input, $\mathcal{H}(\mathbf{x})$를 stacked layers가 학습하려고 하는 mapping이라고 할 때, stacked layers가 어떠한 복잡한 함수도 근사할 수 있다고 가정한다면, $\mathcal{H}(\mathbf{x})$가 아닌 residual function $\mathcal{H}(\mathbf{x}) - \mathbf{x}$ 도 근사할 수 있을 것이다.
- 따라서, 본 논문에서는 **stacked layers가 plain block처럼 $\mathcal{H}(\mathbf{x})$를 학습하는 것이 아니라, 오른편처럼 shortcut connection을 추가한 구조에서 $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$를 학습**하는 아이디어를 제안하였고, 이를 통해 **학습이 더 쉽게 된다**고 주장하였다.
  - Degradation problem을 해결하기 위해서는 깊은 네트워크가 얕은 네트워크와 최소한 동일한 성능을 내주어야 하는데,  이를 위해서는 깊은 네트워크가 최소한 얕은 네트워크의 output을 그대로 반환하는 identity mapping을 학습할 수 있어야 한다.
  - 그런데 plain block의 구조에서 degradation problem이 나타났다는 것은, stacked layers가 처음부터 새롭게 identity mapping을 학습하는 것이 어렵다는 것을 의미한다.
  - 따라서, 본 논문에서는 residual block의 구조에서 stacked layers가 단순히 모든 weights을 0으로 학습하도록 실험하여 degradation problem이 해결됨을 확인하였고, 이를 통해 residual learning이 학습을 더 쉽게 한다고 주장한 것이다.

## Identity Mapping by Shortcuts

- Residual learning의 building block은 수식으로 다음과 같이 나타낼 수 있다. (bias는 생략)

$$ \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x} \tag{1} $$

- Input과 output의 shape가 다른 경우에는 shortcut항에 다음과 같이 차원을 맞춰주는 linear projection $W_s$가 추가되어 다음과 같은 형태가 된다.

$$ \mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + W_s\mathbf{x} \tag{2} $$

- 위의 두 식에서, 각 항에 대한 설명은 다음과 같다.
  - $\mathbf{x}$ : input
  - $\mathbf{y}$ : output
  - $W_i$ : $i$번째 layer의 weights
  - $W_s$ : linear projection
  - $\mathcal{F}(\mathbf{x}, \{W_i\})$ : residual mapping

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_2.png" alt="fig_2" width="450px" style="border:1px solid black">
</p>

- *Figure 2*를 예로 들어 더 구체적인 식으로 나타내면, 식 $(1)$과 $(2)$는 각각 다음과 같다. ($\sigma$는 ReLU)

$$ \mathbf{y} = W_2 \sigma(W_1\mathbf{x}) + \mathbf{x} $$

$$ \mathbf{y} = W_2 \sigma(W_1\mathbf{x}) + W_s\mathbf{x} $$

## Network Architectures

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_3.png" alt="fig_3" width="500px" style="border:1px solid black">
</p>

### Plain Network

- *Figure 3*의 가운데 모델이 **residual learning을 사용하지 않은 plain network**이며, 다음의 design rule을 따른다.
  - 1번째 layer를 제외하고는 모두 $3\times3$ 크기의 filter를 가진 conv layer를 사용
  - 입력과 동일한 크기의 feature map을 출력하는 layer의 경우, 모두 같은 크기의 channel을 출력
  - $/2$로 표기된 부분은 stride를 2로 하여 down sampling을 수행하는 것을 나타내며, 이 경우에는 filter의 수를 2배로 하여 2배의 channel을 출력
  - 마지막은 Global Avearge Pooling + fully-connected layer로 구성
- Plain network는 **VGG-19(*Figure 3*의 왼쪽 모델)과 비교**해서 다음과 같은 특징이 있다.
  - **파라미터(filter의 수)가 더 적다.**
  - **연산량(FLOPs)이 더 적다**.

### Residual Network

- *Figure 3*의 오른쪽 모델이 residual learning을 사용한 residual network이며, **plain netwok에 shortcut connections을 추가**한 모델이다.
- 입출력의 dimension 및 weight의 유무에 따라, shortcut connection은 다음의 3가지 방법이 있다.
  - Identity shortcuts (식 $(1)$)
    - **실선**으로 표시된 shortcut connection이며, **입력과 출력의 dimension이 같은 경우**에 사용
    - **element-wise addition**
  - Zero-padding shortcuts
    - **점선**으로 표시된 shortcut connection이며, **입력보다 출력의 dimension이 큰 경우**에 차원을 맞춰주기 위해 사용
    - **Increasing dimension 부분을 모두 0**으로 채움
  - Projection shortcuts (식 $(2)$)
    - **점선**으로 표시된 shortcut connection이며, **입력보다 출력의 dimension이 큰 경우**에 차원을 맞춰주기 위해 사용
    - **linear projection $W_s$를 곱해주는 방법**으로, **$1\times1$ conv layer**를 통해 구현이 가능
- Downsampling으로 인해 입력보다 출력의 feature map 크기가 작은 경우에는, shortcut connection에 stride 2를 사용한다.

### Implementation

- 본 논문에서 ImageNet 학습에 사용한 구현 상세 내용들은 다음과 같다.
  - 256~480 범위에서 랜덤하게 추출한 값을 이미지의 가로, 세로 중 짧은 쪽(shorter side)의 길이로 하여 scale augmentation (isotropically-rescale)
  - 원본 이미지(또는 horizontal flip된 이미지)에서 각 픽셀별 평균을 빼준 후 (per-pixel mean subtracted), $224 \times 224$ 크기로 ramdom crop
  - Standard color augmentation 수행 (자세한 내용은 <a href="https://machinelearning.wtf/terms/pca-color-augmentation/" target="_blank" rel="noopener noreferrer">여기</a>를 참고)
  - Conv와 ReLU 사이에 batch normalization 수행
  - Dropout은 사용하지 않음
  - He initialization, train from scratch
  - SGD, weight decay 0.0001, momentum 0.9
  - Mini-batch size 256
  - 학습은 $60 \times 10^4$번의 iterations동안 수행
  - Learning rate는 0.1로 시작해서 error가 감소하지 않을 때마다 10으로 나눠줌 (divided by 10 when the error plateaus)

# Experiments

## ImageNet Classification

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_1.png" alt="table_1" style="border:1px solid black">
</p>

- *Table 1*은 본 논문에서 ImageNet 2012 dataset에서 실험한 architecture 구조들을 나타낸 것이다.

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_2.png" alt="table_2" width="500px" style="border:1px solid black">
</p>

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_4.png" alt="fig_4" style="border:1px solid black">
</p>

- *Table 2*와 *Figure 4*는 Plain networks와 ResNet을 18-layer, 34-layer로 학습시킨 후, 각각에 대해서 비교한 결과이다.
- Plain networks와 ResNet의 기본 구조는 *Table 1*과 같으며, 이 실험에서는 ResNet의 shortcut connection에서 dimension이 맞지 않는 경우, zero padding 방법을 사용하였다.

### Plain Networks

- *Table 2*의 결과를 보면, **plain networks**의 경우 **더 깊은 network**인 plain-34에서 **더 높은 error**을 보인다는 것을 알 수 있다.
- *Figure 4*의 결과를 보면, plain-34가 plain-18보다 더 높은 validation error을 보이며, **degradation problem**이 나타났다는 것을 알 수 있다.
- 본 논문에서는 plain networks와 ResNet 모두 batch normalization을 사용하였기 때문에, 위의 결과는 vanishing gradient로 인한 결과로 볼 수 없으며, 기존의 네트워크 구조에서는 degradation problem을 해결하지 못한다고 주장한다.

### Residual Networks

- *Table 2*와 *Figure 4*의 결과를 통해 알 수 있는 것은 다음과 같다.
  - ResNet-34는 ResNet-18보다 더 낮은 training 및 validation error를 보이며 **degradation problem이 해결**되었고, ResNet에서는 아주 깊은 네트워크여도 **깊어질수록 높은 정확도**를 얻을 수 있게 되었다.
  - ResNet-18은 plain-18보다 초기에 더 빠르게 수렴하는 모습을 보였고, 이를 통해 ResNet은 **학습 초기에 빠르게 수렴**할 수 있도록 해준다는 것을 알 수 있다. (ResNet eases the optimization by providing **faster convergence at the early stage**.)

### Identity vs. Projection Shortcuts.

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_3.png" alt="table_3" width="500px" style="border:1px solid black">
</p>

- *Table 3*은 다음의 3가지 모델에서 shortcut을 실험한 결과를 보여준다.
  - ResNet-34 A
    - 입력보다 출력의 dimension이 큰 경우에 zero-padding shortcut 사용
  - ResNet-34 B
    - 입력보다 출력의 dimension이 큰 경우에 projection shortcuts 사용
  - ResNet-34 C
    - 입출력의 dimension에 무관하게 모두 projection shortcut 사용
- *Table 3*의 결과를 통해 알 수 있는 것은 다음과 같다.
  - 위의 3가지 모델 모두 shortcut을 사용하지 않은 plain network보다 좋은 성능을 보였다.
  - ResNet-34 B가 ResNet-34 A보다 더 좋은 성능을 보였다.
    - **입력보다 출력의 dimension이 큰 경우**, zero-padding보다 **projection이 더 좋은 성능**을 보임
    - 논문에서는 이를 zero-padded 부분은 residual learning이 일어나지 않았기 때문이라고 주장
  - ResNet-34 C가 ResNet-34 B보다 약간 더 좋은 성능을 보였다.
    - 모든 shortcut connection에 projection을 사용하는 것이 가장 성능이 좋음
    - 하지만, **향상되는 성능에 비해 추가되는 파라미터의 수가 많다는 단점**이 있음
  - ResNet-34 A에서 ResNet-34 C로 갈수록 성능이 향상되기는 하지만 큰 차이가 나지 않는다.
    - 모든 shortcut에서 성능이 향상되므로 **projection shortcut이 degradation problem을 해결한 것은 아니다.**

### Deep Bottleneck Architectures

- 본 논문에서는, **ResNet-34보다 더 깊은 네트워크에서 시간복잡도가 너무 높아지지 않도록** 하기 위해, **bottleneck design**을 사용하였다.

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_5.png" alt="fig_5" width="500px" style="border:1px solid black">
</p>

- *Figure 5*는 basic residual block(왼쪽)과 bottleneck block(오른쪽)을 비교해서 나타낸 것이다.
  - Bottleneck block은 3개의 stacked layer로 구성된다.
    - $(1\times1) \rightarrow (3\times3) \rightarrow (1\times1)$
  - 1번째, 3번째 $1\times1$ conv layer는 각각 차원을 줄이고 키우는 역할을 수행한다.
    - 1번째 : $256 \rightarrow 64$
    - 3번째 : $64 \rightarrow 256$
  - 입력과 출력의 dimension이 같은 경우에는 identity shortcut을 사용한다.
    - Shortcut connection이 일어나는 양 끝단의 dimension이 높으므로, projection shortcut을 사용하면 모델의 크기가 매우 커질 수 있기 때문

#### 50-layer ResNet

- ResNet-34의 buliding block만 bottleneck으로 변경한 구조 (*Table 1*의 3번째 열)
- 입력보다 출력의 dimension이 큰 경우에만 projection shortcuts 사용
- **ResNet-34보다 성능은 증가**하였지만, **complexity는 큰 차이가 없음** (*Table 1*의 맨 아래 행 FLOPs 참고)

#### 101-layer and 152-layer ResNets

- layer_4x 부분에 더 많은 block을 쌓은 구조 (*Table 1*의 layer name 참고)
- ResNet-50보다 성능이 증가하였고, **ResNet-101에서 ResNet-152로 가면서 성능이 더 증가** (*Table 3*)
- 네트워크가 아주 깊어졌지만, **기존의 네트워크들에 비해 complexity는 여전히 낮음**
  - 가장 깊은 모델인 ResNet-152는 $11.3\times10^9$ FLOPs의 연산량을 가지는데, 이는 VGG-16보다도 낮음
    - VGG-16 : $15.3\times10^9$ FLOPs
    - VGG-19 : $19.6\times10^9$ FLOPs

#### Comparisons with State-of-the-art Methods

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_4.png" alt="table_4" width="500px" style="border:1px solid black">
</p>

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_5.png" alt="table_5" width="500px" style="border:1px solid black">
</p>

- *Table 4*와 *Table 5*는 기존의 모델들과 ResNet의 성능을 각각 single-model, ensemble에서 비교한 것이다.
  - ResNet-34에서 이미 기존의 우수한 모델들에 준하는 성능을 보인다.
  - **ResNet-152**는 **single-model에서 가장 높은 성능**을 보였으며, 또한, 이는 **기존의 우수한 네트워크들이 ensemble을 사용해서 기록한 성능보다도 더 높다.**
  - 각각 다른 깊이를 가지는 총 6개의 ResNet 모델을 ensemble하여 **ILSVRC 2015에서 1위를 기록**하였다.

## CIFAR-10 and Analysis

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/cifar_10.png" alt="cifar_10" width="500px" style="border:1px solid black">
</p>

- CIFAR-10 dataset에 실험한 모델은 총 $(1+6n) + 1 = 6n + 2$개의 stacked layer로 구성된다. (위 table 참고)
  - $1+6n$개의 $3\times3$ conv layer
    - $1+2n, 2n, 2n$개의 layer들에 대한 feature map 크기는 각각 $(32\times32), (16\times16), (8\times8)$
    - $1+2n, 2n, 2n$개의 layer들에 대한 filter 수는 각각 $16, 32, 64$
    - $(32\times32)\rightarrow(16\times16)\rightarrow(8\times8)$로의 subsampling에는 stride 2를 사용
  - Global Avearge Pooling + fully-connected layer
- 학습에 사용한 기타 구현 상세내용들은 다음과 같다.
  - $32\times32$ 크기의 per-pixel mean subtracted image를 입력으로 사용
  - $3\times3$ conv layer의 양끝단에 shortcut connection을 사용하여 총 $3n$개의 shortcuts이 있음
    - 모든 shortcut에는 identity shortcut을 사용 (projection 사용 X)
  - Batch normalization 사용
  - Dropout은 사용하지 않음
  - He initialization 사용
  - weight decay 0.0001, momentum 0.9
  - 2개의 GPU에 각각 mini-batch size 128
  - 학습은 64​k iterations동안 수행
    - Learning rate는 0.1로 시작해서 32k, 48k iterations에서 10으로 나눠줌
  - 45k/5k로 train/val split 수행
  - 4 pixel의 zero padding을 수행한 원본 이미지(또는 horizontal flip된 이미지)에서, $32\times32$ 크기로 random crop

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_6.png" alt="fig_6" style="border:1px solid black">
</p>

- *Figure 6*의 왼쪽, 중간 그래프는 plain network와 ResNet의 학습 결과를 나타낸 것이다.
  - **Plain network**는 20, 32, 44, 56 layer($n=3, 5, 6, 9$)에서 실험하였고, **네트워크가 깊어질수록 error가 높아지는 결과**를 보였다. (degradation problem O)
  - **ResNet**은 20, 32, 44, 56, 110 layer($n=3, 5, 6, 9, 18$)에서 실험하였고, **네트워크가 깊어질수록 error가 낮아지는 결과**를 보였다. (degradtion problem X)
    - ResNet-110의 경우, training error가 80% 이하로 내려갈 때(약 400 iter지점)까지 warm up learning rate 0.01로 학습한 후, 다시 0.1로 증가시켜 학습을 진행

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_6.png" alt="table_6" width="500px" style="border:1px solid black">
</p>

- *Table 6*는 CIFAR-10 test set에서의 성능을 나타낸 것이다.
  - ResNet-110의 경우, 기존의 모델들(FitNet, Highway)보다 **더 적은 파라미터로 더 높은 성능**을 기록하였다.

### Analysis of Layer Responses

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_7.png" alt="fig_7" width="500px" style="border:1px solid black">
</p>

- *Figure 7*은 각 layer의 response들에 대한 표준편차를 나타낸 것이다.

  - 여기서, response는 $3\times3$ conv 및 batch norm을 통과한 이후의 출력을 의미한다. (ReLU를 통과하기 전)
  - 아래쪽 그래프는 response들을 크기순으로 정렬하여 ResNet과 plain net의 차이를 보기 쉽게 나타낸 것이다.

- *Figure 7*의 그래프를 통해 알 수 있는 것은 다음과 같다.

  - **ResNet의 response가 plain network보다 대체로 낮은 결과**를 보이는데, 이는 residual function $\mathcal{F}(\mathbf{x}) = \mathcal{H}(\mathbf{x}) - \mathbf{x}$가 기존의 $\mathcal{H}(\mathbf{x})$보다 0에 더 가까운 출력을 낸다는 것을 의미하며, 이는 **논문에서 의도한 것과 같이 stacked layers의 weights가 0에 가깝게 학습되었다는 것을 의미**한다.

  - **ResNet의 layer수가 많아질수록 평탄한 그래프**가 나타나는데, 이는 layer가 많아질수록 ResNet의 각 layer들이 더 적은 signal을 미세하게 조정하는 것이라고 해석할 수 있다.

### Exploring Over 1000 layers

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/fig_6.png" alt="fig_6" style="border:1px solid black">
</p>

- *Figure 6*의 오른쪽 그래프는 1202개의 layer($n=200$)로 구성된 매우 깊은 네트워크에서의 실험결과이며, 이를 통해 알 수 있는 것은 다음과 같다.
  - Training error가 0.1%보다 낮게 나타나며, **네트워크의 깊이가 아주 깊어져도 수렴이 잘 된다**는 것을 확인할 수 있다. (no optimization difficulty)
  - residual-1202와 residual-110를 비교했을 때, training error는 유사하지만 **test error의 경우 residual-1202가 더 높게** 나타나는데, **논문에서는 이를 overfitting이라고 주장**하며 더 강한 regularization을 사용한다면 이를 해결할 수 있을 것이라고 이야기한다.

## Object Detection on PASCAL and MS COCO

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_7.png" alt="table_7" width="500px" style="border:1px solid black">
</p>

<p align="center">
<img src="/assets/deep_residual_learning_for_image_recognition/table_8.png" alt="table_8" width="500px" style="border:1px solid black">
</p>

- *Table 7*과 *Table 8*은 PASCAL VOC 2007 및 2012, MS COCO dataset에서 ResNet을 적용한 detection 모델을 실험한 결과이다.
  - 본 논문에서는 **Faster R-CNN의 VGG-16을 ResNet-101로 변경한 네트워크**를 사용하였으며, **다음의 4개 competition에서 모두 1위를 기록**하였다.
    - ImagNet detection
    - ImageNet localization
    - COCO detection
    - COCO segmentation
- 위 결과를 통해 **ResNet은 다른 task로의 generalization 성능도 우수**하다는 것을 알 수 있다.

