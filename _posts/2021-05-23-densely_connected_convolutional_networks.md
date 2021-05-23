---
title: "Densely Connected Convolutional Networks"
key: 20210523
sidebar:
  nav: papers-ko
tags: Papers Image&nbspClassification
---

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/densenet_title.png" alt="densenet">
</p>

이 글은 <a href="https://arxiv.org/abs/1608.06993" target="_blank" rel="noopener noreferrer">Densely Connected Convolutional Networks</a> 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

# Abstract

- 본 논문에서는 각 layer들이 모든 preceding layer들과 연결된 구조의 Dense Convolutional Network (DenseNet)을 제안한다.
- DenseNet architecture는 다음과 같은 장점이 있다.
  - Alleviate the vanishing gradient problem
  - Strengthen feature propagation
  - Encourage feature reuse
  - Substantially reduce the number of parameters
- DenseNet은 image recognition benchmarks(CIFAR-10, CIFAR-100, SVHN, ImageNet)에서 적은 연산량으로 SOTA의 성능을 보였다.

# DenseNets

- 본 논문에서는, 다음의 CNN 구조에서 ResNet과 비교하며 DenseNet의 구조를 설명한다.
  - 네트워크는 총 $L$개의 layer로 구성
  - 각 $\ell$번째 layer는 non-linear transformation $H_\ell(\cdot)$을 수행
    - $H_\ell(\cdot)$은 Conv, Pool, Batch Normalization(BN), ReLU, 등으로 구성
  - 입력 이미지 $\mathbf{x}_0$
  - 각 $\ell$번째 layer의 출력 $\mathbf{x}_{\ell}$

## ResNet

- ResNet의 $\ell$번째 layer의 출력은 다음과 같이 나타낼 수 있다.

$$
\mathbf{x}_\ell = H_\ell (\mathbf{x}_{\ell-1}) + \mathbf{x}_{\ell-1} \tag{1}
$$

- ResNet은 identity function을 통해 **later layer로부터 early layer로의 gradient flow가 직접 연결된다는 장점**이 있지만, **identity function과 출력 $H_\ell (\mathbf{x}_{\ell-1})$이 summation됨에 따라 information flow를 방해**할 수 있다.
  - 즉, gradient가 흐르게 된다는 점은 도움이 되지만, forward pass에서 보존되어야 하는 정보들이 summation을 통해 변경되어 보존되지 못할 수 있다는 의미이다. (DenseNet의 경우, concatenation을 통해 그대로 보존)

## DenseNet

### Dense connectivity

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/fig_1.png" alt="fig_1" width="600px" style="border:1px solid black">
</p>

- *Figure 1*은 DenseNet의 connectivity pattern을 보여주며,  **$\ell$번째 layer의 출력**은 다음과 같이 나타낼 수 있다.

$$
\mathbf{x}_\ell = H_\ell ([\mathbf{x}_0, \mathbf{x}_1, \cdots, \mathbf{x}_{\ell-1}]) \tag{2}
$$

- 위 식으로부터, DenseNet의 **$\ell$번째 layer는 모든 preceding layers($0, 1, \cdots, \ell-1$번째 layer)로부터 생성된 feature map들의 concatenation $[\mathbf{x}_0, \mathbf{x}_1, \cdots, \mathbf{x}\_{\ell-1}\]$을 입력**으로 받는다는 것을 알 수 있다.


### Composite function & Pooling layers

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/fig_2.png" alt="fig_2" style="border:1px solid black">
</p>

- **DenseNet은 여러 개의 *dense block*과 이들 사이에서 down-sampling을 수행하는 *transition layers*로 구성**된다. (*Figure 2*)
- Dense block 내의 각 layer들은 다음의 3가지 연산을 연달아 수행하는 composite function $H_\ell(\cdot)$로 정의한다.
  - Batch Normalization(BN)
  - ReLU
  - $3\times3$ conv
- Transition layer는 down-sampling을 위한 pooling layer를 포함해 다음과 같이 구성된다.
  - Batch Normalization(BN)
  - $1\times1$ conv
  - $2\times2$ average pooling

### Growth rate

- **DenseNet에서는 *growth rate*라고 부르는 hyperparameter $k$**가 있는데, 이는 다음과 같이 **출력 feature map의 수를 조절**하는 역할을 한다.
  - DenseNet의 $\ell$번째 layer 입출력 feature map 수는 다음과 같다. (여기서 $k_0$는 입력 이미지의 channel 수를 의미)
    - 입력 : $k_0 + k \times (\ell-1)$
    - 출력 : $k$
- DenseNet은 아주 작은 $k$를 사용하여 아주 얕은 layer들로 구성되는데, 논문에서는 **아주 얕은 layer로 구성해도 DenseNet이 좋은 성능을 보이는 이유**를 다음과 같이 설명한다.
  - Dense block내에서 각 layer들은 모든 preceding feature map에 접근할 수 있는데, 이는 네트워크의 "collective knowledge"에 접근한다는 것을 의미한다.
    - 즉, **preceding feature map들을 네트워크의 global state로써 바라볼 수 있다**는 것을 의미
  - 따라서, growth rate $k$는 각 layer가 global state에 얼마나 많은 새로운 정보를 contribute할 것인지를 조절하는 역할을 한다고도 볼 수 있다.
- **모든 layer가 접근할 수 있는 global state로 인해 DenseNet은** 기존의 네트워크들과 같이 **layer의 feature map을 복사해서 다른 layer로 넘겨주는 등의 작업을 할 필요가 없다**는 장점이 있다.

### Bottleneck layers

- **DenseNet에서도 입력보다 출력의 feature map 수를 조절하는 *bottleneck layer*를 사용**하였다.
- 본 논문에서는 $H_\ell$에 다음과 같은 구조의 bottleneck layer를 사용한 모델을 **DenseNet-B**라고 부른다.
  - Batch Norm $\rightarrow$ ReLU $\rightarrow$ Conv ($1 \times 1$) $\rightarrow$ Batch Norm $\rightarrow$ ReLU $\rightarrow$ Conv ($3\times3$)
  - 이때, 각 $1\times1$ Conv는 $4k$개의 feature map을 출력한다.

### Compression

- **DenseNet에서는 model compactness를 위해 transition layer에서 feature map의 수를 줄이는 방법을 사용**한다.
  - Dense block으로부터 $m$개의 feature map을 입력받았을 때, transition layer는 $\|\theta m\|$개의 feature map을 출력한다.
  - 이때, **feature map 수를 조절하는 값 $\theta$를 compactness factor**라고 부르며 $0 < \theta \le 1$의 크기를 가진다.
    - $\theta = 1$인 경우, transition layer는 feature map수를 줄이지 않고 그대로 유지
- 본 논문에서는 $\theta < 1$을 사용한 모델을 **DenseNet-C**라고 부르며(실험에서는 $\theta = 0.5$의 값을 사용), bottleneck layer와 $\theta < 1$의 transition layer를 모두 사용한 모델을 **DenseNet-BC**라고 부른다.

### Implementation Details

- 본 논문에서 DenseNet을 CIFAR-10, CIFAR-100, SVHN 데이터셋에서 실험할 때의 구현 상세들은 다음과 같다.
  - 1번째 conv layer
    - 입력 이미지를 받아 16개의 featuer map을 출력 (DenseNet-BC에서는 $2k$개를 출력)
  - Dense block
    - 총 3개의 dense block을 사용
      - 각각 $32 \times 32$, $16 \times 16$, $8 \times 8$ 크기의 feature map을 출력 (즉, output size를 의미)
    - Zero padding 1 pixel
  - Transition layer
    - $1 \times 1$ conv layer
    - $2\times2$ average pooling
  - Final layer
    - Global average pooling
    - Fully connected layer (softmax classifier)
- 본 논문에서는, CIFAR-10, CIFAR-100, SVHN 데이터셋에 대해 기본 DenseNet과 DenseNet-BC의 2가지 구조를 실험하였고, 각각에 사용한 설정들은 다음과 같다.
  - DenseNet
    - $\{L=40, k=12\}$
    - $\{L=100, k=12\}$
    - $\{L=100, k=24\}$
  - DenseNet-BC
    - $\{L=100, k=12\}$
    - $\{L=250, k=24\}$
    - $\{L=190, k=40\}$

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/table_1.png" alt="table_1" style="border:1px solid black">
</p>

- DenseNet을 ImageNet에서 실험한 구현 상세들은 다음과 같다. (*Table 1*)
  - 1번째 conv layer
    - 입력 이미지를 받아 $2k$개의 feature map을 출력
    - $7 \times 7$ conv layer, stride 2
  - 총 4개의 dense block을 사용
  - 나머지 구현 상세들은 *Table 1*과 같으며, $k$에 따라 약간씩 달라지게 된다.
- 본 논문에서는, ImageNet 데이터셋에 대해서 DenseNet-BC 구조만을 사용하였다.

# Experiments

### CIFAR

- CIFAR dataset은 $32\times32$ 크기의 이미지 데이터셋이며, 본 논문에서는 CIFAR-10, CIFAR-100 모두에 대해 실험하였다.
  - Training set과 test set은 각각 50000, 10000개의 데이터를 사용하였고, validation set은 training set에서 hold out한 5000개의 데이터를 사용하였다.
  - Data augmentation에는 CIFAR에서 주로 사용되는 다음의 방식을 사용하였다.
    - mirroring / shifting
  - 데이터는 channel mean, standard deviation을 통해 normalize하였다.
- *Table 2*에서는 CIFAR-10, CIFAR-100을 각각 C10, C100으로 표기하였고, data augmentation을 사용한 경우에는 "+"를 추가로 표기하였다.

### SVHN

- Street View House Numbers (SVHN) dataset은 $32\times32$ 크기의 숫자 이미지로 구성된다.
  - Data augmentation은 사용하지 않았고, validation set은 training set에서 6000개의 데이터를 hold out하여 사용하였다.
  - 데이터는 pixel value를 255로 나누어 0~1로 normalize하였다.
- *Table 2*의 결과는 validation error가 가장 적은 모델에서 test set으로 평가한 결과이다.

### ImageNet 

- ImageNet은 ILSVRC 2012 classification dataset을 의미한다.
  - Training에는 ResNet에서와 동일한 data augmentation을 사용하였고, test에는 single-crop 또는 10-crop을 사용하였다.
- 본 논문에서는 validation set에서의 classification error를 사용해 성능을 측정하였다.

## Training

- 각 Dataset에 대한 학습 관련 설정들은 다음과 같다.
  - CIFAR, SVHN, ImageNet
    - SGD (stochastic gradient descent)
    - weight decay $10^{-4}$
    - nesterov momentum 0.9 without dampening
    - He initialization
  - CIFAR, SVHN
    - 64의 batch size를 사용해 각각 300, 40 epoch동안 학습
    - Learning rate는 0.1로 시작하여 epoch의 50%, 75% 지점에서 0.1을 곱해주는 방식을 사용
    - Data augmentation을 사용하지 않는 경우, 각 conv layer 이후에 dropout(rate=0.2) 추가 (1번째 conv layer는 제외)
  - ImageNet
    - 256의 batch size를 사용해 90 epoch동안 학습
    - Learning rate는 0.1로 시작하여 30, 60 epoch에서 0.1을 곱해주는 방식을 사용

## Classification Results on CIFAR and SVHN

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/table_2.png" alt="table_2" style="border:1px solid black">
</p>

- CIFAR, SVHN dataset에 대한 결과는 *Table 2*와 같다.

### Accuracy

- *Table 2*의 결과를 통해, DenseNet-BC가 각 dataset에서 가장 좋은 성능을 보였다는 것을 알 수 있다.
  - DenseNet-BC($L=190, k=40$)는 C10+, C100+에서 가장 좋은 성능을 보인다.
  - DenseNet-BC($L=100, k=24$)는 C10, C100, SVHN에서 가장 좋은 성능을 보인다.

### Capacity

- *Table 2*에서, DenseNet의 C10+에서의 결과를 보면 다음과 파라미터 수가 늘어날수록 성능이 증가한다는 것을 알 수 있다.
  - Error : 5.24% $\rightarrow$ 4.10% $\rightarrow$ 3.74%
  - Number of parameters : 1.0M $\rightarrow$ 7.0M $\rightarrow$ 27.2M
- 위와 같은 변화는 C100+에 대해서도 유사하게 나타나며, 이를 통해 알 수 있는 **DenseNet의 특징**은 다음과 같다.
  - **모델이 더 크고 더 깊어질수록 더 많고 풍부한 representation을 학습할 수 있다.**
  - **Overfitting이나 optimization difficulty가 나타나지 않았다.**

### Parameter Efficiency & Overfitting

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/fig_4.png" alt="fig_4" style="border:1px solid black">
</p>

- *Figure 4*의 왼쪽, 가운데 그래프는 parameter efficiency를 각각 DenseNet의 variants, ResNet과 비교한 결과를 보여준다.
  - **DenseNet중에서는 DenseNet-BC가 가장 parameter efficiency가 좋다.**
  - **DenseNet은 ResNet보다 parameter efficiency가 좋다.**
    - *Table 2*에서 FractalNet, Wide ResNet과 비교해도 적은 파라미터로 더 높은 성능을 보임
- *Figure 4*의 오른쪽 그래프는 ResNet-1001과 DenseNet-BC($L=100, k=12$)의 error를 비교한 것이다.
  - ResNet-1001은 DenseNet-BC에 비해 training loss는 더 낮지만, test error는 비슷한 것을 알 수 있는데, 이는 **DenseNet이 ResNet보다 overfitting이 일어나는 경향이 더 적다**는 것을 보여준다.

## Classification Results on ImageNet

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/table_3.png" alt="table_3" width="400px" style="border:1px solid black">
</p>

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/fig_3.png" alt="fig_3" width="700px" style="border:1px solid black">
</p>

- *Table 3*은 DenseNet의 ImageNet에서의 single crop, 10-crop validation error를 나타낸 것이다.
- *Figure 3*는 DenseNet과 ResNet의 single crop top-1 validation error를 파라미터수와 flops를 기준으로 나타낸 것이다.
- 위의 두 결과를 통해, **DenseNet은 ResNet에 비해 현저히 적은 파라미터와 연산량으로도 동등한 성능을 보인다**는 것을 알 수 있다.

# Discussion

### Model compactness

- DenseNet은 **preceding layer들의 feature map에 접근**할 수 있는 구조를 통해 **feature reuse를 encourage**하고, 이는 **model이 더 compact**해지도록 한다.
- 다음의 여러 실험을 통한 결과들은 DenseNet-BC가 ResNet에 비해 더 높은 parameter efficiency를 가진다는 것을 보여준다.
  - *Table 2*에서 다른 네트워크들에 비해 DenseNet이 적은 파라미터로 더 우수한 성능을 보인다는 것을 확인할 수 있다.
  - *Figure 2*의 왼쪽 그래프를 통해 DenseNet-BC가 가장 parameter efficiency가 좋다는 것을 확인할 수 있다.
  - *Figure 2*의 가운데 그래프를 통해 DenseNet-BC는 $\frac{1}{3}$의 파라미터만으로도 ResNet과 동일한 성능을 낸다는 것을 확인할 수 있다.
  - *Figure 3*의 그래프를 통해 DenseNet-BC가 ResNet에 비해 현저히 적은 파라미터와 연산량으로도 동등한 성능을 보인다는 것을 확인할 수 있다.
  - *Figure 4*의 오른쪽 그래프를 통해 DenseNet-BC는 0.8M의 파라미터수만으로도 10.2M의 파라미터를 가지는 ResNet-1001과 동등한 성능을 낼 수 있다는 것을 확인할 수 있다.

### Implicit Deep Supervision

- 본 논문에서는, **DenseNet이 우수한 성능을 보이는 이유를 다음과 같은 implicit deep supervision으로 인한 결과로 해석할수도 있다**고 이야기한다.
  - 마지막 classifier는 transition layer들을 통해 모든 layer들로 direct supervision을 한다.

### Stochastic vs. deterministic connection

- 본 논문에서는, 다음과 같이 stochastic depth와 DenseNet은 비슷한 connectivity pattern이 존재한다고 이야기하며, **stochastic depth를 DenseNet의 관점에서 해석할수도 있다**고 이야기한다.
  - Stochastic depth에서는, 무작위로 일부 layer를 drop하고 이들을 둘러싸고 있던 layer끼리 직접적으로 연결된다.
  - DenseNet에서는, 아주 낮은 확률이지만 dense block내의 어떠한 2개 layer 사이의 모든 layer가 drop될 경우, 해당 2개 layer는 직접적으로 연결된다.

### Feature Reuse

<p align="center">
<img src="/assets/densely_connected_convolutional_networks/fig_5.png" alt="fig_5" width="600px" style="border:1px solid black">
</p>

- 본 논문에서는, **학습된 DenseNet의 각 layer가 실제로 preceding layer들의 feature map을 활용하는지를 실험**하였다.
- *Figure 5*의 heatmap을 그린 방법은 다음과 같다.
  - C10+에서 DenseNet($L=40, k=12$)을 학습한다.
  - 학습한 네트워크의 각 dense block에서, **$\ell$번째 conv layer에서 $s$번째 layer로의 할당된 average absolute weight를 계산**한다. (absolute는 음의 값을 갖는 weight를 고려한 것으로 보임)
    - Ex) $\ell=3$에서 $s=2$에 대한 average weight를 계산한다고 하면, $\mathbf{x}\_3 = H_\ell([\mathbf{x}_1, \mathbf{x}_2])$을 계산할 때, $\mathbf{x}_2$에 할당된 weight의 average absolute 값을 계산
- ***Figure 5*를 통해** 관찰할 수 있는 것과 이를 통해 **알 수 있는 것**은 다음과 같다.
  - 각 layer들이 preceding layer들에 weight를 spread하고 있다. (각 열에서 weight가 골고루 spread되어 있음)
    - Dense block내에서, **실제로 later layer는 early layer의 feature map을 사용**하고 있다.
  - Transition layer도 preceding layer들에 weight를 spread하고 있다. (가장 오른쪽 열에서 weight가 골고루 spread 되어 있음)
    - Dense block내에서, **1번째 layer에서 가장 마지막 layer까지 information flow가 형성**되어 있다.
  - 2, 3번째 dense block은 transition layer의 output에 매우 적은 weight를 할당한다. (2, 3번째 dense block의 첫번째 행에서 weight가 거의 0에 가까움)
    - **2, 3번째 dense block의 transition layer output**은 redundant features가 많아서 매우 적은 weight를 할당한 것이며(**중복된 정보들이 많아 모두 사용하지 않아도 된다는 의미**), 이는 **DenseNet-BC에서 compression factor $\theta$로 이러한 redundant feature들을 compress하는 것이 reasonable하다**는 것을 보여준다.
  - 마지막 classification layer는 early layer보다 later layer의 feature map을 더 많이 사용한다. (3번째 dense block의 가장 마지막 열에서 weight가 아래쪽으로 치우쳐 있음)
    - **High-level feature가 later layer에 더 많이 존재**한다.

# Conclusion

- 본 논문에서는, 동일한 feature map 크기를 가지는 **layer간의 direct connection 구조를 가지는 Dense Convolutional Network (DenseNet)을 제안**한다.
- DenseNet은 **optimization difficulty없이** 수백개의 layer를 가지는 네트워크로 확장할 수 있다.
- DenseNet은 파라미터 수의 증가에 따라, **degradation problem이나 overfitting이 나타나지 않고**, 일관되게 성능이 향상된다.
- DenseNet은 **여러 dataset에서 적은 파라미터수와 연산량으로도 SOTA의 성능을 기록**하였다.
- DenseNet의 **connnectivity rule은 identity mapping, deep supervision, diversified depth의 효과들을 통합**한다.