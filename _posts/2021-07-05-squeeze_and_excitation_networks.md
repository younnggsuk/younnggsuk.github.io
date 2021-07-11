---
title: "Squeeze-and-Excitation Networks"
key: 20210705
sidebar:
  nav: papers-ko
tags: Papers Image&nbspClassification
---

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/senet_title.png" alt="senet">
</p>

이 글은 <a href="https://arxiv.org/abs/1709.01507" target="_blank" rel="noopener noreferrer">Squeeze-and-Excitation Networks</a> 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

본 글을 바탕으로 직접 구현한 코드는 <a href="https://github.com/younnggsuk/CV-Paper-Implementation/tree/main/squeeze_and_excitation_networks" target="_blank" rel="noopener noreferrer">이곳</a>에서 확인하실 수 있습니다.
{:.success}

# Abstract

- 본 논문에서는 feature channel간의 interdependencies를 explicitly modeling하여 channel-wise feature response를 adaptively recalibrate하는 Squeeze-and-Excitation (SE) block을 제안한다.
- SE block을 적용하면, 약간의 computational cost만으로 성능을 크게 향상시킬 수 있다.
- SENet은 ILSVRC 2017 classification에서 1위를 차지하였다.

# Sqeeze-and-Excitation Blocks

- 입력 $\mathbf{X}$를 받아서 feature map $\mathbf{U}$로 mapping하는 transformation을 $\mathbf{F}_{tr}$이라고 하자.
- $\mathbf{F}_{tr}$을 convolution이라고 본다면, 입력 $\mathbf{X}$가 filter $\mathbf{V}$를 통해 output feature map $\mathbf{U}$로 계산될 때, output feature map의 $c$번째 channel $\mathbf{u}_c$는 식 (1)과 같이 나타낼 수 있다.
  - $\mathbf{F}_{tr}$ : transformation (여기서는 convolution)
  - $\mathbf{X} \in \mathbb{R}^{H' \times W' \times C'}$ : input
  - $\mathbf{U} \in \mathbb{R}^{H \times W \times C}$ : output feature map
  - $\mathbf{V} = [\mathbf{v}_1, \mathbf{v}_2, \cdots, \mathbf{v}_c]$ : filter kernels
  - $*$ :  convolution operator

$$
\mathbf{u}_c = \mathbf{v}_c * \mathbf{X} = \sum^{C'}_{s=1} \mathbf{v}_c^s * \mathbf{x}^s  \tag{1}
$$

- 여기서, $\mathbf{v}^s_c$는 $\mathbf{v}_c$의 각 channel들을 의미하며, 이들은 해당하는 입력의 각 채널 $\mathbf{x}^s$와 컨볼루션 연산을 수행한다. 그리고 이 결과 채널들이 모두 더해져서 출력의 한 채널 $\mathbf{u}_c$을 구성하게 된다. 
  - $\mathbf{v}_c = [\mathbf{v}_c^1, \mathbf{v}_c^2, \cdots, \mathbf{v}_c^{C'}]$
  - $\mathbf{X} = [\mathbf{x}^1, \mathbf{x}^2, \cdots, \mathbf{x}^{C'}]$
  - $\mathbf{u}_c \in \mathbb{R}^{H \times W}$
- 앞서 살펴본 convolution 연산은 output의 계산 과정에서 모든 결과 채널들을 더하기 때문에, channel dependency는 $\mathbf{v}_c$에 implicit하게 embed되며 filter가 capture한 local spatial correlation과도 복잡하게 얽혀 있게 된다.
  - 즉, convolution 연산을 통해 모델링된 channel dependency 정보는 implicit하고 local하다고 볼 수 있다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/fig_1.png" alt="fig_1" style="border:1px solid black">
</p>

- 본 논문에서는, 이러한 channel dependency를 emplicit하게 모델링한다면 네트워크가 feature들을 더 잘 학습할 수 있을 것이라고 보았고, 이를 다음의 2단계를 거치는 SE block을 통해 구현하였다. (*Figure 1*)
  - *squeeze* : global information에 접근할 수 있도록 한다.
  - *excitation* : filter의 response를 recalibrate한다.

## Squeeze: Global Information Embedding

- Convolutional layer의 각 filter는 local receptive field로 연산하기 때문에 contextual information을 exploit하기 어렵다는 문제가 있다.
- 따라서, *squeeze*에서는 global average pooling을 통해 channel-wise statistics $\mathbf{z} \in \mathbb{R}^C$를 계산하여 이러한 문제를 완화하고자 하였다.
  - $\mathbf{U}$를 $\mathbf{z}$로 squeeze한다고 할 때, $c$번째 element $z_c$는 식 (2)를 통해 계산된다.

$$
z_c = \mathbf{F}_{sq}(\mathbf{u_c}) = \dfrac{1}{H \times W} \sum^H_{i=1} \sum^W_{j=1} u_c(i, j) \tag{2}
$$

## Excitation: Adaptive Recalibration

- 앞서 계산된 $\mathbf{z}$를 사용해 channel-wise dependency를 capture하기 위해서는 다음의 2가지를 학습할 수 있는 함수가 필요했다.
  - 채널간의 non-linear interaction
  - 채널간의 non-mutually-exclusive relationship (one-hot activation이 아니라, 여러 채널들이 강조될 수 있어야 하기 때문)
- 따라서, *excitation*에서는 sigmoid와 relu를 사용한 식 (3)을 사용하였다. 

$$
\mathbf{s} = \mathbf{F}_{ex}(\mathbf{z}, \mathbf{W}) = \sigma(\mathbf{W}_2\delta(\mathbf{W}_1, \mathbf{z})) \tag{3}
$$

- 이때, 2개의 fully-connected (FC) layer는 reduction ratio $r$값을 통해 차원을 감소시켰다가 증가시키는 bottleneck layer로 동작한다. (모델의 complexity가 너무 커지는 것을 제한하고 일반화 성능을 높이기 위해)
  - $\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r} \times C}$ : $C$차원에서 $\frac{C}{r}$차원으로 차원을 감소
  - $\mathbf{W}_2 \in \mathbb{R}^{C \times \frac{C}{r}}$ : $\frac{C}{r}$차원에서 $C$차원으로 차원을 증가

- 마지막으로, *excitation*을 통해 구한 $\mathbf{s}$로 $\mathbf{U}$를 rescaling하게 되면, SE block의 최종 출력 $\stackrel{\thicksim}{\mathbf{X}} = [\stackrel{\sim}{\mathbf{x}}_1, \stackrel{\sim}{\mathbf{x}}_2, \cdots, \stackrel{\sim}{\mathbf{x}}_C]$가 된다.
  - 각 채널별 계산 과정은 식 (4)와 같으며, $\mathbf{F}_{scale}(\mathbf{u}_c, s_c)$는 스칼라 값 $s_c$와 feature map $\mathbf{u}_c \in \mathbb{R}^{H \times W}$의 channel-wise multiplication을 의미한다.

$$
\stackrel{\sim}{\mathbf{x}}_c = \mathbf{F}_{scale}(\mathbf{u}_c, s_c) = s_c \mathbf{u}_c \tag{4}
$$

## Instantiations

- SE block은 convolution 이후에 바로 적용될 수 있기 때문에, 다른 architecture들에 통합될 수 있다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/fig_2_3.png" alt="fig_2_3" style="border:1px solid black">
</p>

- *Figure 2, 3*은 각각 Inception module과 Residual module에 SE block을 적용한 예를 보여준다. (위 2가지 네트워크 외에도 비슷한 구조로 적용 가능)
  - Inception module의 경우, 전체 출력에 SE block을 적용
  - Residual module의 경우, non-identity branch에 SE block을 적용

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_1.png" alt="table_1" style="border:1px solid black">
</p>

- *Table 1*은 SE block을 적용한 구체적인 예로, SE-ResNet-50, SE-ResNeXt-50을 나타낸 것이다.

# Model and Computational Complexity

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_2.png" alt="table_2" style="border:1px solid black">
</p>

- *Table 2*에서 $224 \times 224$ 크기의 이미지를 입력으로 받아 single forward pass를 할 경우의 연산량과 성능을 비교한 결과는 다음과 같다. (SE block의 reduction ratio $r$ 값은 16을 사용)
  - SE block을 적용한 경우, 적용하지 않은 네트워크에 비해 0.01~0.02 GFLOPs정도의 아주 적은 연산량 추가만으로도, 한단계 더 깊은 네트워크와 유사한 성능을 보였다.
  - 따라서, 연산량 추가에 비해 성능 이득이 큰 것을 알 수 있다.
- SE block으로 인해 추가되는 파라미터는 *excitation*의 2개 FC layer가 전부이며, 구체적인 파라미터 수는 식 (5)를 통해 계산할 수 있다.
  - $r$ : reduction ratio
  - $S$ : stage 수 (여기서, stage는 동일한 크기($W, H$)의 feature map에 동작하는 block들을 의미)
  - $C_s$ : output channel의 차원
  - $N_s$ : stage $s$에서 반복되는 block의 수

$$
\dfrac{2}{r} \sum^S_{s=1} N_s \cdot {C_s}^2 \tag{5}
$$

- SE-ResNet-50의 경우, 약 25 million개의 파라미터 수를 가지는 ResNet-50에 비해 약 2.5 million개(10% 정도)의 파라미터 수가 증가하였다.
  - 증가하는 대부분의 파라미터가 위치하는 네트워크의 final stage에서 SE block을 제거하여도 이로 인한 성능 증가의 감소는 미미하기 때문에, 메모리에 제약을 받는 상황에서도 적절히 SE block을 적용하면 좋은 성능을 낼 수 있다.

# Experiments

## Image Classification

- 본 논문에서는 ImageNet 2012 dataset을 사용해 SE block이 성능에 미치는 영향을 평가하였다.
  - 학습은 train set을 사용하였고, 평가에는 top-1, top-5 error를 측정하였다.
- 논문에 명시된 실험 상세 내용들은 다음과 같다.
  - Each baseline network and its corresponding SE counterpart are trained with identical optimization schemes
  - We follow Standard practices and perform data augmentation
    - random cropping using scale and aspect ratio ($224 \times 224$ or $299 \times 299$)
    - random horizontal flipping
  - Each input image is normalized through mean RGB-channel subtraction
  - Optimization is performed using synchronous SGD (all models are trained on distributed learning system *ROCS*)
    - momentum 0.9
    - mini-batch size of 1024
  - Initial learning rate is set to 0.6
    - decreased by a factor of 10 every 30 epochs
  - Models are trained for 100 epochs from scratch
    - using He initialization
  - Reduction ratio $r$ is set to 16 by default (except where stated otherwise)
  - When evaluating the models
    - apply centre-cropping so that $224 \times 224$ or $299 \times 299$ pixels are cropped from each image, after its shorter edge is first resized to $256$ or $352$

### Network depth

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_2.png" alt="table_2" style="border:1px solid black">
</p>

- *Table 2*에서 네트워크의 깊이에 따라 SE-ResNet과 ResNet을 비교한 결과는 다음과 같다.
  - 모든 depth에서 SE block은 매우 적은 computational complexity의 증가만으로 성능을 향상시켰다.
    - SE-ResNet50은 더 깊은 네트워크인 ResNet-101에 준하는 성능을 냈으며, 이로인해 거의 절반 수준의 연산량으로 깊은 네트워크의 성능을 낼 수 있게 되었다. (3.87 GFLOPs vs. 7.58 GFLOPs)
- 위의 결과로 인해 본 논문에서 이야기하는 SE block이 네트워크에 미치는 영향은 다음과 같다.
  - SE block은 연산량 측면에서 아주 효율적이다. (더 얕은 네트워크에서 더 깊은 네트워크의 성능을 낼 수 있음)
  - SE block은 단순한 네트워크의 깊이 증가로 인한 성능 개선을 보완할 수 있다. (추가로 더 성능을 향상시킬 수 있음)

### Integration with modern architectures

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_1.png" alt="table_1" style="border:1px solid black">
</p>

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_2.png" alt="table_2" style="border:1px solid black">
</p>

- *Table 2*에서 Inception-ResNet-v2와 ResNeXt($32 \times 4$d)에서 SE block을 적용하였을 때의 결과는 다음과 같다. (적용한 각 모델은 SE-Inception-ResNet-v2와 SE-ResNeXt이며, SE-ResNeXt의 구조는 *Table 1*과 같다.)
  - SE-ResNeXt-50은 더 깊은 네트워크인 ResNeXt-101보다 더 우수한 성능을 보였고, 이로인해 거의 절반 수준의 연산량과 파라미터수로 깊은 네트워크보다 우수한 성능을 낼 수 있게 되었다. 
  - SE-Inception-ResNet-v2는 Inception-ResNet-v2보다 더 우수한 성능을 보였다.

### Non-residual networks

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_2.png" alt="table_2" style="border:1px solid black">
</p>

- Residual network가 아닌 네트워크에서 SE block이 미치는 영향을 실험한 결과는 다음과 같다. (*Table 2*의 VGG-16와 BN-Inception)
  - Residual baseline들과 마찬가지로, VGG-16과 BN-Inception 모두에서 SE block은 성능을 향상시켰다.

### Insights into influence of SE blocks

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/fig_4.png" alt="fig_4" style="border:1px solid black">
</p>

- 앞서 실험한 네트워크들에서 SE block이 어떠한 영향을 미쳤는지에 대한 insight를 얻기 위해 training curve를 살펴본 결과는 다음과 같다.
  - 전반적인 optimization 과정에서, SE block을 사용한 경우가 사용하지 않은 경우에 비해 성능이 꾸준하게 개선된다.

### Mobile setting

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_3.png" alt="table_3" style="border:1px solid black">
</p>

- *Table 3*은 mobile-optimized network인 MobileNet과 ShuffleNet에서 SE block을 적용한 결과이다.
  - MobileNet과 ShuffleNet 모두에서 매우 적은 연산량 증가만으로 성능을 크게 향상시켰다.

### Additional datasets

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_4_5.png" alt="table_4_5" width="450px" style="border:1px solid black">
</p>

- 본 논문에서는 ImageNet이 아닌 다른 데이터셋으로 CIFAR-10과 CIFAR-100 datasets을 사용해 classification 성능을 평가하였다. (*Table 4, 5*)
  - 실험한 모든 네트워크에서 SE block은 성능을 향상시켰으며, 이는 SE block이 ImageNet이 아닌 다른 dataset에서도 성능을 향상시킨다는 것을 보여준다.

## Scene Classification

- 본 논문에서는 Places365-Challenge dataset을 사용해 scene classification task에서도 SE block을 실험하며, SENet의 generalization ability를 평가하였다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_6.png" alt="table_6" width="400px" style="border:1px solid black">
</p>

- *Table 6*은 Places 365 validation set에서 ResNet-152를 baseline으로 실험한 결과이다.
  - SE-ResNet-152가 baseline인 ResNet-152보다 우수한 성능을 보였다.
  - SE-ResNet-152가 기존의 SOTA 모델인 Places-365-CNN보다 우수한 성능을 보였다.

## Object Detection on COCO

- 본 논문에서는 COCO dataset을 사용해 object detection task에서도 SE block을 실험하며, SENet의 generalization ability를 평가하였다.
  - *minival* protocol을 사용해 validation set의 데이터를 추가로 학습에 사용하였다.
    - 학습 : $80$k training set + $35$k val subset
    - 평가 : $5$k val subset

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_7.png" alt="table_7" width="450px" style="border:1px solid black">
</p>

- *Table 7*은 Faster R-CNN의 trunk architecture(ResNet)만을 SE-ResNet으로 변경하였을 때의 성능을 비교한 것이다.
  - ResNet-50과 101 모두에서 SE block을 적용한 경우 성능이 더 향상되었으며, 이는  SE block으로 인해 detector에 전달되는 representation이 더 좋아졌다고 해석할 수 있다. (SE block을 적용한 것 외에는 다른 변경사항이 없었으므로)

## ILSVRC 2017 Classification Competition

- SENet은 ILSVRC classification competition에서 1위를 차지하였다.
  - submission에는 ResNeXt를 수정한 모델에 SE block을 적용한 SENet-154를 사용하였다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_8.png" alt="table_8" width="450px" style="border:1px solid black">
</p>

- *Table 8*은 ImageNet validation set에서 SENet-154와 다른 모델들의 성능을 비교한 것이다.
  - SENet-154가 가장 우수한 성능을 보였다.

# Ablation Study

- 본 논문에서는 SE block의 여러 구성에 따른 효과를 이해하기 위한 ablation study를 진행하였으며, 다음의 조건에서 수행하였다.
  - ResNet-50을 backbone으로 사용해 ImageNet dataset으로 학습
  - Excitation의 FC layer에는 bias를 사용하지 않음 (본 논문에서는 실험을 통해 excitation의 FC layer에 bias는 사용하지 않는 것이 channel dependency를 modeling하는데 더 용이하다고 판단하였다.)

## Reduction ratio

- Reduction ratio $r$은 SE block의 capacity와 computational cost간의 trade-off를 조절하는 hyperparameter이다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_10.png" alt="table_10" width="450px" style="border:1px solid black">
</p>

- *Table 10*은 SE-ResNet-50에서 여러 $r$값을 실험한 결과이다.
  - complexity를 높일수록 성능이 단조증가(monotonically)하지는 않는다.
  - $r$이 작을수록, 모델 파라미터수는 크게 줄어든다.
- 따라서, 본 논문에서는 $r=16$을 capacity와 computational cost간에 balance를 잘 이룬 값이라고 판단하여 사용하였다.

## Squeeze Operator

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_11.png" alt="table_11" width="450px" style="border:1px solid black">
</p>

- *Table 11*은 Squeeze 연산에 global max pooling과 global average pooling을 비교한 결과이다.
  - Global average pooling이 약간 더 좋은 성능을 보였다.

## Excitation Operator

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_12.png" alt="table_12" width="450px" style="border:1px solid black">
</p>

- *Table 12*는 excitation에서 사용되는 non-linearity function들의 성능을 비교한 결과이다.
  - Sigmoid가 가장 좋은 성능을 보였다.
  - SE block에서 excitation operator는 아주 중요한 역할을 한다. (ReLU를 사용할 경우, ResNet-50 baseline보다도 성능이 나빠진다.)

## Different stages

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_13.png" alt="table_13" width="450px" style="border:1px solid black">
</p>

- *Table 13*은 ResNet-50의 각 stage에 SE block을 추가하였을 때(one stage at a time), 성능에 미치는 영향을 실험한 것이다.
  - SE block을 각 stage에 추가한 모든 경우(SE_Stage_2, SE_Stage_3, SE_Stage_4)에서 성능이 향상되었다.
  - SE block을 모든 stage에 추가하였을 때(SE_ALL) 성능이 더욱 향상되었으므로, 각각으로 인한 성능의 향상은 complementary한 관계이다.

## Integration strategy

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/fig_5.png" alt="fig_5" style="border:1px solid black">
</p>

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_14.png" alt="table_14" width="450px" style="border:1px solid black">
</p>

- *Table 14*는 residual block에 SE block을 적용하는 위치에 따른 성능을 비교한 것이다. (*Figure 5* 참고)
  - SE block은 branch aggregation전에만 사용된다면 location에 robust하게 성능이 향상된다.
    - SE-PRE와 SE-Identity의 경우, standard SE block과 유사한 성능을 보였다.
    - SE-POST의 경우, standard SE block에 비해 성능이 저하되었다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_15.png" alt="table_15" width="450px" style="border:1px solid black">
</p>

- *Table 15*는 residual unit 내부에($3 \times 3$ conv layer 바로 다음) SE block을 적용한 결과를 보여준다.
  - SE_$3 \times 3$은 standard SE block에 비해 더 적은 파라미터수로도 큰 성능 저하 없이 괜찮은 성능을 보였다. ($3 \times 3$ layer가 더 적은 channel수를 가지므로 파라미터 수가 더 적음)

# Role of SE Blocks

## Effect of Squeeze

- 본 논문에서는 squeeze에서 global embedding이 미치는 영향을 실험하기 위해 다음과 같이 수정한 NoSqueeze라는 모델과 성능을 비교하였다.
  - Global average pooling 제거
  - 2개의 FC layer를 $1 \times 1$ conv layer로 변경
- 위의 변화가 네트워크에 미치는 영향은 다음과 같다.
  - $1\times1$ point-wise convolution은 channel간의 정보들밖에 접근하지 못한다. (local operator)
  - 실제로는 네트워크가 깊어질수록 global receptive field를 가지게 되지만, 더이상 global embedding 정보로의 direct access는 불가능하게 되었다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/table_16.png" alt="table_16" width="450px" style="border:1px solid black">
</p>

- *Table 16*은 NoSqueeze의 성능을 비교한 결과이다.
  - NoSqueeze는 SE에 비해 성능이 하락하였다.
    - Global information은 모델의 성능에 큰 영향을 미치며, squeeze operation은 SE block에서 중요한 역할을 한다.
  - NoSqueeze는 SE에 비해 연산량이 더 증가하였다.
    - SE block design은 global information을 더 효율적인 방식으로 사용한다.

## Role of Excitation

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/fig_6.png" alt="fig_6" style="border:1px solid black">
</p>

- *Figure 6*는 서로 다른 class에서의 excitation 분포를 확인하기 위해, 4개의 class(goldfish, pug, plane, cliff)를 선정하고, 각 class마다 50개의 이미지에 대한 average activation을 각 stage의 마지막 SE block(downsampling 직전)에서 계산한 결과를 나타낸다. (all은 1000개 class 모두에 대한 mean activation)
  - SE_2_3(layer가 얕을 때)에서는 서로 다른 class일지라도 비슷한 분포를 보인다.
    - earlier layer features are typically more general
  - SE_4_6, SE_5_1(layer가 깊을 때)에서는 서로 다른 class간 분포의 차이가 점점 나타난다.
    - later layer features exhibit greater levels of specificity
  - SE_5_2에서는 대부분의 activation이 1로 수렴하는 형태가 나타났다.
    - SE block이 identity operator가 됨
  - SE_5_3에서는 서로 다른 class들간에 유사한 패턴이 나타났다. (크기가 일관되게 달라서 구분이 될 수 있는 형태)
    - classifier로 전달되어 분류하기 좋은 형태
- 위의 결과로부터, 마지막 stage(SE_5_2, SE_5_3)에서는 recalibration이 덜 중요하다는 것을 알 수 있으며, 이는 [[Model and Computational Complexity]](#model-and-computational-complexity)섹션에서 네트워크의 final stage를 제거하여도 성능 감소는 미미하다는 실험 결과를 뒷받침해준다.

<p align="center">
<img src="/assets/squeeze_and_excitation_networks/fig_7.png" alt="fig_7" style="border:1px solid black">
</p>

- *Figure 7*은 같은 class(goldfish, plane)간에 mean과 standard deviation이 어떻게 분포하는지를 나타낸 것이다.
  - 마지막 layer로 갈수록 한 class 내에서 representation이 다양해지며, 이에 네트워크는 feature recalibration을 사용해 discriminative 성능을 향상시키려고한다. (한 class 내에서 instance간의 다른 점을 학습하려함)

- *Figure 6, 7*을 요약하면 다음과 같다.
  - SE block은 class-specific(서로 다른 class 간 분류)과 instance-specific(한 class 내에서 instance간 분류) 모두에 대한 response를 생성한다.

# Conclusion

- 본 논문에서는 representation power를 높이고, channel-wise feature recalibration을 가능하게 하는 SE block을 제안한다.
- SENet은 여러 dataset과 task에서 SOTA의 성능을 기록하였다.
- SE block은 기존의 architecture들이 channel-wise feature dependency를 적절하게 modeling하지 못한다는 것을 보였고, 이를 개선하여 성능을 향상시켰다.
- SE block은 model compression을 위한 network pruning과 같은 task에 활용될 수 있다.
