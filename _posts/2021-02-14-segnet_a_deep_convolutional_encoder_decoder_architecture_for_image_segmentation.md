---
title: "[풀잎스쿨 14기] SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation"
key: 20210214
sidebar:
  nav: papers-ko
tags: 논문 ML/DL Semantic&nbspSegmentation
---

<p align="center">
<img src="/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/segnet_title.png" alt="segnet">
</p>

본 포스팅은 모두의연구소 (<a href="http://home.modulabs.co.kr/" target="_blank" rel="noopener noreferrer">home.modulabs.co.kr</a>) 풀잎스쿨에서 진행된 **'Semantic Segmentation 논문으로 입문하기'** 과정 내용을 공유 및 정리한 자료입니다.
{:.success}

이 글은 <a href="https://arxiv.org/abs/1511.00561" target="_blank" rel="noopener noreferrer">SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation</a> 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

# Abstract

- SegNet은 **Encoder**, **Decoder**, pixel-wise **classification layer**로 구성된다.
  - Decoder는 encoder에서의 max-pooling 위치 정보(**pooling indices**)를 사용해서 non-linear upsampling을 수행한다.
- SegNet은 **scene understanding**을 목적으로 다음과 같이 설계되었다.
  - Memory와 inference time 모두에서 효율적이도록 함
  - 모델의 학습 파라미터 수를 줄임
  - End to end로 SGD를 통해 학습
- SegNet의 성능은 Road scene 및 indoor scene segmentation task의 dataset으로 평가하였다.

# Introduction

![fig_1](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/fig_1.png){:style="border:1px solid black"}

- SegNet은 다음과 같이 **road scene understading**을 목적으로 설계되었다.
  - **Appearance**(도로, 건물 등), **shape**(자동차, 보행자 등)를 잘 인식하고 다른 class간의 spatial-relationship(**context**)를 이해할 수 있어야 함
  - Encoder에서 추출된 image representation으로부터 **boundary information을 유지하는 것이 중요**
    - 대부분 pixel을 차지하는 도로나 건물 같은 **큰 object**들에 대해서는 **smooth segmentation**을 생성할 수 있어야 함
    - 또한, 보행자와 같은 **작은 object**에 대한 **shape**도 잘 나타낼 수 있어야 함
  - **계산량 관점**에서는 **memory 및 inference time**에서 효율적으로 동작할 수 있어야 함
    - 전체 네트워크의 학습 파라미터를 end-to-end로 한번에 학습
    - 빠르게 weight update를 반복하면서 수렴할 수 있는 SGD를 사용
- SegNet의 Encoder와 Decoder 구성은 다음과 같다.
  - Encoder
    - VGG16의 입력 이후 13개의 convolutional layer까지는 동일
    - 학습 파라미터 수를 줄여서 쉽게  학습하기 위해 마지막의 fully connected layer는 제거
  - **Decoder**
    - Encoder와 동일한 계층 구조를 가짐
    - Encoder의 max pooling layer에서 위치정보(**max-pool indices**)를 받아서 non-linear upsampling을 수행
      - **Boundary delineation 성능은 향상**되면서 모델의 학습 **파라미터 수는 크게 증가하지 않음**
      - 어떠한 encoder-decoder architecture라도 큰 수정 없이 적용 가능
- **본 논문의 주된 contribution**은 다음과 같다.
  - **SegNet과 FCN의 decoder**에서 **핵심 설계 요소**(key design factors)들을 분리하여 **각각의 장단점을 분석**하였음
- SegNet의 성능은 **2가지 scene segmentation task에서 평가**하였다. (*Figure 1*)
  - **CamVid** road scene segmentation / **SUN RGB-D** indoor scene segmentation
  - 주로 사용하는 **Pascal VOC** dataset은 여러 배경에서 1~2개의 class들만 존재하는 이미지가 대부분인데, 이는 **scene understanding task와 적합하지 않음**
    - Scene understanding task에서 robust하기 위해서는 object의 **co-occurrence** 및 **spatial-context**를 학습할 수 있는 이미지 데이터가 필요

# Architecture

![fig_2](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/fig_2.png){:style="border:1px solid black"}

- **SegNet architecture**는 다음과 같다. (*Figure 2*)
  - Encoder
    - Conv + batch norm + ReLU
      - Conv는 ImageNet에서 학습된 VGG16의 13개 convolutional layers를 사용
    - Max-pooling
      - Kernel size : 2$\times$2, Stride : 2
  - Decoder
    - Upsampling
      - Encoder에서 max-pooling한 위치 정보(**max-pooling indices**)를 받아서 upsampling을 수행(*Figure 2*의 화살표)
    - Conv + batch norm + ReLU
      - Conv는 encoder의 각 계층별 위치에 해당하는 conv와 동일한 channel의 feature map을 출력 (RGB 3 channel의 이미지를 입력으로 받는 encoder의 첫번째 layer는 제외)
  - Soft-max classifier
    - Soft-max는 각 픽셀별로 class probability를 계산하여 K channel의 확률 이미지를 출력 (여기서 K는 전체 클래스 수를 의미)
    - 위 결과에서, 각 픽셀별로 가장 확률이 높은 class만을 출력하면 최종 segmentation이 됨
- SegNet은 **encoder의 feature map 정보를 효율적으로 decoder로 전달**하여 boundary delineation을 향상시키면서 메모리도 아끼는 방식을 사용하였다.
  - Segmentation의 boundary delineation을 위해 encoder의 feature map에서 boudary 정보를 capture and store하는 것은 필수적이다.
  - 전체 feature map을 저장하기 위해서는 많은 메모리가 필요하므로, SegNet에서는 **max-pooling indices** 정보만을 저장하는 방식을 사용하였다. (accuracy는 아주 조금 감소하지만 memory는 크게 아낄 수 있음)

##  Decoder Variants

![fig_3](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/fig_3.png){:style="border:1px solid black"}

- *Figure 3*은 SegNet과 FCN에서의 decoding 방법을 나타낸 것이다.
  - **SegNet**은 **max-pooling indices**를 사용
  - **FCN**은 **transposed convolution 및 dimensionality reduction**(skip-architecture)을 사용
- 본 논문에서는 SegNet과 FCN에서 decoder 구조만을 조금씩 변형시킨 다음의 네트워크들(decoder variants)을 서로 비교하며 decoding 과정에서 각 요소들이 성능에 미치는 영향을 상세히 분석하였다. (결과는 Analysis 챕터에서 나옴)
  - **Bilinear-Interpolation** (**SegNet, FCN의 decoding 방법을 사용하지 않음**)
    - Encoder는 SegNet-Basic과 동일
    - Decoder는 upsampling에 bilinear interpolation만을 사용
  - **SegNet-Basic**
    - Encoder와 decoder가 각각 4개의 layer block으로 구성됨 (SegNet은 각각 5개 씩이었음, *Figure 2* 참고)
    - Convolutional layer의 kernel size는 7$\times$7을 사용하였고, bias는 사용하지 않음
    - Decoder에는 ReLU를 사용하지 않음
  - **SegNet-Basic-EncoderAddition** (**SegNet-Basic보다 더 많은 정보 전달**)
    - Encoder는 SegNet-Basic과 동일
    - Decoder는 SegNet-Basic에서 추가로 encoder의 각 layer마다 64개 feature map을 추출해서 해당하는 위치의 decoder layer에 더해줌
  - **SegNet-Basic-SingleChannelDecoder** (**SegNet-Basic보다 더 적은 정보 전달**)
    - Encoder는 SegNet-Basic과 동일
    - Decoder의 convolution filter를 single channel로 변경하여 SegNet-Basic을 더 축소시킨 형태
  - **FCN-Basic**
    - Encoder는 SegNet-Basic과 동일
    - Decoder는 transposed convolution 및 dimensionality reduction을 사용
  - **FCN-Basic-NoAddition** (**FCN-Basic보다 더 적은 정보 전달**)
    - Encoder는 SegNet-Basic과 동일
    - Decoder는 transposed convolution만을 사용
  - **FCN-Basic-NoDimReduction** (**FCN-Basic보다 더 많은 정보 전달**)
    - Encoder는 SegNet-Basic과 동일
    - Decoder는 FCN-Basic에서 encoder의 정보를 더해줄 때 dimension을 축소하지 않고 바로 더해줌
  - **FCN-Basic-No-Addition-NoDimReduction** (FCN-Basic보다 더 적은 정보 전달, **FCN과 SegNet의 decoder 자체만을 비교하기 위함**)
    - Encoder는 SegNet-Basic과 동일
    - Decoder는 transposed convolution 및 dimensionality reduction을 모두 사용하지 않음

## Training

- 아래의 설정들은 다음 챕터(Analysis)에서 decoder variants의 학습에 사용된 설정들이다.
  - Training dataset : CamVid road scenes
  - Weight initialization : He initialization (encoder와 decoder  모두)
  - Optimizer : SGD with momentum 0.9
  - Learning rate : 0.1
  - Batch size : 12
  - Loss function : cross-entropy
  - Class balancing : median frequency balancing
    - 학습 데이터에서 pixel의 대부분을 차지하는 class들에는 1보다 작은 weight를 주고 그렇지 않은 class들에는 1보다 큰 weight를 주어 loss를 계산

## Analysis

![table_1](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/table_1.png){:style="border:1px solid black"}

- Decoder variants들을 비교한 결과는 *Table 1*과 같다.
  - **Performance measure** (<a href="https://younnggsuk.github.io/2021/01/24/fully_convolutional_networks_for_semantic_segmentation.html#metrics" target="_blank" rel="noopener noreferrer">FCN 논문 정리 - metric</a> 참고)
    - G - Global Average : FCN 논문에서의 pixel accuracy와 동일
    - C - Class Average : FCN 논문에서의 (pixel) mean accuracy와 동일
    - mIoU : FCN 논문에서의 mean IU와 동일
    - BF - Boundary F1-measure : mIoU만으로는 boundary accuracy를 평가하기 어려우므로 이를 보완하기 위해 사용된 척도
      - Ground truth의 boundary와 예측한 boundary간의 precision, recall을 구해서 F1-measure를 계산하는 방식 (자세한 계산 방식은 <a href="http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf" target="_blank" rel="noopener noreferrer">Csurka et al.의 논문</a> 3.2절 참고)
    - *Table 1*의 성능들은 Validation set에서 **Global Accuracy(G)가 가장 높게 나왔을 때,** test set에서 평가한 결과이다.
      - Global Accuracy를 기준으로 잡은 이유
        - 전체 데이터에 대한 **smooth segmentation**을 위해서는 global accuracy가 중요
        - Autonomous driving에서의 segmentation은 도로, 건물, 보도, 하늘과 같이 **대부분의 pixel을 차지하는 class들의 delineation 성능이 중요**한데, **global accuracy가 이러한 class들의 segmentation 성능에 해당**
  - Bilinear-Interpolation
    - 가장 성능이 좋지 못함
    - **Segmentation을 위해서는 decoder가 필요**하다는 것을 보여줌
  - SegNet-Basic vs. FCN-Basic
    - **SegNet-Basic이 더 적은 메모리**를 사용
      - SegNet-Basic은 **max-pooling indices**를 저장해서 메모리 사용량이 적음
      - FCN-Basic은 encoder의 feature map을 저장해서 메모리 사용량이 SegNet의 11배나 됨
    - **FCN-Basic이 더 빠른 inference time (forward pass)**을 기록
      - SegNet-Basic의 decoder는 각 layer별로 총 64개의 feature map을 가짐
      - FCN-Basic의 decoder는 **dimensionality reduction**을 통해 각 layer별로 더 적은 11개의 feature map을 가짐
    - 둘 모두 비슷하게 좋은 결과를 보이므로, 다음과 같이 정리할 수 있다.
      - **메모리가 적은데 inference는 느려도 되는 경우에는 SegNet-Basic**
      - **메모리는 많은데 inference가 빨라야 하는 경우에는 FCN-Basic**
  - SegNet-Basic vs. FCN-Basic-NoAddition
    - SegNet-Basic이 더 좋은 성능을 보임
    - **Encoder의 feature map을 decoder로 전달**하는 것이 **성능 향상**에 큰 영향을 미친다는 것을 보여줌
  - SegNet-Basic vs. FCN-Basic-No-Addition-NoDimReduction
    - SegNet-Basic이 더 좋은 성능을 보임
    - Decoder를 단순히 크게 하는 것보다 **encoder의 feature map을 decoder로 전달하는 것이 중요**하다는 것을 보여줌
  - SegNet-Basic-SingleChannelDecoder vs. FCN-Basic-NoAddition
    - SegNet-Basic-SingleChannelDecoder가 더 좋은 성능을 보임
    - **SegNet architecture가 좋은 성능**을 보인다는 것을 보여줌
      - Max-pooling indices를 사용한 upsampling
      - FCN에 비해 더 큰 decoder
  - SegNet-EncoderAddition & FCN-Basic-NoDimReduction
    - 둘 모두 각자의 variants에서 가장 높은 성능을 보임
    - Memory와 inference time에 제약을 받지 않는다면, **더 많은 정보를 전달**하는 **큰 모델**이 **높은 accuracy**를 보인다는 것을 보여줌
      - **Memory와 accuracy는 trade-off** 관계
  - **Class balancing을 수행하지 않은 경우** (*Table 1*의 가장 오른쪽 두 열)
    - Class balancing을 수행한 결과(바로 왼쪽의 두 열)보다 **성능이 좋지 못함**
      - Class Average와 mIoU가 크게 감소함
      - Global Average에서는 성능이 약간 높게 나옴
        - 도로나 하늘, 건물과 같은 큰 class들이 대부분의 pixel을 차지하기 때문
- 위의 결과들을 모두 정리하면 다음과 같다.
  - Encoder의 **feature map을 모두 사용하면 가장 좋은 성능**을 보인다.
  - **Memory가 제한될 경우**에는 다음의 방법을 시도해볼 수 있다.
    - Encoder의 **feature map을 compress**하여 저장하는 방법을 사용
      - Dimensionality reduction
      - Max-pooling indices
    - **SegNet type의 decoder**를 사용
  - **Decoder가 클수록 성능이 향상**된다.

# Benchmarking

- SegNet의 성능은 2가지 scene segmentation benchmark에서 평가하였다.
  - Cam Vid dataset - road scene segmentation
  - SUN RGB-D dataset - indoor scene segmentation
- SegNet의 성능을 다른 architecture(FCN, DeepLab-LargeFOV, DeconvNet 등)와 비교하기 위한 설정들은 다음과 같다.
  - SegNet
    - Convolutional layer 뒤에 batch normalization 추가
  - DeepLab-LargeFOV
    - Max pooling의 stride를 3에서 1로 변경하여 마지막 출력의 크기가 45$\times$60이 되도록 함
  - DeconvNet
    - Fully connected layer의 feature size를 1024로 제한하여 다른 모델들과 동일한 batch size로 학습할 수 있도록 함
  - 모든 모델의 깊은 convolutional layer 마지막에 dropout 0.5 추가
  - Optimization
    - SGD with momentum 0.9
    - Learing rate : $10^{-3}$
    - Batch size
      - Road scene : 5
      - Indoor scene : 4

## Road Scene Segmentation

![fig_4](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/fig_4.png){:style="border:1px solid black"}

- SegNet 및 다른 모델들의 segmentation 결과는 *Figure 4*와 같다.
  - **SegNet**
    - **가장 우수한 성능**을 보임
    - **작은 object**와 큰 object 모두에서 잘 동작하며 **smooth segmentation**을 생성
  - DeepLab-LargeFOV
    - 가장 효율적인 모델 (*Table 6*의 계산량 비교 참고)
    - CRF post-processing으로 인해 좋은 성능을 보임
    - 작은 object에 대해서는 잘 동작하지 않음
  - FCN with learnt deconvolution
    - Fixed bilinear upsampling(FCN)보다 좋은 성능을 보임
  - DeconvNet
    - 가장 크고 비효율적인 모델 (*Table 6*의 계산량 비교 참고)
    - 작은 object에 대해서는 잘 동작하지 않음

![table_3](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/table_3.png){:style="border:1px solid black"}

- SegNet 및 다른 모델들의 더 자세한 성능 비교 결과는 *Table 3*와 같다.
  - Training settings
    - Class balancing을 사용하면 DeconvNet과 같은 큰 모델을 학습시키기가 어려워서 사용하지 않음
    - $>80$K는 더이상의 성능 향상이 나타나지 않거나 overfitting이 시작되는 전 지점을 의미(즉, 최대 성능을 보인 지점)
  - **SegNet, DeconvNet이 가장 좋은 성능**을 보임
    - **DeconvNet**은 **더 높은 BF**(boundary delineation accuracy)를 보이고, **SegNet**은 **더 효율적**으로 동작한다. (*Table 6*의 계산량 비교 참고)
  - FCN (learnt deconv)
    - Fixed bilinear upsampling(FCN)보다 좋은 성능을 보임
  - DeepLab-LargeFOV
    - 가장 모델의 크기가 작고 빠른 효율적인 모델임에도 불구하고 좋은 성능을 보임
    - BF(bounary delineation accuracy)가 낮음
  - DeepLab-LargeFOC-denseCRF
    - DeepLab-LargeFOV에 비해 Global Average, mIoU, BF가 향상되고 Class Average는 감소함
      - 특히 BF가 크게 향상됨

![table_6](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/table_6.png){:style="border:1px solid black"}

- *Table 6*는 앞서 비교한 모델들간의 효율성을 비교한 결과이다.
  - **SegNet**이 **inference에서 가장 적은 GPU 메모리**를 사용한다.
  - **DeepLab-LargeFOV**가 **forward, backward, training 모두에서 가장 빠르게 동작**한다.

## SUN RGB-D Indoor Scenes

- **SUN RGB-D dataset**은 다음과 같은 특징 때문에 **아주 까다로운 segmentation challenge** 중 하나이다.
  - Object의 shape, size, pose가 모두 다양하다.
  - Object의 일부분이 가려진(partial occlusion) 이미지도 많이 존재한다.
- Road scene image에 비해 **indoor scene이 더 어려운 이유**는 다음과 같다.
  - **View point**
    - 차량에 부착된 카메라의 위치는 거의 항상 도로 표면과 평행하므로, view point가 크게 변하지 않는다.
    - 고정된 위치에서 실내를 바라보지 않기 때문에, view point는 모두 제각각이며 크게 변할 수 있다.
  - **Object의 수, 크기 및 배치**
    - 도로 위 차량들의 크기 및 대수는 크게 변하지 않으며, 차선을 따라 이들의 배치는 크게 변하지 않는다.
    - 실내에 있는 가구들의 크기와 갯수 및 이들의 배치는 모두 제각각이므로, 크게 변할 수 있다.

![fig_5](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/fig_5.png){:style="border:1px solid black"}

- *Figure 5*는 SegNet과 여러 모델들의 segmentation 결과를 비교한 것이다. (Depth map을 사용하려면 네트워크 구조를 변경해야 하므로, 학습에는 RGB만 사용)
  - **SegNet**은 RGB image만으로도 **큰 object에서 좋은 성능**을 보였다.
    - RGB image는 depth sensor로 감지하기 어려운 의자나 책상의 다리와 같이 얇은 구조의 segmentation에서도 유용하게 사용됨
  - (다른 outdoor scene에서의 결과들과 비교해서) 전체적으로 noisy한 결과를 보인다.
    - 특히 3번째 열과 같이 **어수선한 장면의 경우(when clutter is increased), 아주 noisy**한 결과가 나타남

![table_4](/assets/segnet_a_deep_convolutional_encoder_decoder_architecture_for_image_segmentation/table_4.png){:style="border:1px solid black"}

- *Table 4*는 SegNet과 여러 모델들의 성능을 비교한 것이다.
  - **모든 모델들에서 전반적으로 성능이 낮게** 나왔으며, 특히 class average, mIoU, BF가 크게 낮다.
    - 대부분의 class가 image에서 아주 작은 부분을 차지하면서 아주 드물게 등장하기 때문
  - **SegNet**은 다른 모델들과 비교해서 **G, C, mIoU, BF 모두에서 우수한 성능**을 보인다.
    - mIoU는 DeepLab-LargeFOV보다 약간 낮음

# Conclusion

- 본 논문에서는 **road 및 indoor scene understanding을 목적**으로 메모리 및 연산 시간에서 **효율적으로 동작**하도록 설계한 **SegNet architecture**를 제안한다.
- SegNet과 FCN의 decoder variants를 비교하며 **memory vs. accuracy의 trade-off를 분석**하였다.
  - Encoder의 feature map 정보를 더 많이 사용할수록 높은 accuracy를 얻지만, 더 많은 메모리를 사용한다.
- SegNet은 **max-pooling indices**를 저장하는 방법을 사용해 **효율적으로 동작하면서도 좋은 성능**을 보인다.
- SegNet은 **scene understanding benchmark에서 좋은 성능**을 보였다.