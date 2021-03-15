---
title: "[풀잎스쿨 14기] U-Net: Convolutional Networks for Biomedical Image Segmentation"
key: 20210208
sidebar:
  nav: papers-ko
tags: 논문 ML/DL Semantic&nbspSegmentation
---

<p align="center">
<img src="/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/u-net_title.png" alt="u-net">
</p>

본 포스팅은 모두의연구소 (<a href="http://home.modulabs.co.kr/" target="_blank" rel="noopener noreferrer">home.modulabs.co.kr</a>) 풀잎스쿨에서 진행된 **'Semantic Segmentation 논문으로 입문하기'** 과정 내용을 공유 및 정리한 자료입니다.
{:.success}

이 글은 <a href="https://arxiv.org/abs/1505.04597" target="_blank" rel="noopener noreferrer">U-Net: Convolutional Networks for Biomedical Image Segmentation</a> 논문을 읽고 정리한 것입니다. 이 글에 있는 대부분의 사진들은 논문에서 가져온 것임을 밝힙니다.
{:.info}

# Abstract

- 본 논문에서는 적은 데이터셋을 더 효율적으로 사용하기 위한 **data augmentation** 기법과 이를 통해 학습한 네트워크(**U-Net**)를 제안한다.
- U-Net에 대해 정리하면 다음과 같다.
  - **Contracting path**와 **expanding path**로 구성된다.
  - **적은 학습 이미지**로부터 end-to-end로 학습한다.
  - ISBI challenge에서 가장 높은 성능을 기록하였다.

# Introduction

- **Biomedical image processing** 분야의 특징은 다음과 같다.
  - 출력 결과는 반드시 localization을 포함해야 하므로, 각 픽셀 별 classification(**segmentation**)을 수행해야 한다.
  - 일반적으로 biomedical task에서는 수천개의 이미지 데이터를 구축하기 어려우며, 이는 **학습에 사용가능한 이미지의 수가 적다**는 것을 의미한다.

- ISBI 2012의 EM segmentation challenge에서 가장 높은 성능을 기록하였던 **<a href="https://papers.nips.cc/paper/2012/file/459a4ddcb586f24efd9395aa7662bc7c-Paper.pdf" target="_blank" rel="noopener noreferrer">Ciresen et al.의 연구</a>**에는 다음의 2가지 **단점**이 존재하였다.
  - 이미지로부터 patch를 추출하여 학습하는 방법을 사용하였는데, 네트워크가 각 patch별로 독립적으로 동작하기 때문에 **속도가 느렸고** 겹치는 patch들로 인한 **중복(redundancy)**이 많았다.
  - Localization accuracy와 네트워크가 바라보는 context간의 **trade-off**가 있었다.
    - **Large patch** -->  more max-pooling layer --> **reduce localization accuracy** 
    - **Small patch** --> less max-pooling layer --> **see only little context**

- 본 논문에서는 다음과 같이 **<a href="https://arxiv.org/abs/1411.4038" target="_blank" rel="noopener noreferrer">fully convolutional network(FCN)</a>**를 수정하여 **적은 학습 이미지로 정확한 segmentation**을 수행할 수 있는 **U-Net architecture**를 제안한다.
  - **Upsampling**에서 **더 많은 feature channel**을 사용하여 네트워크가 context 정보를 상위 layer로 전파할 수 있도록 하였다.
    - Contracting path에서의 feature를 expanding path로 copy and crop
  - Fully connected를 사용하지 않고 **convolution만을 사용**하였다.
  - **Overlap-tile strategy**를 통해 patch단위로 segmentation을 수행하여 큰 이미지에 대해서도 중간에 끊김 없이 매끄러운 segmentation 결과를 얻을 수 있도록 하였다.
    - 이미지의 **테두리에서는 mirroring**을 통해 missing context를 보완하였다.
    - 큰 이미지에서 GPU 메모리의 부족으로 인해 해상도가 제한되는 것을 막아준다.
  - **Elastic deformation**을 적용한 data augmentation을 수행하였다.
    - 부족한 학습 데이터를 보완하였다.
    - 네트워크가 deformation에서 robust하도록 해준다.
    - **세포 조직이 각 순간순간 마다 조금씩 변화하는 것을 elastic deformation이 효과적으로 시뮬레이션 할 수 있기 때문**에, biomedical segmentation에서는 매우 효과적이다.
  - 같은 class이면서 가까이 붙어있는 cell들을 분리(**instance segementation**)하기 위해 **weighted loss**를 사용하였다.
    - Loss function에서 붙어있는 cell들 사이에서 이들을 분리하는 border의 background label이 큰 weight를 갖도록 하였다.

- U-Net은 다양한 biomedical segmentation에 적용할 수 있다.
  - ISBI의 EM segmentation challenge와 cell tracking challenge에서 가장 높은 성능을 기록하였다.

# Network Architecture

![fig_1](/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/fig_1.png){:width="700px" style="border:1px solid black"}

- **U-Net architecture**는 다음과 같이 구성된다. (*Figure 1*)
  - **Contracting path** (downsampling)
    - 3$\times$3 convolution (padding X), batch normalization, ReLU
    - 2x2 max pooling (stride 2)
    - conv, max pool을 통한 각 downsampling step마다 output feature channel의 수는 2배가 된다.
  - **Expansive path** (upsampling)
    - 2$\times$2 up-convolution (transposed convolution)
      - Feature channel을 반으로 줄이고, **contracting path의 feature map을 concatenate**(copy and crop)
    - 3$\times$3 convolution (padding X), batch normalization, ReLU
  - Final layer (output map)
    - 1x1 convolution
- Over-lap tiling을 적용하기 위해서는 2$\times$2  max pooling으로 **최종 출력 크기가 원본 크기와 같아지는 input tile size를 선택**해야 한다.

# Training

![fig_2](/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/fig_2.png){:width="700px" style="border:1px solid black"}

- 네트워크의 학습에는 Caffe 프레임워크의 **stochasic gradient descent**를 사용하였다.
- Unpadded convolution으로 인해 출력 이미지는 입력 이미지보다 크기가 줄어든다. (*Figure 2*)
  - Over-lap tiling을 위해 **원본 이미지에 mirroring으로 테두리를 추가한 이미지가 입력**으로 들어가게 되고, **추가된 테두리만큼 출력 이미지의 크기는 줄어들게 된다.**
- GPU 메모리 사용을 최대로 하기 위해 **batch size보다 input tile size를 크게**하는 방법을 사용하였다.
  - Batch size는 1(single image)로 하였다.
  - **작은 batch size를 보완하기 위해 momentum은 0.99로 큰 값을 사용**하여 previous step에서의 여러 training sample들이 current step에서의 update에 더 많은 영향을 미치도록 하였다.

![fig_3](/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/fig_3.png){:width="700px" style="border:1px solid black"}

- Energy function은 최종 feature map에서의 pixel-wise **softmax와 weight map에 cross entropy loss**를 적용한 형태이다.
  - **Softmax $p_k(\mathbf{x})$**

    $$
    p_k(\mathbf{x}) = \text{exp}(a_k(\mathbf{x})) / (\sum^K_{k'=1} \text{exp}(a_{k'}(\mathbf{x})))
    $$

    - $a_k(\mathbf{x})$은 픽셀 위치 $\mathbf{x} \in \Omega$에서 feature channel $k$의 activation이다.
    - 즉, $p_k(\mathbf{x}) \approx 1$이라면, k번째 channel에서 activation($a_k(\mathbf{x})$)이 가장 커서 $k$의 확률이 가장 높다는 의미이다.

  - **Weight map $w(\mathbf{x})$** (*Figure 3*)

    $$
    w(\mathbf{x}) = w_c(\mathbf{x}) + w_0 \cdot \text{exp} \left(-\frac{(d_1(\mathbf{x}) + d_2(\mathbf{x}))^2}{2\sigma^2}\right)
    $$

    - Weight map은 학습 데이터에서의 class간 imbalance를 맞춰주고, **가까이 붙어있는 cell들 사이의 작은 경계(border)를 학습할 수 있도록** 하기 위해 미리 계산하는 값이다.
    - $w_c : \Omega \rightarrow \mathbf{R}$은 class frequency의 균형을 맞춰주기 위한 weight map이다.
    - $d_1:\Omega \rightarrow \mathbf{R}$은 가장 가까운 cell의 경계(border)까지의 거리이다.
    - $d_2:\Omega \rightarrow \mathbf{R}$은 두번째로 가까운 cell의 경계(border)까지의 거리이다.
    - 논문에서는 $w_0=10, \sigma \approx 5$로 두고 실험을 수행하였다.

  - **$p_k(\mathbf{x})$와 $w(\mathbf{x})$에 cross entropy를 적용한 energy function $E$**

    $$
    E = \substack{\sum \\ \mathbf{x} \in \Omega} w(\mathbf{x}) \text{log}(p_{l(\mathbf{x})}(\mathbf{x}))
    $$

    - Energy function은 각 픽셀 위치에서 **$p_{l(\mathbf{x})}(\mathbf{x})$와 1의 차이에 페널티**를 준다.
      - 즉, 정답일 확률이 낮을수록 패널티를 주어 정답 확률이 높아지도록 하는 것이다.
    - $l : \Omega \rightarrow \{1, \cdots, K\}$은 각 픽셀에 대한 true label을 의미한다.
    - 즉, $p_{l(\mathbf{x})}(\mathbf{x})$는 정답 label($k=l$)에서의 softmax 결과(정답의 확률)를 의미한다.

- Weight Initialization에는 **<a href="https://arxiv.org/abs/1502.01852" target="_blank" rel="noopener noreferrer">He Initialization</a>**을 사용하였다.

## Data Augmentation

<figure>
  <img src="/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/random_elastic_deformation.png" alt="random_elastic_deformation"/>
  <figcaption><i><a href="https://towardsdatascience.com/review-u-net-biomedical-image-segmentation-d02bf06ca760" target="_blank" rel="noopener noreferrer">그림 출처 : Towards Data Science - Review: U-Net (Biomedical Image Segmentation)</a></i></figcaption>
</figure>

- **<a href="https://hj-harry.github.io/HJ-blog/2019/01/30/Elastic-distortion.html" target="_blank" rel="noopener noreferrer">Random elastic deformation</a>**을 통해 data augmentation을 수행하였으며, 이는 **적은 데이터로도 네트워크를 학습할 수 있었던 key concept**이다.
  - Microscopical image의 경우, shift and rotation invariance와 robustness to deformation and gray value variations가 요구되는데 이를 충족시켰다.

# Experiments

![table_1](/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/table_1.png){:width="600px" style="border:1px solid black"}

![fig_4](/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/fig_4.png){:width="600px" style="border:1px solid black"}

![table_2](/assets/u-net_convolutional_networks_for_biomedical_image_segmentation/table_2.png){:width="600px" style="border:1px solid black"}

- U-Net은 3가지의 다른 segmentation task에서 모두 좋은 성능을 보였다.
  - EM segmentation challenge
    - Dataset은 *Figure 2*를 참고
  - Cell tracking challenge의 2가지 dataset
    - Dataset("PhC-U373")은 *Figure 4 - (a), (b)*를 참고
    - Dataset("DIC-HeLa")은 *Figure 3, Figure 4 - (c), (d)*를 참고

# Conclusion

- U-Net은 **biomedical segmentation task**에서 **높은 성능**을 기록하였다.
- **Elastic deformation**을 통한 data augmentation을 통해 적은 이미지 데이터로 네트워크를 성공적으로 학습할 수 있었다.