---
title: "1.2 MLE"
key: 20210324
sidebar:
  nav: machine_learning_icmoon-ko
tags: ML/DL
---

이 글은 카이스트 문일철 교수님의 [인공지능 및 기계학습 개론 1 강의](https://kooc.kaist.ac.kr/machinelearning1_17)를 듣고 정리한 것입니다.
{:.info}

# Experience from trials

동전을 던졌을 때 앞면 또는 뒷면이 나올 확률을 생각하듯이, 압정을 던지는 경우를 생각해보자. 압정은 동전과 달리 양면이 대칭이 아니므로, 쉽게 떠올리기 어려울 것이다.

그렇다면, 5번 압정을 던진 결과가 다음과 같이 나온 경우를 생각해보자.

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/1_2_mle/thumbtack_head.jpg" alt="head_1" width="100px">
  <img src="/assets/machine_learning_icmoon_1/1_2_mle/thumbtack_head.jpg" alt="head_2" width="100px">
  <img src="/assets/machine_learning_icmoon_1/1_2_mle/thumbtack_tail.jpg" alt="tail_1" width="100px">
  <img src="/assets/machine_learning_icmoon_1/1_2_mle/thumbtack_head.jpg" alt="head_3" width="100px">
  <img src="/assets/machine_learning_icmoon_1/1_2_mle/thumbtack_tail.jpg" alt="tail_2" width="100px">
</p>

압정의 머리가 위로 나온 경우를 H, 꼬리가 위로 나온 경우를 T라고 한다면, 위의 결과로부터 H와 T가 나올 각각의 확률은 직관적으로 다음과 같이 생각할 수 있다.

$$P(H) = \dfrac{3}{5} \\ P(T) = \dfrac{2}{5}$$

그런데, 우리는 무슨 근거로 위의 두 확률을 구한 것일까?

# Binomial Distribution

앞서 구한 두 확률이 어디서 온 것인지를 알기 위한 몇가지 개념들을 살펴보자.

**Binomial Distribution**
> Binomial Distribution은 연속된 $n$번의 independent trial에서 각 trial이 확률 $p$를 가질 때의 이산 확률 분포를 의미한다. 이때, 각 trial은 bernoulli trial이라고 부른다.

압정을 던진 실험에서 생각해보면, 압정을 던졌던 횟수는 n=5이며, 압정의 머리가 나올 확률은 $p=P(H)$라고 생각할 수 있다.

따라서, 압정 실험은 binomial distribution을 따른다고 볼 수 있다.

**i.i.d condition**
> i.i.d(independent and identically distributed) condition은 각각의 trial은 모두 independent이며, 모두 동일한 확률분포를 따른다는 것을 의미한다.
- **i**ndependent : 각각의 trial은 모두 independent이다.
- **i**dentically **d**istributed : 각각의 trial은 모두 동일한 확률 분포를 따른다.

압정을 던지는 trial들은 이전에 수행한 trial에 의존하지 않는 독립적인 trial이었으며(independent), 각 실험에서 압정이 손상되지 않는다고 가정하면 확률 분포에 영향이 없기 때문에 동일한 확률 분포를 따른다(identically distributed)고 볼 수 있다.

따라서, 압정 실험은 i.i.d condition에서 수행되었다고 볼 수 있다.

# Maximum Likelihood Estimation

이제 본격적으로 두 확률이 어디서 왔는지를 살펴보자.

압정의 머리가 나올 확률을 $\theta$라고 하면, H와 T가 나올 각각의 확률과 압정을 던진 실험으로부터 관측된 데이터(HHTHT)의 확률은 다음과 같다.

- $P(H) = \theta$
- $P(T) = 1-\theta$
- $P(HHTHT) = \theta \theta (1-\theta) \theta (1-\theta) = \theta^3 (1-\theta)^2$

따라서, (압정의 모양에 따른) H가 나올 확률 $\theta$가 주어졌을 때, $D = \\{H,H,T,H,T\\}$라는 데이터가 관측될 확률 $P(D\|\theta)$는 다음과 같이 정리할 수 있다.

$$P(D|\theta) = \theta^{a_H} (1-\theta)^{a_T}$$

위 식에서 각 변수에 대한 설명은 다음과 같다.
- $n$ : 총 trial 횟수
- $a_H$ : H가 나온 횟수
- $a_T$ : T가 나온 횟수

그런데, 위 식은 "압정의 실험(관측한 데이터)이 $\theta$의 binomial distribution을 따른다."라는 hypothesis를 바탕으로 전개한 식이다. 우리의 hypothesis를 더 신뢰할 수 있게 만들기 위해 확률 $P(D\|\theta)$를 어떻게 하면 최대화할 수 있을까?

이는 "데이터를 가장 잘 나타내는 $\theta$를 어떻게 하면 찾을 수 있을까?"라는 질문과 동일하며, 이러한 $\theta$를 추정하는 기법을 Maximum Likelihood Estimation(MLE)라고 한다.

## Maximum Likelihood Estimation of $\theta$ : $\hat\theta$

MLE를 식으로 나타내면 다음과 같으며, 여기서의 $\hat\theta$는 $P(D\|\theta)$를 최대화하는 $\theta$값을 의미한다.

$$\hat\theta = argmax_\theta P(D|\theta)$$

## MLE Calculation

이제 압정 실험에서의 MLE를 계산해보자. 식은 다음과 같다.

$$\begin{align*} \hat\theta &= argmax_\theta P(D|\theta) \\ &= argmax_\theta \theta^{a_H} (1-\theta) ^{a_T} \end{align*} $$

위 식은 제곱항으로 인해 복잡한 형태이므로, 최소/최대값을 구하기 위해 미분하기 까다로운 형태이다. 로그를 취해서 미분하기 편한 형태로 바꿔보자. ($P(D\|\theta)$를 최대화 하는 $\theta$는 $\ln\left( P(D\|\theta) \right)$를 최대화 하는 $\theta$와 동일)

$$\begin{align*} \hat\theta &= argmax_\theta \ln\left( P(D|\theta) \right) \\ &= argmax_\theta \ln \left( \theta^{a_H} (1-\theta) ^{a_T} \right) \\ &= argmax_\theta \left( a_H \ln \theta + a_T \ln(1-\theta) \right) \end{align*}$$

이제 위 식을 $\theta$에 대해 미분하면, 다음과 같이 $\theta$의 MLE 값인 $\hat\theta$를 구할 수 있다.

$$ \dfrac{d}{d\theta} \left( a_H \ln \theta + a_T \ln(1-\theta) \right) = \dfrac{a_H}{\theta} - \dfrac{a_T}{1-\theta} = 0 $$

$$ \therefore \theta = \dfrac{a_H}{a_T + a_H} = \hat\theta$$

위 결과에 압정 실험에서의 값들을 대입하면, 우리가 직관적으로 생각한 결과(단순히 5번 던져서 3번 나왔으니까 $\dfrac{3}{5}$)와 동일한 결과가 나온다는 것을 알 수 있다.

$$ \hat\theta = \dfrac{a_H}{a_T + a_H} = \dfrac{3}{2 + 3} = \dfrac{3}{5} $$

정리하자면, 우리가 처음에 생각했던 확률 $\dfrac{3}{5}$이라는 값은 "MLE 관점에서 최적화된 $\theta$ 값"을 구한 것이며, MLE는 데이터(실험 결과)를 가장 잘 나타내는 $\theta$를 추정하는 기법을 의미한다.

# Number of Trials

이번에는 다음과 같이 압정을 50번 던져서 H가 30번 나온 경우를 생각해보자.

- $n = 50$
- $a_H = 30$
- $a_T = 20$

MLE를 구해보면 다음과 같이 결과는 동일하다.

$$ \hat\theta = \dfrac{a_H}{a_T + a_H} = \dfrac{30}{20 + 30} = \dfrac{3}{5} $$

그렇다면, 더 많은 실험은 의미가 없는 것일까?

## Simple Error Bound

많은 trial이 어떠한 의미를 가지는지에 대해서는 error bound를 표현하는 다음의 Hoeffding's inequality 식을 통해 알 수 있다.

$$ P(|\hat\theta-\theta^*| \ge \varepsilon) \le 2e^{-2N\varepsilon} $$

위 식에서 각 변수에 대한 설명은 다음과 같다.

- $\hat\theta$ : 추정한 값
- $\theta^*$ : 참 값
- $\varepsilon$ : error bound
- $N$ : trial 수

많은 실험을 통해 $N$값이 커지면 $2e^{-2N\varepsilon}$ 값이 작아지고, 이를 통해 참값과 추정한 값의 차이가 error bound보다 커질 확률 $P(\|\hat\theta-\theta^*\| \ge \varepsilon)$값도 작아진다.

따라서, trial을 많이 하는 것은 특정 error 값의 범위를 넘어설 확률을 줄여준다는 의미가 있다.