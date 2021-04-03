---
title: "1. Motivations and Basics"
key: 20210403
sidebar:
  nav: machine_learning_icmoon-ko
tags: ML/DL 인공지능&nbsp및&nbsp기계학습&nbsp개론&nbsp1 Probability&nbspDistribution
---

이 글은 카이스트 문일철 교수님의 [인공지능 및 기계학습 개론 1 강의](https://kooc.kaist.ac.kr/machinelearning1_17)를 듣고 정리한 것이며, 몇가지 정의들은 위키피디아를 참고하였습니다.
{:.info}

# Experience from trials

압정을 던져서 머리가 위로 나올 확률을 구하기 위해 5번의 실험을 한 결과가 다음과 같다고 하자. 

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/ch1/thumbtack_head.jpg" alt="head_1" width="100px">
  <img src="/assets/machine_learning_icmoon_1/ch1/thumbtack_head.jpg" alt="head_2" width="100px">
  <img src="/assets/machine_learning_icmoon_1/ch1/thumbtack_tail.jpg" alt="tail_1" width="100px">
  <img src="/assets/machine_learning_icmoon_1/ch1/thumbtack_head.jpg" alt="head_3" width="100px">
  <img src="/assets/machine_learning_icmoon_1/ch1/thumbtack_tail.jpg" alt="tail_2" width="100px">
</p>

확률 변수는 각각 압정의 머리가 위로 나온 경우를 H, 꼬리가 위로 나온 경우를 T라고 하자. 그리고 압정의 머리가 나올 확률을 $\theta$라고 하면, H와 T가 나올 각각의 확률과 압정을 던진 실험으로부터 관측된 데이터(HHTHT)의 확률은 다음과 같다.

- $P(H) = \theta$
- $P(T) = 1-\theta$
- $P(HHTHT) = \theta \theta (1-\theta) \theta (1-\theta) = \theta^3 (1-\theta)^2$

따라서, (압정의 모양에 따른) H가 나올 확률 $\theta$가 주어졌을 때, $D = \\{H,H,T,H,T\\}$라는 데이터가 관측될 확률 $P(D\|\theta)$는 다음과 같이 정리할 수 있다.

$$P(D|\theta) = \theta^{a_H} (1-\theta)^{a_T}$$

위 식에서 각 변수에 대한 설명은 다음과 같다.
- $n$ : 총 trial 횟수
- $a_H$ : H가 나온 횟수
- $a_T$ : T가 나온 횟수

# MLE

MLE(maximum likelihood estimation)는 어떤 확률변수에서 표집한 값들을 토대로 그 확률변수의 모수(parameter)를 구하는 방법이다. 어떤 모수가 주어졌을 때, 원하는 값들이 나올 가능도(likelihood)를 최대로 만드는 모수를 선택하는 방법이다.

이를 압정실험에서 생각해본다면, 모수 $\theta$가 주어졌을 때 $D$라는 Data가 관측될 확률 $P(D \| \theta)$를 최대로 만드는 모수 $\hat\theta$을 구하는 것으로 생각할 수 있다.

MLE를 식으로 나타내면 다음과 같다.

$$\hat\theta = argmax_\theta P(D|\theta)$$

## MLE Calculation

압정 실험에서의 결과를 바탕으로 MLE를 계산해보자.

$$\begin{align*} \hat\theta &= argmax_\theta P(D|\theta) \\ &= argmax_\theta \theta^{a_H} (1-\theta) ^{a_T} \end{align*} $$

최소/최대값을 구하기 위해서는 미분을 사용할 수 있는데, 위 식은 제곱항으로 인해 복잡한 형태이므로 미분하기 까다로운 형태이다.

따라서, 다음과 같이 로그를 취해서 미분하기 편한 형태로 바꿀 수 있다.($P(D\|\theta)$를 최대화 하는 $\theta$는 $\ln\left( P(D\|\theta) \right)$를 최대화 하는 $\theta$와 동일)

$$\begin{align*} \hat\theta &= argmax_\theta \ln\left( P(D|\theta) \right) \\ &= argmax_\theta \ln \left( \theta^{a_H} (1-\theta) ^{a_T} \right) \\ &= argmax_\theta \left( a_H \ln \theta + a_T \ln(1-\theta) \right) \end{align*}$$

이제 위 식을 $\theta$에 대해 미분하면, 다음과 같이 $\hat\theta$을 구할 수 있다.

$$ \dfrac{d}{d\theta} \left( a_H \ln \theta + a_T \ln(1-\theta) \right) = \dfrac{a_H}{\theta} - \dfrac{a_T}{1-\theta} = 0 $$

$$ \therefore \hat\theta = \dfrac{a_H}{a_T + a_H} $$

위 값이 최솟값이 아닌 최댓값이라는 것을 확인하기 위해 한번 더 미분한 결과를 살펴보면, 다음과 같이 음의 값이 나오므로 최댓값이라는 것을 알 수 있다.

$$ \dfrac{d}{d\theta} \left( \dfrac{a_H}{\theta} - \dfrac{a_T}{1-\theta} \right) = - \dfrac{a_H}{\theta^2} - \dfrac{a_T}{(1-\theta)^2} \lt 0 $$

$\hat\theta$식에 압정 실험에서의 값들을 대입하면, 다음과 같이 MLE를 구할 수 있다. (우리가 직관적으로 생각할 수 있는 값(단순히 5번 던져서 3번 나왔으니까 $\frac{3}{5}$)과 동일)

$$ \hat\theta = \dfrac{a_H}{a_T + a_H} = \dfrac{3}{2 + 3} = \dfrac{3}{5} $$

## Number of Trials

앞서 구한 식을 통해 압정을 50번 던져서 머리가 30번 나온 경우의 MLE를 구해보면 다음과 같다.

$$ \hat\theta = \dfrac{a_H}{a_T + a_H} = \dfrac{30}{20 + 30} = \dfrac{3}{5} $$

이는 5번 던져서 3번 나온 것과 동일한 결과이므로, "trial을 많이 하는 것은 아무 의미가 없는 것이 아닐까?" 라는 의문을 가질 수 있다.

### Simple Error Bound

많은 trial이 어떠한 의미를 가지는지에 대해서는 error bound를 표현하는 다음의 Hoeffding's inequality 식을 통해 알 수 있다.

$$ P(|\hat\theta-\theta^*| \ge \varepsilon) \le 2e^{-2N\varepsilon} $$

위 식에서 각 변수에 대한 설명은 다음과 같다.

- $\hat\theta$ : 추정한 값
- $\theta^*$ : 참 값
- $\varepsilon$ : error bound
- $N$ : trial 수

많은 실험을 통해 $N$값이 커지면 $2e^{-2N\varepsilon}$ 값이 작아지고, 이를 통해 참값과 추정한 값의 차이가 error bound보다 커질 확률 $P(\|\hat\theta-\theta^*\| \ge \varepsilon)$값도 작아진다.

따라서, trial을 많이 하는 것은 특정 error 값의 범위를 넘어설 확률을 줄여준다는 의미가 있다.

# MAP

MAP(maximum a posteriori)는 베이즈 통계학에서 사후 확률의 최빈값을 가리킨다. MAP 또한 모수의 추정에 사용할 수 있지만, MLE에서는 어떤 사건이 일어날 확률을 가장 높이는 모수를 찾는 것에 비해, MAP를 이용한 모수 추정에서는 모수의 사전 확률과 결합된 확률을 고려한다는 점이 다르다.

이를 압정실험에서 생각해본다면, $D$라는 Data가 관측되었을 때 모수 $\theta$가 맞을 확률 $P(\theta \| D)$로 생각할 수 있다.

MAP는 다음의 식으로 구할 수 있다.

$$ P(\theta | D) = \dfrac{P(D | \theta) P(\theta)}{P(D)} \  \ \ (Posterior = \dfrac{Likelihood \times Prior\ Knowledge}{Normalizing\ Constant})$$ 

## Maximum a Posteriori Estimation

MAP를 이용한 모수의 추정은 다음의 식을 통해 수행할 수 있다.

$$ \hat\theta = argmax_\theta P(\theta | D) $$

압정 실험에서의 결과를 바탕으로 MAP를 이용해 모수를 추정해보자.

MAP식에서 normalizing constant를 고려하지 않으면, $P(\theta\|D)$는 $P(D \| \theta) P(\theta)$에 비례하므로 다음과 같다.

$$\begin{align*} P(\theta | D) & \propto P(D | \theta) P(\theta) \\ & \propto \theta^{a_H} (1-\theta)^{a_T}P(\theta) \end{align*}$$

여기서, $P(\theta)$는 확률을 모델링하는데 유용한 beta distribution을 사용하면 다음과 같이 나타낼 수 있다.

$$ P(\theta) = \dfrac{\theta^{\alpha-1} (1-\beta) ^{\beta - 1}}{B(\alpha, \beta)} \ \ \left(\because B(\alpha, \beta) = \dfrac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}, \Gamma(\alpha) = (\alpha - 1)! \right)$$

따라서, 식은 다음과 같이 정리할 수 있다.

$$\begin{align*} P(\theta | D) &\propto \theta^{a_H} (1-\theta)^{a_T} \theta^{\alpha - 1} (1-\theta)^{\beta - 1} \\ &\propto \theta^{a_H + \alpha - 1} (1-\theta)^{a_T + \beta - 1}\end{align*}$$

이제, 위 식의 우변을 $\theta$에 대해서 미분해서 최댓값을 구하면 $\hat\theta$를 구할 수 있다. 먼저, 식이 복잡하므로 log를 취하면 다음과 같다.

$$ \ln\left( \theta^{a_H + \alpha - 1} (1-\theta)^{a_T + \beta - 1} \right) = (a_H + \alpha - 1) \ln \theta + (a_T + \beta - 1) \ln(1-\theta)$$

위 식을 $\theta$에 대해 미분하면 다음과 같이 $\hat\theta$을 구할 수 있다.

$$ \dfrac{d}{d\theta} \left\{ (a_H + \alpha - 1) \ln\theta + (a_T + \beta - 1) \ln(1-\theta) \right\} = \dfrac{a_H + \alpha - 1}{\theta} - \dfrac{a_T + \beta - 1}{1-\theta} = 0 $$

$$ \therefore\ \hat\theta = \dfrac{a_H + \alpha - 1}{a_H + a_T + \alpha + \beta - 2} $$

위 값이 최솟값이 아닌 최댓값이라는 것을 확인하기 위해 한번 더 미분한 결과를 살펴보면, 다음과 같이 음의 값이 나오므로 최댓값이라는 것을 알 수 있다.

$$ \dfrac{d}{d\theta} \left\{ \dfrac{a_H + \alpha - 1}{\theta} - \dfrac{a_T + \beta - 1}{1-\theta} \right\} = \dfrac{-(a_H + \alpha - 1)}{\theta^2} - \dfrac{a_T + \beta - 1}{(1-\theta)^2} \lt 0 $$

### Conclusion

MAP를 통해 추정한 $ \hat\theta = \dfrac{a_H + \alpha - 1}{a_H + a_T + \alpha + \beta - 2} $의 식에서 다음의 2가지를 알 수 있다.

- $a_H, a_T$의 값이 커질수록 $\alpha, \beta$, 상수항들이 식에 미치는 영향이 줄어들기 때문에, 다음과 같이 MAP식은 MLE와 같아지게 된다.
  
$$\dfrac{a_H + \alpha - 1}{a_H + a_T + \alpha + \beta - 2} \rightarrow \dfrac{a_H}{a_H + a_T}$$

- $a_H, a_T$의 값이 작아질수록 사전 정보($\alpha, \beta$)는 중요해진다.

# Probability and Distributions

## Probability

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/ch1/probability.png" alt="probability" width="350px">
</p>

- Sample Space $\Omega$ : 모든 일어날 수 있는 experiments
- Event $E$ : sample space의 subset
- Probability $P(E)$ : Event $E$에서 일어날 수 있는 experiments의 수 / 모든 일어날 수 있는 experiments의 수

확률에 관한 수학적 특징들은 다음과 같다.

- $P(\Omega) = 1$
- $0 \le P(E) \le 1$
- $P(\varnothing) = 0$
- $P(E^{c}) = 1 - P(E)$
- $P(A \cup B) = P(A) + P(B) - P(A \cap B)$
- 모든 event가 mutually exclusive ($A \cap B = \varnothing$) 일 때, $P(E_1 \cup E_2 \cdots) = \sum\limits^\infty_{i=1} P(E_i)$ 
- 만약 $A \subseteq B$ 라면, $P(A) \le P(B)$

### Conditional Probability

조건부 확률(conditional probability)은 주어진 사건이 일어났다는 가정 하에 다른 한 사건이 일어날 확률을 뜻한다.

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/ch1/conditional_probability.jpg" alt="conditional_probability" width="350px">
</p>

Event $B$가 일어났다는 가정 하에(condition) event $A$가 일어날 조건부 확률 $P(A\|B)$는 다음과 같이 구할 수 있다. 즉, 위 그림에서 (빨간색 빗금 영역) / (B의 영역)이다.

$$P(A|B) = \dfrac{P(A \cap B)}{P(B)}$$

### Bayes' theorem

베이즈 정리(Bayes’ theorem)는 두 확률 변수의 사전 확률과 사후 확률 사이의 관계를 나타내는 정리다. 베이즈 확률론 해석에 따르면 베이즈 정리는 사전확률로부터 사후확률을 구할 수 있다.

베이즈 정리 식은 다음과 같다.

$$P(B|A) = \dfrac{P(A|B)P(B)}{P(A)}$$

위 식의 유도과정은 다음과 같다.

$$ P(A|B) = \dfrac{P(A \cap B)}{P(B)} \rightarrow P(A \cap B) = P(A|B)P(B) \\ P(B|A) = \dfrac{P(B \cap A)}{P(A)} \rightarrow P(B \cap A) = P(B|A)P(A) \\ P(A \cap B) = P(B \cap A) = P(A|B)P(B) = P(B|A) P(A) \\ \therefore P(B|A) = \dfrac{P(A|B)P(B)}{P(A)}$$

### Law of Total Probability

전체 확률의 법칙(law of total probability) 또는 전확률 정리는 조건부 확률과 관계된 법칙이다. 조건부 확률로부터 조건이 붙지 않은 확률을 계산할 때 쓸 수 있다. 또한 베이즈 정리 공식의 일부에 전확률 정리 공식이 들어간다.

전체 확률의 법칙 식은 다음과 같다. ($B_i$들이 모두 mutually exclusive events일 경우에 성립)

$$ P(A) = \sum\limits_n P(A|B_n)P(B_n) $$

아래 그림의 예에서, 전체 확률의 법칙 식을 유도해보면 다음과 같다.

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/ch1/total_probability.png" alt="total_probability" width="300px">
</p>

$$\begin{align*} P(A) &= P(A \cap B_1) + P(A \cap B_2) + P(A \cap B_3) + P(A \cap B_4) + P(A \cap B_5) \\ &= P(A|B_1)P(B_1) + P(A|B_2)P(B_2) + P(A|B_3)P(B_3) + P(A|B_4)P(B_4) + P(A|B_5)P(B_5) \\ &= \sum\limits_n P(A|B_n)P(B_n) \end{align*}$$

## Probability Distribution

확률 분포(probability distribution)는 확률 변수가 특정한 값을 가질 확률을 나타내는 함수를 의미한다. 예를 들어, 주사위를 던졌을 때 나오는 눈에 대한 확률변수가 있을 때, 그 변수의 확률분포는 다음과 같은 discrete uniform distribution이 된다.

### Normal Distribution

정규 분포(normal distribution) 또는 가우스 분포(Gaussian distribution)는 연속 확률 분포의 하나이다. 정규분포는 수집된 자료의 분포를 근사하는 데에 자주 사용되는데, 이는 중심극한정리(central limit theorem)에 의하여 독립적인 확률변수들의 평균은 정규분포에 가까워지는 성질이 있기 때문이다.

정규 분포의 표기법, pdf, 평균, 분산은 다음과 같다.
- Notation : $N(\mu, \sigma^2)$
- $f(x; \mu, \sigma) = \dfrac{1}{\sqrt{2 \pi \sigma^2}}e^{-(x-\mu)^2 / 2\sigma^2}, \ \ (-\infty \lt x \lt \infty)$
- Mean : $\mu$
- Variance : $\sigma^2$

정규 분포의 pdf 그래프(왼쪽) 및 cdf 그래프(오른쪽)는 다음과 같다. (붉은색 그래프는 표준정규분포 $N(0, 1)$)

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/ch1/Normal_Distribution_PDF.svg" alt="Normal_Distribution_PDF" width="400px">
  <img src="/assets/machine_learning_icmoon_1/ch1/Normal_Distribution_CDF.svg" alt="Normal_Distribution_CDF" width="400px">
</p>

### Beta Distribution

베타 분포(beta distribution)는 두 매개변수 $\alpha$와 $\beta$에 따라 \[0,1\] 구간에서 정의되는 연속 확률 분포의 하나이다. \[0,1\] 구간에서 정의되는 특성 때문에, 확률을 modeling하는 경우에 유용하게 사용된다.

베타 분포의 표기법, pdf, 평균, 분산은 다음과 같다.

- Notation : $Beta(\alpha, \beta)$
- $f(\theta; \alpha, \beta) = \dfrac{\theta^{\alpha-1} (1-\theta)^{\beta-1}}{B(\alpha, \beta)}, \ \ (0 \le \theta \le 1)\ \left( \because B(\alpha, \beta) = \dfrac{\Gamma(\alpha) + \Gamma(\beta)}{\Gamma(\alpha + \beta)}\ , \Gamma(\alpha) = (\alpha-1)! \right)$
- Mean : $\dfrac{\alpha}{\alpha + \beta}$
- Variance : $\dfrac{\alpha\beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}$

베타 분포의 pdf 그래프(왼쪽) 및 cdf 그래프(오른쪽)는 다음과 같다.

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/ch1/Beta_distribution_pdf.svg" alt="Beta_distribution_pdf" width="400px">
  <img src="/assets/machine_learning_icmoon_1/ch1/Beta_distribution_cdf.svg" alt="Beta_distribution_cdf" width="400px">
</p>

### Binomial Distribution

이항 분포(binomial distribution)는 연속된 n번의 독립적 시행에서 각 시행이 확률 p를 가질 때 정의되는 이산 확률 분포의 하나이다. 이러한 시행은 베르누이 시행이라고 불리기도 한다.

이항 분포의 표기법, pmf, 평균, 분산은 다음과 같다.

- Notation : $B(n, p)$
- $p(k; n, p) = \binom{n}{k} p^k (1-p)^{n-k}, \ \ (k=1, 2, \cdots, n) \ \left(\because \binom{n}{k} = \dfrac{n!}{k!(n-k)!} \right)$
- Mean : $np$
- Variance : $np(1-p)$

이항 분포의 pmf 그래프(왼쪽) 및 cdf 그래프(오른쪽)는 다음과 같다.

<p align="center">
  <img src="/assets/machine_learning_icmoon_1/ch1/Binomial_distribution_pmf.svg" alt="Binomial_distribution_pmf" width="400px">
  <img src="/assets/machine_learning_icmoon_1/ch1/Binomial_distribution_cdf.svg" alt="Binomial_distribution_cdf" width="400px">
</p>

### Multinomial Distribution

다항 분포(multinomial distribution)는 여러 개의 값을 가질 수 있는 독립 확률변수들에 대한 확률분포로, 여러 번의 독립적 시행에서 각각의 값이 특정 횟수가 나타날 확률을 정의한다. 다항 분포에서 차원이 2인 경우 이항 분포가 되므로, 이항분포의 일반화라고 볼 수 있다.

다항 분포의 표기법, pmf, 평균, 분산은 다음과 같다.

- Notation : $\text{Mult}(P)$, $P=<p_1, \cdots, p_k>$
- $p(x_1, \cdots, x_k; n, p_1, \cdots, p_k) = \dfrac{n!}{x_1! x_2! \cdots x_k!} p_1^{x_1} p_2^{x_2} \cdots p_k^{x_k}$
- Mean : $np_i$
- Variance : $np_i(1-p_i)$
