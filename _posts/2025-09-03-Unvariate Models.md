---
layout: post  
title: "Unvariate Models"  
date: 2025-03-09 10:00:00 +0700  
categories: [Machine Learning]  
tags: [AI,Probability]  
math: true  
# author: "Your Name"
---

# Probability Unvariate Models

------------------------------------------------------------------------

## Introduction

Calling $\textbf{sample space}$ $\mathcal{X}$ are all possible
experiences, and $\textbf{event}$ will be a subset of the
$\textbf{sample space}$.

### Union 

$$
Pr(A \land B)=Pr(A,B)
$$

If independent events

$$
Pr(A \land B)=Pr(A)Pr(B)
$$

We say a set of variables $X_1, . . . , X_n$ is (mutually) independent
if the joint can be written as a product of marginals for all subsets
$\{X_1,...,X_m\} \subseteq \{X_1,...,X_n\}$

$$
p(X_1,X_2,...,X_n)=\prod_{i=1}^{m}p(X_i)
$$

### Disjoint

$$
Pr(A \vee B)=Pr(A)+Pr(B)-Pr(A\land B)
$$

### Conditional probability

$$
Pr(B|A)\triangleq \frac{Pr(A,B)}{Pr(A)}
$$

If events A and B are conditionally independent given event C

$$
Pr(A,B|C)=Pr(A|C)Pr(B|C)
$$

Be careful, we say $X_1,X_2,X_3$ are mutually independent if the following
conditions hold: 

$$
\begin{split}    
        p(X_1,X_2,X_3)=p(X_1)p(X_2)p(X_3)\\
        ,p(X_1,X_2)=p(X_1)p(X_2) \\
        ,p(X_1,X_3)=p(X_1)p(X_3) \\
        ,p(X_2,X_3)=p(X_2)p(X_3) \\
\end{split}
$$

## Random variables

![image](assets/img/rv.PNG){: width="700" height="500"}

Given an experiment with sample space $\mathbb{S}$, a random
variable (r.v.) is a function mapping from the sample $\mathbb{S}$ to the
real value $\mathbb{R}$.

### Discrete random variables

If sample space $\mathbb{S}$ is finite or countable, it is called a
Discrete r.v. Denote probability of events in $\mathbb{S}$ and having
value x by $Pr(X=x)$. This is called the probability mass function or **pmf**,
a function which computes the probability of events that have the
value x.

$$
p(x)\triangleq Pr(X=x)
$$

The pmf satisfies
$0\leq p(x) \leq 1$ and $\sum_{x\in \mathcal{X}}p(x)=1$.

### Continuous random variables

If $X \in \mathbb{R}$, it is called a continuous r.v. The values no longer create a finite set of distinct possible values it can take on.

### Cumulative distribution function (cdf)

$$
P(x) \triangleq Pr(X\leq x)
$$

We can compute the probability of any interval

$$
P(a\leq x \leq b) = P(b)-P(a-1)
$$

In discrete r.v, the cdf will compute

$$
P(x)=\sum_{x\in \mathcal{X}}p(x)
$$

In continuous r.v, the cdf will compute

$$
P(x)=\int_{x\in \mathcal{X}}p(x)
$$

### Probability density function (pdf)

Define the pdf as the derivative of the cdf

$$
p(x) \triangleq \frac{d}{dx}P(x)
$$

As the size of the interval gets smaller, we can write

$$
Pr(x<X<x+dx) \approx p(x)dx
$$

### Quantiles

If the cdf $P$ is monotonically increasing, it has an inverse called the
$\textbf{inverse cdf}$. If $P$ is the cdf of $X$, then $P^{-1}(q)$ is the
value $x_q$ such that $Pr(X\leq x_q)=q$; this is called the q'th quantile of $P$.

### Sets of related random variables

Suppose we have two r.v. $X$ and $Y$. We can define the joint distribution

$$
p(x,y)=Pr(X=x,Y=y)
$$

for all possible values of x and y. We can represent all possible values by a 2D table. For example:

$$
\begin{array}{c|cc}
        p(X,Y) & Y = 0 & Y = 1 \\
        \hline
        X = 0 & 0.2 & 0.3 \\
        X = 1 & 0.3 & 0.2
\end{array}
$$

Here, $Pr(X=0,Y=1)=0.3$, and

$$
\sum_{x \in \mathcal{X},y \in \mathcal{Y}}p(x,y)=1
$$

### Moments of a distribution

The **mean** (or expected value) is defined for a continuous r.v. as:

$$
\mathbb{E}[X]=\int_{x\in \mathcal{X}}xp(x)dx
$$

If the integral is not finite, the mean is not defined. For discrete r.v, the mean is defined as:

$$
\mathbb{E}[X]=\sum_{x\in \mathcal{X}}xp(x)
$$

Since the mean is linear, we have the **linearity of expectation**:

$$
\mathbb{E}[aX+b]=a\mathbb{E}[X]+b
$$

For n random variables, the sum of expectations is:

$$
\mathbb{E}[\sum X_i]=\sum \mathbb{E}[X_i]
$$

If they are independent, the expectation of the product is:

$$
\mathbb{E}[\prod X_i]=\prod \mathbb{E}[X_i]
$$

When dealing with two or more dependent r.v's, we can compute the moment of one given the others:

$$
\mathbb{E}[X]=\mathbb{E}_Y[\mathbb{E}[X|Y]]
$$

A similar formula exists for the variance:

$$
\var[X]=\mathbb{E}_Y[\var[X|Y]] +\var_Y[\mathbb{E}[X|Y]]
$$

The **variance** is a measure of how "spread out" the distribution is, denoted as $\sigma^2$ and defined as:

$$
\begin{split}
\mathbb{V}[X] \triangleq \mathbb{E}[(X- \mu)^2] &=\int (x-\mu)^2p(x)dx \\
&=\int x^2p(x)dx+\mu\int p(x)dx-2\mu\int xp(x)dx\\
&=\mathbb{E}[X^2]-\mu^2
\end{split}
$$

![image](assets/img/Variance.png){: width="700" height="500"}

The **standard deviation** is given by:

$$
std[X]=\sqrt{\mathbb{V}[X]}=\sigma
$$

Lower deviation means the distribution is closer to the mean; higher deviation means it is further away.

The variance of a shifted and scaled version of a random variable is:

$$
\mathbb{V}[aX+b]=a^2\mathbb{V}[X]
$$

For n independent random variables, the variance of their sum is the sum of their variances:

$$
\mathbb{V}[\sum X_i]=\sum \mathbb{V}[X_i]
$$

The variance of their product is:

$$
\begin{split}
\mathbb{V}[\prod X_i]&= \mathbb{E}\left[\left(\prod X_i\right)^2\right]-\left(\mathbb{E}\left[\prod X_i\right]\right)^2 \\
&= \prod(\sigma_i^2 + \mu_i^2)-\prod\mu_i^2
\end{split}
$$

**Mode of a distribution**

The **mode** of a distribution is the value with the highest probability mass or density:

$$
\mathbf{x^*}= \arg \max_{\mathbf{x}} p(\mathbf{x})
$$

For multimodal distributions, the mode may not be unique.

## Bayes' Rule

### Bayes' Rule, and with extra conditioning 

$$
P({ A=a}|{ B=b})  = \frac{P({ B=a}|{ A=b})P({ A=a})}{P({ B=b})}
$$

$$
P({ A=a}|{ B=b}, { C=c}) = \frac{P({ B=b}|{ A=a}, { C=c})P({ A=a} | { C=c})}{P({ B=b} | { C=c})}
$$

The term $p(A)$ represents what we know about the possible values of $A$
before observing any data; this is the **prior distribution**. (If
$A$ has $K$ possible values, then $p(A)$ is a vector of $K$
probabilities summing to 1.) The term $p(B \mid A = a)$ represents the
distribution over possible outcomes of $B$ if $A = a$; this is the **observation distribution**.
Evaluated at the observed $b$, the function $p(B = b \mid A = a)$ is called the **likelihood**.

Multiplying the prior $p(A = a)$ by the likelihood $p(B = b \mid A = a)$ for each $a$
gives the unnormalized joint distribution $p(A = a, B = b)$. Normalizing by dividing by $p(B = b)$ (the **marginal likelihood**) gives:

$$
p(B = b) = \sum_{a' \in \mathcal{A}} p(A = a') p(B = b \mid A = a') 
    = \sum_{a' \in \mathcal{A}} p(A = a', B = b)
$$

**Odds Form of Bayes' Rule**

$$
\frac{P({ A}| { B})}{P({ A^c}| { B})} = \frac{P({ B}|{ A})}{P({ B}| { A^c})}\frac{P({ A})}{P({ A^c})}
$$

The *posterior odds* of $A$ equal the *likelihood ratio* times the *prior odds*.

## Bernoulli and Binomial Distributions

For an experiment tossing a coin with head probability $0\leq\theta \leq 1$, let $Y = 1$ denote heads and $Y = 0$ denote tails. So, $p(Y=1)=\theta$ and
$p(Y=0)=1-\theta$. This is the **Bernoulli distribution**, written as:

$$
Y \sim Ber(\theta)
$$

The pmf is defined as:

$$
\text{Ber}(y \mid \theta) =
    \begin{cases} 
        1 - \theta & \text{if } y = 0 \\[8pt]
        \theta & \text{if } y = 1
    \end{cases}
$$

It can also be written as:

$$
\text{Ber}(y \mid \theta) \triangleq \theta^y(1-\theta)^{1-y}
$$

A Bernoulli trial is a special case of the **Binomial distribution**. Tossing a coin N times gives a set of N Bernoulli trials, denoted
$y_n \sim Ber(\cdot\mid\theta)$. Let $s = \sum_{n=1}^{N}\mathbb{I}(y_n = 1)$ be the number of heads. Then, $s$ follows a binomial distribution:

$$
Bin(s \mid\theta,N) \triangleq \binom{N}{s}\theta^s(1-\theta)^{N-s}
$$

When predicting a binary variable $y \in \{0, 1\}$ given inputs $x \in X$, we use a conditional distribution of the form:

$$
p(y|\inp,\theta) = Ber(y\mid f(\inp;\theta))
$$

To ensure $0\leq f(\inp;\theta)\leq1$, we often write:

$$
p(y|\inp,\theta) = Ber(y\mid \sigma (f(\inp;\theta)))
$$

where $\sigma()$ is the **sigmoid** (or logistic) function, defined as:

$$
\sigma(a) \triangleq \frac{1}{1+e^{-a}}= \frac{e^a}{1 + e^a}
$$

with $a = f(\inp,\theta)$. Its inverse is the **logit function**:

$$
a=logit(p) =\sigma^{-1}(p) \triangleq \log\frac{p}{1-p}
$$

Thus, the sigmoid transforms a function from $\mathbb{R}$ into a probability in $[0,1]$, while the logit transforms a probability into a real number, making it easier to model with linear regression techniques.

Some useful properties of these functions:

$$
\sigma(x) \triangleq \frac{1}{1 + e^{-x}} = \frac{e^x}{1 + e^x}
$$

$$
\frac{d}{dx} \sigma(x) = \sigma(x)(1 - \sigma(x))
$$

$$
1 - \sigma(x) = \sigma(-x)
$$

$$
\sigma_+(x) \triangleq \log(1 + e^x) \triangleq \text{softplus}(x)
$$

$$
\frac{d}{dx} \sigma_+(x) = \sigma(x)
$$

In particular, note an issue often encountered with activation functions: obtaining zero (or near zero) gradient during backpropagation.

When $x = 0$:

  
$$
f(0) = \frac{1}{1 + e^0} = \frac{1}{2}
$$

  
$$
f'(0) = \frac{1}{2} \cdot \left(1 - \frac{1}{2} \right) = \frac{1}{4} = 0.25
$$

This is **not very small** (but smaller than 1).

When $x \gg 0$ (large positive):

$$
f(x) \to 1, \quad f'(x) = 1 \cdot (1 - 1) = 0
$$

**Gradient approaches 0**.

When $x \ll 0$ (large negative):

$$
f(x) \to 0, \quad f'(x) = 0 \cdot (1 - 0) = 0
$$

**Gradient also approaches 0**.

## Categorical and Multinomial Distributions

For more than 2 classes, the **categorical distribution** represents a distribution over a finite set of labels $y \in \{1,\dots,C\}$, generalizing the Bernoulli to $C > 2$. Its pmf is:

$$
\text{Cat}(y \mid \theta) \triangleq \prod_{c=1}^{C} \theta_c^{\mathbb{I}(y=c)}
$$

In other words, $p(y=c \mid \theta)=\theta_c$, where the parameters are constrained so that $0\leq\theta_c\leq1$.

By converting $y$ into a **one-hot vector** with $C$ elements (e.g., for $C=3$, the classes are encoded as $(1,0,0)$, $(0,1,0)$, and $(0,0,1)$), we can view a fair 6-sided die with outcomes:

$$
y \in \{1, 2, 3, 4, 5, 6\}
$$

Each face has an equal probability, so:

$$
\theta = \left( \frac{1}{6}, \frac{1}{6}, \frac{1}{6}, \frac{1}{6}, \frac{1}{6}, \frac{1}{6} \right)
$$

If we roll a 3, the one-hot encoding is:

$$
(0, 0, 1, 0, 0, 0)
$$

Substituting into the PMF:

$$
P(y = 3 \mid \theta) = \theta_3 = \frac{1}{6}
$$

For a biased die with:

$$
\theta = (0.10, 0.15, 0.30, 0.20, 0.15, 0.10)
$$

the probability of rolling a 3 becomes:

$$
P(y = 3 \mid \theta) = \theta_3 = 0.30
$$

The categorical distribution is a special case of the **multinomial distribution**. If we observe $N$ categorical trials, $y_n \sim Cat(\cdot\mid\theta)$ for $n = 1,\dots,N$, and define $y_c = \sum_{n=1}^{N}\mathbb{I}(y_n = c)$, then the vector $\mathbf{y}$ follows:

$$
\mathcal{M}(\mathbf{y} \mid N, \theta) \triangleq 
\binom{N}{y_1 \dots y_C} \prod_{c=1}^{C} \theta_c^{y_c}
$$

In the conditional case, we can define

$$
p(y \mid x, \theta) = \text{Cat}(y \mid f(x; \theta))
$$

or equivalently,

$$
p(y \mid x, \theta) = \mathcal{M}(y \mid 1, f(x; \theta))
$$

requiring that $0 \leq f_c(x; \theta) \leq 1$ and $\sum_{c=1}^{C} f_c(x; \theta) = 1$.

To avoid forcing $f$ to directly predict a probability vector, it is common to pass its output into the **softmax** function (or **multinomial logit**), defined as:

$$
softmax(\mathbf{a})\triangleq
\begin{bmatrix}
\frac{e^{a_1}}{\sum_{c'=1}^{C}e^{a_{c'}}}, \dots,
\frac{e^{a_C}}{\sum_{c'=1}^{C}e^{a_{c'}}}
\end{bmatrix}
$$

This converts $\mathbb{R}^C$ into a probability vector in $[0,1]^C$. One weakness is that if

$$
p_c=\frac{e^{a_c}}{Z(\mathbf{a})}=\frac{e^{a_c}}{\sum_{c'=1}^{C}e^{a_{c'}}}
$$

for logits $\mathbf{a}=f(\inp,\theta)$, then for extreme values like $\mathbf{a}=(1000,1000,1000)$ or $\mathbf{a}=(-1000,-1000,-1000)$, numerical issues (overflow or underflow) occur. To avoid this, we use the trick:

$$
\log\sum_{c'=1}^{C}e^{a_{c'}} = m +\log\sum_{c'=1}^{C}e^{a_{c'}-m}
$$

with $m=\max_c a_c$, ensuring the largest value is zero. This is the **log-sum-exp trick**.

Thus, we have:

$$
\begin{aligned}
p(y=c|\inp)&= \frac{\exp(a_c-m)}{\sum_{c'} \exp(a_{c'}-m)} \\
&=e^{a_c-lse(\mathbf{a})}
\end{aligned}
$$

The loss for the softmax function is given by:

$$
J(\weight; \inp, \out) = - \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ji} \log (a_{ji})
$$

or equivalently,

$$
J(\weight; \inp, \out) = - \sum_{i=1}^{N} \sum_{j=1}^{C} y_{ji} \log \left( \frac{\exp(\weight_j^T \inp_i)}{\sum_{k=1}^{C} \exp(\weight_k^T \inp_i)} \right)
$$

For a single data point $(\inp_i, \out_i)$, the loss is:

$$
J_i(\weight) \triangleq J(\weight; \inp_i, \out_i) = - \sum_{j=1}^{C} y_{ji} \log \left( \frac{\exp(\weight_j^T \inp_i)}{\sum_{k=1}^{C} \exp(\weight_k^T \inp_i)} \right)
$$

Which can be rewritten as:

$$
J_i(\weight) = - \sum_{j=1}^{C} y_{ji} \weight_j^T \inp_i + \log \left( \sum_{k=1}^{C} \exp(\weight_k^T \inp_i) \right)
$$

The gradient for each column $j$ is computed as:

$$
\frac{\partial J_i(\weight)}{\partial \weight_j} = - y_{ji} \inp_i + \frac{\exp(\weight_j^T \inp_i) \inp_i}{\sum_{k=1}^{C} \exp(\weight_k^T \inp_i)}
$$

which simplifies to:

$$
\frac{\partial J_i(\weight)}{\partial \weight_j} = \inp_i (a_{ji} - y_{ji})
$$

or

$$
\frac{\partial J_i(\weight)}{\partial \weight_j} = \inp_i e_{ji}, \quad \text{where } e_{ji} = a_{ji} - y_{ji}
$$

Collecting for all columns, we have:

$$
\frac{\partial J_i(\weight)}{\partial \weight} = \inp_i [e_{i1}, e_{i2}, \dots, e_{iC}] = \inp_i e_i^T
$$

Thus, the full gradient is:

$$
\frac{\partial J(\weight)}{\partial \weight} = \sum_{i=1}^{N} \inp_i e_i^T = \inp E^T
$$

where $E = A - Y$. This compact gradient expression is useful for **Batch Gradient Descent**, **Stochastic Gradient Descent (SGD)**, and **Mini-batch Gradient Descent**.

Assuming SGD, the weight matrix $\weight$ is updated as:

$$
\weight = \weight + \eta \inp_i (y_i - a_i)^T
$$

## Univariate Gaussian (Normal) Distribution

![image](assets/img/gauss_plot.PNG){: width="700" height="500"}

The cdf of the Gaussian is defined by:

$$
  \phi(y;\mu,\variance)\triangleq\int_{-\infty}^{y}\normal(z,\mu,\variance)dz
$$

It can be implemented using:

$$
  \phi(y;\mu,\variance)=\frac{1}{2}\Bigl[1+erf\Bigl(\frac{z}{\sqrt{2}}\Bigr)\Bigr]
$$

where $z=(y-\mu)/\sigma$ and $erf(u)$, the **error function**, is defined as:

$$
erf(u)\triangleq\frac{2}{\sqrt{\pi}}\int_{0}^{u}e^{-t}dt
$$

The pdf of the Gaussian is given by:

$$
\normal(y\mid\mu,\variance)=\frac{1}{\sqrt{2\pi\variance}}e^{-\frac{1}{2\variance}(y-\mu)^2}
$$

where $\sqrt{2\pi\variance}$ is the normalization constant ensuring the density integrates to 1.

The mean of the distribution is:

$$
\mean[\normal(\cdot\mid\mu,\variance)]=\mu
$$

and the standard deviation is:

$$
std[\normal(\cdot\mid\mu,\variance)]=\sigma
$$

It is common to parameterize the Gaussian as a function of input variables to create a conditional density model of the form:

$$
p(y\mid\inp,\theta)=\normal\Bigl(y\mid f_{\mu}(\inp;\theta),f_{\sigma}(\inp;\theta)^2\Bigr)
$$

where $f_{\mu}(\inp;\theta)\in\R$ predicts the mean and $f_{\sigma}(\inp;\theta)\in\R_+$ predicts the variance.

Assuming fixed variance independent of the input (homoscedastic regression) and a linear mean, we have:

$$
p(y\mid\inp,\theta)=\normal\Bigl(y\mid \weight^T\inp+b,\variance\Bigr)
$$

with $\theta=(\weight,b,\variance)$. If the variance depends on the input (heteroskedastic regression), then:

$$
p(y\mid\inp,\theta)=\normal\Bigl(y\mid \weight_{\mu}^T\inp+b,\sigma_+\bigl(\weight_\sigma^T\inp\bigr)\Bigr)
$$

with $\theta=(\weight_{\mu},\weight_\sigma)$, and $\sigma_+(x)$ being the softplus function mapping $\R$ to $\R_+$.

The Gaussian distribution is popular because it is parameterized only by mean and variance, and the sum of independent random variables approximates a Gaussian (Central Limit Theorem), making it useful for modeling noise.

When the variance approaches 0, the Gaussian becomes infinitely narrow:

$$
\lim_{\sigma\rightarrow0}\N(y\mid\mu,\variance)\rightarrow\delta(y-\mu)
$$

where $\delta$ is the Dirac delta function, defined as:

$$
\delta(x) = 
    \begin{cases}
        +\infty & \text{if } x=0 \\[8pt]
        0  & \text{if } x \neq 0
    \end{cases}
$$

and similarly,

$$
\delta_y(x) = 
    \begin{cases}
        +\infty & \text{if } x=y \\[8pt]
        0  & \text{if } x \neq y
    \end{cases}
$$

with $\delta_y(x) = \delta(x-y)$.

## Some Common Other Univariate Distributions

![image](assets/img/common-uni-distribution.PNG){: width="700" height="500"}

### Student t Distribution

The Student t distribution is given by:

$$
\mathcal{T}(y | \mu, \sigma^2, \nu) \propto 
\left[ 1 + \frac{1}{\nu} \left( \frac{y - \mu}{\sigma} \right)^2 \right]^{-\frac{\nu + 1}{2}}
$$

It has heavier tails than the Normal distribution, meaning more probability mass is closer to the mean. Here, $\mu$ is the mean, $\sigma > 0$ is the scale parameter, and $\nu$ is the degrees of freedom. Its properties include:

$$
\text{mean} = \mu, \quad \text{mode} = \mu, \quad \text{var} = \frac{\nu \sigma^2}{(\nu - 2)}
$$

The mean is defined if $\nu > 1$, and the variance if $\nu > 2$. For $\nu \gg 5$, it approaches a Gaussian. A common choice is $\nu = 4$.

### Cauchy Distribution

For $\nu=1$, the Student t distribution becomes the **Cauchy** (or **Lorentz**) distribution:

$$
\mathcal{C}(x\mid\mu,\gamma)=\frac{1}{\pi\gamma}
\left[1+\left(\frac{x-\mu}{\gamma}\right)^2\right]^{-1}
$$

Due to the heavy tails, it has notable robustness properties.

The **Half Cauchy** distribution (with $\mu=0$) is defined as:

$$
\mathcal{C}_+(x\mid \gamma) \triangleq \frac{2}{\gamma\pi}
\left[1+\left(\frac{x}{\gamma}\right)^2\right]^{-1}
$$

This is useful in Bayesian modeling when a positive real value is needed.

### Laplace Distribution

Also known as the **double-sided exponential** distribution, the Laplace distribution has the pdf:

$$
\text{Laplace}(y | \mu, b) \triangleq \frac{1}{2b} \exp \left( -\frac{|y - \mu|}{b} \right)
$$

with $\mu$ as the location parameter and $b > 0$ as the scale parameter. Its properties are:

$$
\text{mean} = \mu, \quad \text{mode} = \mu, \quad \text{var} = 2b^2
$$

![image](assets/img/beta-gamma-distribution.PNG){: width="700" height="500"}

### Beta Distribution

The **Beta distribution** is supported on the interval $[0,1]$ and is defined as:

$$
\text{Beta}(x | a, b) = \frac{1}{B(a, b)} x^{a-1} (1 - x)^{b-1}
$$

where the **beta function** is:

$$
B(a, b) \triangleq \frac{\Gamma(a) \Gamma(b)}{\Gamma(a + b)}
$$

and $\Gamma(a)$, the Gamma function, is defined as:

$$
\Gamma(a) \triangleq \int_{0}^{\infty} x^{a-1} e^{-x} \,dx
$$

We require $a, b > 0$. For $a = b = 1$, this is the uniform distribution. When both $a$ and $b$ are less than 1, the distribution is bimodal with spikes at 0 and 1; when both are greater than 1, it is unimodal. Its properties include:

$$
\text{mean} = \frac{a}{a+b}, \quad \text{mode} = \frac{a-1}{a+b-2}, \quad \text{var} = \frac{ab}{(a+b)^2(a+b+1)}
$$

## Transformation of Random Variables

Suppose $\inp\sim p()$ is a random variable, and $\out=f(\inp)$ is a transformation of it. We discuss how to compute $p(\out)$.

### Discrete Case

For discrete r.v's, the pmf of $\out$ is obtained by summing the pmf of all $\inp$ such that $f(\inp)=\out$:

$$
p_{\out}(\out)=\sum_{\inp:f(\inp)=\out}p_{\inp}(\inp)
$$

For example, if $f(\inp)=1$ when $\inp$ is even and $0$ otherwise, and $\inp$ is uniformly distributed over $\{1,2,...,10\}$, then $p_{\out}(1)=0.5$ and $p_{\out}(0)=0.5$.

### Continuous Case

For continuous r.v's, we work with the cdf:

$$
P_{\out}(\out)=Pr(Y\leq \out) = Pr(f(\inp) \leq \out)=Pr(\inp\in\{x \mid f(x) \leq \out\})
$$

If $f$ is invertible, differentiating the cdf yields the pdf; if not, Monte Carlo approximation may be used.

### Invertible Transformation (Bijections or One-to-One)

![image](assets/img/bijection.PNG){: width="700" height="500"}

For a monotonic (and hence invertible) function, if $x\sim Uni(0,1)$ and $y=f(x)=2x+1$, then for general $p_x(x)$ and any monotonic $f:\R\rightarrow\R$, let $g=f^{-1}$ with $y=f(x)$ and $x=g(y)$. Then:

$$
P_y(y)=Pr(f(X)\leq y)=Pr(X\leq f^{-1}(y))=P_x(f^{-1}(y))=P_x(g(y))
$$

Differentiating gives:

$$
p_y(y) \triangleq  \frac{d}{dy}P_y(y)=p_x(g(y))\left|\frac{dx}{dy}\right|
$$

This is known as the **change of variables** formula.

For multivariate cases, if $f:\R^n\rightarrow\R^n$ is invertible with inverse $g$, then:

$$
p_y(\out)=p_x(\inp)\left|det\Bigl[\mathbf{J}_g(\out)\Bigr]\right|
$$

where $\mathbf{J}_g=\frac{d\mathbf{g(y)}}{d\mathbf{y^T}}$ is the Jacobian.

### Convolution Theorem

![image](assets/img/convolution.PNG){: width="700" height="500"}

For $Y=X_1+X_2$ with independent r.v's $X_1$ and $X_2$, in the discrete case:

$$
p(Y=y)=\sum_{x=-\infty}^{\infty}p(X_1=x)p(X_2=y-x)
$$

In the continuous case:

$$
p(y)=\int p_1(x_1)p_2(y-x_1)dx_1
$$

This is written as:

$$
p = p_1 \circledast p_2
$$

where $\circledast$ is the **convolution** operator.

### Central Limit Theorem

![image](assets/img/CLT.PNG){: width="700" height="500"}

The Central Limit Theorem states that the sum of $N$ independent and identically distributed (i.i.d.) random variables (regardless of their original distribution) will approximate a Gaussian distribution as $N$ increases.

### Monte Carlo Approximation

![image](assets/img/monte-carlo.PNG){: width="700" height="500"}

Suppose $\inp$ is a random variable and $\out=f(\inp)$. When computing $p(\out)$ directly is difficult, one can approximate it by drawing a large number of samples from $p(x)$, computing $y_s=f(x_s)$, and forming the empirical distribution:

$$
p_s(y) \;\triangleq\; \frac{1}{N_s} \sum_{s=1}^{N_s} \delta\bigl(y - y_s\bigr)
$$
