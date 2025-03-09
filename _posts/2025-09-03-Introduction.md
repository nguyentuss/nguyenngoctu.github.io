---
layout: post
title: "Introduction in Machine Learning"
date: 2025-03-09 10:00:00 +0700
categories: [Machine Learning]
tags: [AI,Introduction]
math: true
# author: "Your Name"
---

## Supervised Learning

The task $T$ is to learn a mapping $f$ from $x \in X$ to $y \in Y$. The
$x$ are also called the $\mathbf{features}$. The output $y$ is called
the $\mathbf{label}$. The experience $E$ is given in the form of a set
of $N$ input-output pair
$\mathcal{D} = \{(x_n,y_n)\}, n = 1 \xrightarrow{} N$ is called
$\textbf{training set}$. ($N$ is called the $\textbf{sample size}$). The
performance $P$ depends on the type of output we want to predict.

## Classification

In classification problem, the output space is a set of C labels called
$\textbf{classes}$, $Y = \{1,2,...,C\}$. The problem predicting the
class label given a input is called $\textbf{pattern recognition}$. The
goal of supervised learning in classification problem is want to predict
the label. A common way to measure the perform on this task is called
$\textbf{misclassification rate}$.


$\mathcal{L}(\boldsymbol{\theta}) \triangleq \frac{1}{N} \sum_{n=1}^{N} \mathbb{I}\left(y_n \neq f(x_n; \boldsymbol{\theta})\right)$

Where $\mathbb{I}(e)$ is indicator function, which return 1 if the
conditional is true, return 0 otherwise. We can also use the notation
$\textbf{loss function}$ $l(y,\hat{y})$.

$\mathcal{L}(\boldsymbol{\theta}) \triangleq \frac{1}{N} \sum_{n=1}^{N} \ell\left(y_n,  f(x_n; \boldsymbol{\theta})\right)$

## Regression

Similarly to the classification problem, but now the output in
regression are a real-value $y \in \mathbb{R}$ instead the discrete
value $y \in \{1,...,C\}$; this is known as $\textbf{regression}$. So we
need to use a different loss function. For regression, the most common
choice is to use quadratic loss, or $\ell_2$ loss (L2 normalization)

$\ell_2(y,\hat{y}) = (y - \hat{y})^2$

This penalizes large residuals $y-\hat{y}$. The empirical risk when use
quadratic risk is equal to the $\textbf{Mean squared error}$ or
$\textbf{MSE}$.

$MSE(\boldsymbol{\theta})=\frac{1}{N} \sum_{n=1}^{N}(y_n-f(x_n;\boldsymbol{\theta}))^2$

![image](assets/img/linear_regression.PNG){: width="700" height="500"}

An example of the regression model in 1d data, we can fix the data using
the $\textbf{linear regression}$ model.

$f(x;\boldsymbol{\theta})=b+wx$

Where w is the $\textbf{slope}$, b is the $\textbf{bias}$, and $\theta$
are the parameters of the model, we can minimize the sum square error.

$$\hat{\boldsymbol{\theta}}=\arg\min_{\boldsymbol{\theta}}MSE(\boldsymbol{\theta})$$

If we have multiple input features, we can write

$f(\mathbf{x};\boldsymbol{\theta})=b+w_1x_1+...+w_Dx_D=b+\mathbf{w^Tx}$

![image](assets/img/polynomial_regression.PNG){: width="700" height="500"}

We can improve the fit by using a $\textbf{Polynomial regression}$ model
with degree $\mathcal{D}$. This now have the form

$f(x;\mathbf{w})=\mathbf{w^T}\phi(x)$

Where $\phi(x)$ are the feature vector derived from the input

$\phi(x)=[1,x,x^2,...,x^D]$

## Overfitting

Empirical risk (training loss function)

$$
    \mathcal{L}(\boldsymbol{\theta}; \mathcal{D}_{\text{train}}) = 
    \frac{1}{|\mathcal{D}_{\text{train}}|} 
    \sum_{(\mathbf{x,y}) \in \mathcal{D}_{\text{train}}} 
    \ell(y, f(x; \boldsymbol{\theta}))
$$

The difference $\mathcal{L}( \boldsymbol{\theta};p^*)-\mathcal{L}(\boldsymbol{\theta};D_{train})$
called $\textbf{generalization gap}$.

If a model has a large generalization gap (i.e., low empirical risk but high population risk),
it is a sign that it is overfitting. In practice we don't know $p^*$.
However, we can partition the data we do have into two subsets, known as
the training set and the $\textbf{test set}$.

Then we can approximate the population risk using the $\textbf{test risk}$:

$$
\mathcal{L}(\boldsymbol{\theta}; \mathcal{D}_{\text{test}}) = 
    \frac{1}{|\mathcal{D}_{\text{test}}|} 
    \sum_{(\mathbf{x,y}) \in \mathcal{D}_{\text{test}}} 
    \ell(y, f(x; \boldsymbol{\theta}))
$$


![image](assets/img/overfitting.PNG){: width="700" height="500"}

We can make the training loss function to zero if we increase the degree
$\mathcal{D}$, but it will increase the testing loss function. The
purpose about the prediction accuarcy on new data, A model that fit the
training data but which is too much complex. It will call the
$\textbf{overfitting}$. If $\mathbf{D}$ is too small, the model will be
$\textbf{underfitting}$.

## Unsupervised learning

In supervised learning, we assume that each input $x$ in training set
have a output targets $y$, and our goal is to learn the input-output
mapping. Although this is useful, and difficult, supervised learning is
just a to find a mathematical function to fit the data points. So go
back to the unsupervised learning, we will opposed to just learning a
mapping. We just get $\mathcal{D} = \{(x_n : n = 1 : N)\}$ without any
ouputs $y_n$.This is called $\textbf{unsupervised learning}$.

From a probabilistic perspective, we can view the task of unsupervised
learning as fitting an unconditional model of the form $p(x)$, which can
generate new data x, whereas supervised learning involves fitting a
conditional model, $p(y|x)$, which specifies (a distribution over)
outputs given inputs.

Unsupervised learning avoids the needs of collect large labeled datasets
for training, which can be often time comsuming and expensive and does
not rely on manually labeled data or predefined categories, unlike
supervised learning, which learns from labeled examples. Instead,
unsupervised learning finds patterns, structures, or groupings in the
data based on inherent similarities or relationships.

## Clustering

![image](assets/img/clustering.PNG){: width="700" height="500"}

A simple example of unsupervised learning is the problem of finding
clusters in data. The goal is to partition the input into regions that
contain "similar" points.

## Self-supervised learning

$\textbf{Self-supervised learning}$ that automatically generating
$\textbf{labels}$ from $\textbf{unlabeled data}$. Try to learn to
predict a color image from a grayscale image, or to mask out words in a
sentence and then try to predict them given the surrounding context. The
hope is that the resulting predictor
$\hat{x}_1 = f(x_2;\boldsymbol{\theta})$. Where $x_2$ is the observed
input and $\hat{x}_1$ is the predict output, will learn useful features
from the data, that can be used in standard.

## Reinforcement learning

The system or agent has to learn how to interact with its environment.
For example, creating a bot playing Mario, a bot will interact and
integration with the world, run left or right or the bot will jump if
they see a block
stone.([`Click to see the detail`](https://pytorch.org/tutorials/intermediate/mario_rl_tutorial.html))

![image](assets/img/3typeML.PNG){: width="700" height="500"}

## Preprocessing discrete input data

### One-hot encoding

When we have categorical features, we need to scale it into numerical
values, so that the compute makes sense. The standard way to preprocess
such categorical variables is to use a $\textbf{one-hot encoding}$.
one-hot(x) = $[\mathbb{I}(x=1),...,\mathbb{I}(x=K)]$. If a variable x
has $K$ values (3 colors red,green,blue), the corresponding one-hot
vectors will be one-hot(red)=\[1,0,0\], one-hot(green)=\[0,1,0\],
one-hot(blue)=\[0,0,1\].

### Feature crosses

Converting the original datasets into a $\textbf{wide format}$, with
more many columns. Suppose we want to predict the fuel efficiency of a
vehicle given two categorical input variables:

-   $x_1$: The type of car (SUV, Truck, Family car).

-   $x_2$: The country of origin (USA, Japan).

Using one-hot encoding, we represent these variables as separate binary
indicators:
$$\phi(x) = [1, I(x_1 = S), I(x_1 = T), I(x_1 = F), I(x_2 = U), I(x_2 = J)]$$
However, this encoding does not capture interactions between the
features. For example, we expect trucks to be less fuel-efficient
overall, but perhaps trucks from the USA are even less efficient than
trucks from Japan. This cannot be captured using a simple linear model.
We define a new composite feature representing all possible pairs:
$$\text{(Car type, Country)} = \{(S, U), (T, U), (F, U), (S, J), (T, J), (F, J)\}$$

The new model becomes: $$f(x; w) = w^T \phi(x)$$. Express the equation we have: 

$$\begin{split}
    f(x; w) = & \; w_0 + w_1 I(x_1 = S) + w_2 I(x_1 = T) + w_3 I(x_1 = F) \\
    &+ w_4 I(x_2 = U) + w_5 I(x_2 = J) + w_6 I(x_1 = S, x_2 = U) \\
    &+ w_7 I(x_1 = T, x_2 = U) + w_8 I(x_1 = F, x_2 = U) \\
    &+ w_9 I(x_1 = S, x_2 = J) + w_{10} I(x_1 = T, x_2 = J)\\ 
    &+ w_{11} I(x_1 = F, x_2 = J)
\end{split}$$
