---
layout: post  
title: "Multivariate Models"  
date: 2025-03-10 00:00:00 +0700  
categories: [Machine Learning]  
tags: [AI,Probability]  
math: true  
# author: "Your Name"
---


# Multivariate Models

------------------------------------------------------------------------

## Joint distributions for multiple random variables

### Covariance

The **covariance** between two rv's \(X\) and \(Y\) measures the **direction** of the **linear relationship** to which \(X\) and \(Y\) are (linearly) related. It quantifies how the random variables change together.

-   Positive: If one increases, the other also increases.
-   Negative: If one increases while the other decreases.
-   Zero: There is no relationship between the variables.

$$
\cov[X,Y]\triangleq\mean[(X-\mean[X])(Y-\mean[Y])]=\mean[XY]-\mean[X]\mean[Y]
$$

If \(\mathbf{x}\) is a \(D\)-dimensional random vector, its **covariance matrix** is defined as

$$
\cov[\mathbf{x}]\triangleq\mean\Bigl[(\mathbf{x}-\mean[\mathbf{x}])(\mathbf{x}-\mean[\mathbf{x}])^T\Bigr] \triangleq \mathbf{Cov} =
$$

$$
\begin{pmatrix}
        \var[X_1] & \cov[X_1,X_2] & \cdots & \cov[X_1,X_D] \\
        \cov[X_2,X_1] & \var[X_2] & \cdots &\cov[X_2,X_D] \\
        \vdots & \vdots & \ddots & \vdots \\ 
        \cov[X_D,X_1] & \cov[X_D,X_2] & \cdots & \var[X_D]
\end{pmatrix}
$$

Covariance itself is the variance of the distribution, from which we can get the important result

$$
\mean[\mathbf{x}\mathbf{x}^T] = \mathbf{Cov} + \Mean \Mean^T
$$

Another useful result is that the covariance of a linear transformation

$$
\cov[\mathbf{A}\mathbf{x}+b]=\mathbf{A}\cov[\mathbf{x}]\mathbf{A}^T
$$

The **cross-covariance** between two random vectors is defined by

$$
\cov[\mathbf{x},\mathbf{y}]=\mean\Bigl[(\mathbf{x}-\mean[\mathbf{x}])(\mathbf{y}-\mean[\mathbf{y}])^T\Bigr]
$$

### Correlation

![image](assets/img/correlation.PNG){: width="700" height="500"}

Covariance can range from negative to positive infinity. Sometimes it is more convenient to work with a normalized measure that is bounded. The **correlation coefficient** between \(X\) and \(Y\) is defined as

$$
\rho\triangleq\corr[X,Y]\triangleq\frac{\cov[X,Y]}{\sqrt{\var[X]\var[Y]}}
$$

While covariance can be any real value, correlation is always between \(-1\leq\rho\leq1\). In the case of a vector \(\mathbf{x}\) of related variables, the correlation matrix is given by

$$
\begin{aligned}
    \corr(\mathbf{x})=
    \begin{pmatrix}
    1 & \frac{\mathbb{E}[(X_1 - \mu_1)(X_2 - \mu_2)]}{\sigma_1 \sigma_2} & \cdots & \frac{\mathbb{E}[(X_1 - \mu_1)(X_D - \mu_D)]}{\sigma_1 \sigma_D} \\
    \frac{\mathbb{E}[(X_2 - \mu_2)(X_1 - \mu_1)]}{\sigma_2 \sigma_1} & 1 & \cdots & \frac{\mathbb{E}[(X_2 - \mu_2)(X_D - \mu_D)]}{\sigma_2 \sigma_D} \\
    \vdots & \vdots & \ddots & \vdots \\ 
    \frac{\mathbb{E}[(X_D - \mu_D)(X_1 - \mu_1)]}{\sigma_D \sigma_1} & \frac{\mathbb{E}[(X_D - \mu_D)(X_2 - \mu_2)]}{\sigma_D \sigma_2} & \cdots & 1
    \end{pmatrix}
\end{aligned}
$$

But note that **uncorrelated does not imply independent**. For example, let \(X\sim\Unif(-1,1)\) and \(Y=X^2\). Even though \(\cov[X,Y]=0\) and \(\corr[X,Y]=0\), \(Y\) clearly depends on \(X\). There are many datasets where \(X\) and \(Y\) exhibit a clear dependence, yet the correlation coefficient is 0.

### Simpson Paradox

![image](assets/img/SimpsonParadox.PNG){: width="700" height="500"}

Simpson's paradox states that a statistical trend or relationship observed in several different groups of data can disappear or reverse when these groups are combined.

## The multivariate Gaussian (normal) distribution

The multivariate Gaussian (normal) distribution generalizes the univariate Gaussian to multiple dimensions.

$$
\N(\mathbf{y};\Mean,\Cov) \triangleq \frac{1}{(2\pi)^{D/2}|\Cov|^{1/2}}\exp\left(-\frac{1}{2}(\mathbf{y}-\Mean)^T\Cov^{-1}(\mathbf{y}-\Mean)\right)
$$

where \(\Mean = \mean[\mathbf{y}] \in \mathbb{R}^D\) is the mean vector, and \(\Cov=\cov[\mathbf{y}]\) is the \(D\times D\) **covariance matrix**.

![image](assets/img/MVN.PNG){: width="700" height="500"}

In 2D, the MVN is known as the **Bivariate Gaussian** distribution. The pdf can be defined as \(\mathbf{y}\sim\N(\Mean,\Cov)\), where \(\Mean,\mathbf{y}\in\mathbb{R}^2\) and

$$
\Cov=
\begin{pmatrix}
        \sigma_1^2  & \rho\sigma_1\sigma_2  \\
        \rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}
$$

A diagonal covariance has \(D\) parameters with 0s in the off-diagonals. A spherical covariance (isotropic variance matrix) is of the form \(\Cov=\sigma^2\mathbf{I}\).

### Mahalanobis distance

The log-pdf of a **multivariate Gaussian** is

$$
\log p(\mathbf{y}\mid\Mean,\Cov)=-\frac{1}{2}(\mathbf{y}-\Mean)^T\Cov^{-1}(\mathbf{y}-\Mean) + \text{const}
$$

The quadratic term 

$$
(\mathbf{y}-\Mean)^T\Cov^{-1}(\mathbf{y}-\Mean)
$$

represents the **Mahalanobis distance squared**:

$$
\Delta^2 \triangleq (\mathbf{y}-\Mean)^T\Cov^{-1}(\mathbf{y}-\Mean)
$$

This distance measures how far a point is from the distribution’s mean, taking into account the shape of the data distribution. Contours of constant (log) probability correspond to contours of constant Mahalanobis distance.

To gain insight into these contours, consider the eigendecomposition of \(\Cov\):

$$
\Cov = \sum_{d=1}^{D} \lambda_d u_d u_d^T
$$

where \(u_d\) are the eigenvectors of \(\Cov\) and \(\lambda_d\) are the eigenvalues. Since \(\Cov\) is positive definite, its inverse is

$$
\Cov^{-1} = \sum_{d=1}^{D} \frac{1}{\lambda_d} u_d u_d^T
$$

Defining \(z_d\triangleq u_d^T(\mathbf{y}-\Mean)\) so that \(z=U(\mathbf{y}-\Mean)\), the Mahalanobis distance can be written as

$$
(\mathbf{y}-\Mean)^T\Cov^{-1}(\mathbf{y}-\Mean)=\sum_{d=1}^{D}\frac{z_d^2}{\lambda_d}
$$

### Marginals and conditionals of an MVN

Suppose \(\mathbf{y} = (y_1, y_2)\) is jointly Gaussian with parameters

$$
\bm{\mu} =
\begin{pmatrix}
\mu_1 \\
\mu_2
\end{pmatrix}, \quad
\bm{\Sigma} =
\begin{pmatrix}
\Sigma_{11} & \Sigma_{12} \\
\Sigma_{21} & \Sigma_{22}
\end{pmatrix}, \quad
\bm{\Lambda} = \bm{\Sigma}^{-1} =
\begin{pmatrix}
\Lambda_{11} & \Lambda_{12} \\
\Lambda_{21} & \Lambda_{22}
\end{pmatrix}
$$

where \(\bm{\Lambda}\) is the **precision matrix**. Then the marginals are given by

$$
\begin{aligned}
    p(y_1)&=\N(y_1\mid\mu_1,\Sigma_{11}) \\ 
    p(y_2)&=\N(y_2\mid\mu_2,\Sigma_{22})
\end{aligned}
$$

and the conditional distribution is

$$
\begin{aligned}
    p(y_1\mid y_2)&=\N(y_1\mid\mu_{1|2},\Sigma_{1|2}) \\
    \mu_{1|2}&=\mu_1+\Sigma_{12}\Sigma_{22}^{-1}(y_2-\mu_2) \\
    \Sigma_{1|2}&=\Sigma_{11}-\Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}
\end{aligned}
$$

Let us consider a 2D example. The covariance matrix is

$$
\bm{\Sigma} =
\begin{pmatrix}
\sigma_1^2 & \rho \sigma_1 \sigma_2 \\
\rho \sigma_1 \sigma_2 & \sigma_2^2
\end{pmatrix}
$$

The marginal \(p(y_1)\) is a 1D Gaussian obtained by projecting the joint distribution onto the \(y_1\) axis:

$$
p(y_1) = \mathcal{N}(y_1 \mid \mu_1, \sigma_1^2)
$$

Suppose we observe \(Y_2 = y_2\); the conditional \(p(y_1 \mid y_2)\) is given by

$$
p(y_1 \mid y_2) = \mathcal{N} \Bigl( y_1 \Big| \mu_1 + \frac{\rho \sigma_1 \sigma_2}{\sigma_2^2} (y_2 - \mu_2), \, \sigma_1^2 - \frac{(\rho \sigma_1 \sigma_2)^2}{\sigma_2^2} \Bigr)
$$

If \(\sigma_1 = \sigma_2 = \sigma\), then

$$
p(y_1 \mid y_2) = \mathcal{N} \Bigl( y_1 \Big| \mu_1 + \rho (y_2 - \mu_2), \, \sigma^2 (1 - \rho^2) \Bigr)
$$

For example, if \(\rho = 0.8\), \(\sigma_1 = \sigma_2 = 1\), \(\mu_1 = \mu_2 = 0\), and \(y_2 = 1\), then \(\mathbb{E}[y_1 \mid y_2 = 1] = 0.8\) and

$$
\text{Var}(y_1 \mid y_2 = 1) = 1 - 0.8^2 = 0.36.
$$

If \(\rho = 0\), then \(p(y_1 \mid y_2) = \mathcal{N}(y_1 \mid \mu_1, \sigma_1^2)\) since \(y_2\) conveys no information about \(y_1\).

### Example: Missing data

Suppose we sample \(N=10\) vectors from an 8D Gaussian, and then hide 50% of the data. We compute the missing entries given the observed entries and the true model parameters. For each example \(n\) in the data matrix, we compute

$$
p(\mathbf{y}_{n,h}\mid\mathbf{y}_{n,v},\theta)
$$

where for data point \(n\), \(v\) denotes indices of the visible features and \(h\) the indices of the hidden features, with \(\theta=(\Mean,\Cov)\). From this marginal distribution for each missing variable \(i\in\mathbf{h}\), we compute the posterior mean

$$
\bar{y}_{n,i} = \mathbb{E} \Bigl[ y_{n,i} \mid \mathbf{y}_{n,v}, \theta \Bigr].
$$

The posterior mean represents our "best guess" for that entry (minimizing expected squared error), and the variance

$$
\var\Bigl[y_{n,i}\mid \mathbf{y}_{n,v}, \theta\Bigr]
$$

measures our confidence. A small variance indicates that the prediction (posterior mean) is likely close to the actual value, whereas a large variance indicates higher uncertainty.

## Linear Gaussian systems

We now extend this approach to handle noisy observations. Let \(z\in\mathbb{R}^L\) be an unknown data value, and \(y\in\mathbb{R}^D\) be a noisy measurement of \(z\). We assume they are related via the joint distribution

$$
\begin{aligned}
    p(z)&=\N(z\mid\mu_z,\Cov_z) \\
    p(y\mid z)&=\N(y\mid Wz+b,\Cov_y)
\end{aligned}
$$

where \(W\) is a \(D\times L\) matrix. This constitutes a **linear Gaussian system**.

The joint distribution \(p(z,y)=p(z)p(y\mid z)\) is an \((L+D)\)-dimensional Gaussian with mean

$$
\mu =
\begin{pmatrix}
\mu_z \\
W \mu_z + b
\end{pmatrix}
$$

and covariance

$$
\Cov =
\begin{pmatrix}
\Cov_z & \Cov_z W^T \\
W \Cov_z & \Cov_y + W \Cov_z W^T
\end{pmatrix}
$$

### Bayes rule for Gaussians

The posterior is given by

$$
\begin{aligned}
    p(z\mid y)&=\N(z\mid\mu_{z\mid y},\Cov_{z\mid y}) \\
    \Cov_{z\mid y}^{-1}&=\Cov_{z}^{-1} + W^T\Cov_y^{-1}W \\ 
    \mu_{z\mid y}&= \Cov_{z\mid y}\Bigl[W^T\Cov_y^{-1}(y-b)+\Cov_z^{-1}\mu_z\Bigr]
\end{aligned}
$$

The normalization constant of the posterior is given by

$$
\begin{aligned}
    p(y)&=\int\N(z\mid\mu_z,\Cov_z)\N(y\mid Wz+b,\Cov_y)dz \\
    &= \N(y\mid W\mu_z+b,\Cov_y+W\Cov_zW^T)
\end{aligned}
$$

### Derivation

The log of the joint distribution is

$$
\begin{aligned}
    \log p(z,y) = &-\frac{1}{2}(z-\mu_z)^T\Cov_z^{-1}(z-\mu_z) \\
    &-\frac{1}{2}(y-Wz-b)^T\Cov_y^{-1}(y-Wz-b)
\end{aligned}
$$

This is a joint Gaussian distribution since it is the exponential of a quadratic form.

It can be rearranged as

$$
Q = -\frac{1}{2} 
\begin{pmatrix} 
z \\ y 
\end{pmatrix}^T
\begin{pmatrix} 
\Sigma_z^{-1} + W^T \Sigma_y^{-1} W & -W^T \Sigma_y^{-1} \\
- \Sigma_y^{-1} W & \Sigma_y^{-1} 
\end{pmatrix}
\begin{pmatrix} 
z \\ y 
\end{pmatrix}
$$

where the precision matrix is defined as

$$
\Sigma^{-1} =
\begin{pmatrix} 
\Sigma_z^{-1} + W^T \Sigma_y^{-1} W & -W^T \Sigma_y^{-1} \\
- \Sigma_y^{-1} W & \Sigma_y^{-1} 
\end{pmatrix}
\triangleq \Lambda =
\begin{pmatrix} 
\Lambda_{zz} & \Lambda_{zy} \\
\Lambda_{yz} & \Lambda_{yy} 
\end{pmatrix}
$$

Using standard results for the conditional distribution of Gaussians, we have

$$
p(z \mid y) = \N(\mu_{z|y}, \Sigma_{z|y})
$$

with

$$
\Sigma_{z|y} = \Lambda_{zz}^{-1} = \left( \Sigma_z^{-1} + W^T \Sigma_y^{-1} W \right)^{-1}
$$

and

$$
\begin{aligned}
    \mu_{z|y} &= \Sigma_{z|y}\Bigl[ \Lambda_{zz}\mu_z - \Lambda_{zy}(y-(W\mu_z+b)) \Bigr] \\
    &= \Sigma_{z|y}\Bigl[ \Sigma_z^{-1}\mu_z + W^T \Sigma_y^{-1}(y-b) \Bigr]
\end{aligned}
$$

When working with linear Gaussian systems, it is common to complete the square in the exponent. In the scalar case, a quadratic function

$$
f(x) = ax^2 + bx + c
$$

can be rewritten as

$$
a(x-h)^2 + k \quad \text{with} \quad h = \frac{-b}{2a}, \quad k = c - \frac{b^2}{4a}.
$$

In the vector case, one can similarly complete the square for

$$
x^T A x + x^T b + c.
$$

### Example: Inferring an unknown scalar

Suppose we make \(N\) noisy measurements \(y_i\) of a scalar \(z\) with fixed measurement noise precision \(\lambda_y=\frac{1}{\sigma^2}\):

$$
p(y_i\mid z) = \N(y_i\mid z,\lambda_y^{-1})
$$

and a prior

$$
p(z)=\N(z\mid\mu_0,\lambda_0^{-1})
$$

The posterior is given by

$$
p(z\mid y_1,\dots,y_N) = \N(z\mid\mu_N,\lambda_N^{-1})
$$

where the posterior precision is

$$
\lambda_N=\lambda_0+N\lambda_y
$$

and the posterior mean is

$$
\mu_N=\frac{N\lambda_y \overline{y} + \lambda_0\mu_0}{\lambda_N}
$$

which can also be written as

$$
\mu_N=\frac{\sigma^2\mu_0+\tau_0^2\overline{y}}{N\tau_0^2+\sigma^2}
$$

The posterior **variance** is

$$
\tau_N^2 = \frac{\sigma^2 \tau_0^2}{N\tau_0^2 + \sigma^2}
$$

This variance shrinks as we incorporate more measurements, reducing our uncertainty about \(z\).

We can also update sequentially. After observing \(y_1\),

$$
p(z \mid y_1) = \N(z \mid \mu_1, \sigma_1^2)
$$

with

$$
\mu_1 = \frac{\sigma_y^2 \mu_0 + \sigma_0^2 y_1}{\sigma_0^2 + \sigma_y^2}, \quad
\sigma_1^2 = \frac{\sigma_0^2 \sigma_y^2}{\sigma_0^2 + \sigma_y^2}.
$$

Then treat \(p(z \mid y_1)\) as the new prior and update with \(y_2\), and so on. After \(N\) observations, the updates are

$$
\mu_N = \frac{\sigma_y^2 \mu_{N-1} + \sigma_{N-1}^2 y_N}{\sigma_{N-1}^2 + \sigma_y^2}
$$

$$
\sigma_N^2 = \frac{\sigma_{N-1}^2 \sigma_y^2}{\sigma_{N-1}^2 + \sigma_y^2}
$$

- **Posterior Variance Decreases:** Each new observation reduces uncertainty.
- **Posterior Mean as Weighted Average:** It balances the prior mean and observed data.
- **Sequential Bayesian Update:** The previous posterior becomes the new prior.

**Signal-to-noise Ratio (SNR):**

To quantify the influence of the prior relative to the measurement noise, define

$$
\text{SNR} = \frac{\mathbb{E}[Z^2]}{\mathbb{E}[\epsilon^2]} = \frac{\Sigma_0 + \mu_0^2}{\Sigma_y}
$$

This ratio compares the prior variance (and mean) to the measurement noise, indicating how much the data refines our estimate.

## Mixture Models
