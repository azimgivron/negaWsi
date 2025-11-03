# Non-Euclidean Gradient Algorithm for matrix completion With Side Information (negaWsi)

This repository implements the **Non-Euclidean Gradient Algorithm (NEGA)** for matrix completion and its extensions for incorporating biological side information.
The method is applied to **gene–disease association matrix completion** for **gene prioritization** tasks.

## 1. Overview

In gene prioritization, we aim to infer missing associations between genes and diseases represented by a sparse matrix
$R \in \mathbb{R}^{n \times m}$, where $n$ is the number of genes and $m$ is the number of diseases.
The algorithm completes this matrix by learning two low-rank latent representations: 

$$
R \approx W H^\top, \quad
W \in \mathbb{R}^{n \times k}, \; H \in \mathbb{R}^{m \times k}
$$

where $k \ll \min(n, m)$ is the latent dimension.
The goal is to assign high scores to plausible but unobserved gene–disease pairs.

## 2. Objective Function

The core optimization problem is:

$$
\min_{W,H}
\frac{1}{2} \|B \odot (R - WH^\top)\|_F^2 +\lambda_g \|W\|_F^2 + \lambda_d \|H\|_F^2
$$

where:
- $B$ is a binary mask for observed entries,
- $\lambda_g, \lambda_d$ are regularization coefficients,
- $\odot$ denotes the element-wise (Hadamard) product.

This formulation captures reconstruction error and applies Frobenius regularization to prevent overfitting.

## 3. Non-Euclidean Gradient Algorithm (NEGA)

NEGA generalizes gradient descent by using **Bregman distances** instead of the Euclidean norm, allowing updates that respect the **geometry of the problem**.

Given a differentiable, strictly convex kernel function $h$, the **Bregman distance** between $x$ and $y$ is:

$$
\mathcal{D}_h(x, y) = h(x) - h(y) - \langle \nabla h(y), x - y \rangle
$$

The NEGA update rule is defined as:

$$
x^{k+1} = \arg\min_x \left\{ \langle \nabla f(x^k), x - x^k \rangle + \frac{1}{\alpha} \mathcal{D}_h(x, x^k) \right\}
$$

where $\alpha > 0$ is the step size.

When $h(x) = \tfrac{1}{2}\|x\|_2^2$, this reduces to standard gradient descent.
For matrix completion, the chosen kernel is:

$$
h(V) = \frac{1}{4}\|V\|_F^4 + \frac{\tau}{3}\|V\|_F^2
\quad V = \begin{pmatrix} W \ H \end{pmatrix}
$$

This kernel leads to **closed-form updates** and **adaptive step-size selection** through a backtracking line search, ensuring convergence under relative smoothness conditions.

## 4. Adaptive NEGA Algorithm

The adaptive variant employs a **nested loop** structure:

1. **Outer loop:** Updates the joint variable $v = (W, H)$ based on the current gradient.
2. **Inner loop:** Performs backtracking line search to ensure monotonic decrease of the objective.

Termination occurs when the Bregman distance between successive iterates falls below a threshold $\varepsilon$.

## 5. Extensions: Incorporating Side Information

To enhance gene prioritization, two extensions of NEGA are implemented.

### 5.1. SI via Feature Space based factorization

Side information is incorporated through feature matrices:
$$
R \approx X W H^\top Y^\top
$$
where:

* $X \in \mathbb{R}^{n \times g}$ encodes gene features,
* $Y \in \mathbb{R}^{m \times d}$ encodes disease features,
* $W \in \mathbb{R}^{g \times k}$, $H \in \mathbb{R}^{d \times k}$.

The corresponding objective is:

$$
\min_{W,H}
\frac{1}{2}\|B \odot (R - XWH^\top Y^\top)\|_F^2
+ \lambda_g\|W\|_F^2
+ \lambda_d\|H\|_F^2
$$

### 5.2. SI via Regularized Latent Alignment

Following the GeneHound model, side information is linked to the latent variables through regularization:

$$
\begin{aligned}
\min_{W,H,\beta_g,\beta_d} \;
& \frac{1}{2}\|B \odot (R - WH^\top)\|_F^2
+ \frac{\lambda_g}{2}\|\beta_g\|_F^2
+ \frac{\lambda_d}{2}\|\beta_d\|_F^2 \\
& + \frac{\lambda_{\beta_g}}{2}\|P_W W - X\beta_g\|_F^2
+ \frac{\lambda_{\beta_d}}{2}\|P_H H - Y\beta_d\|_F^2
\end{aligned}
$$

where $P_W, P_H$ are centering matrices enforcing mean-zero latent representations.

This formulation assumes Gaussian priors on parameters and residuals, while preserving the convergence guarantees of NEGA.

## 7. Repository Structure

```
negaWsi
├── __init__.py
├── base.py
├── early_stopping.py
├── flip_labels.py
├── nega.py
├── nega_fs.py
├── nega_reg.py
├── result.py
└── utils.py
```


## 8. References

* **IMC:** Natarajan, N. and Dhillon, I. (2014). *Inductive Matrix Completion for Predicting Gene–Disease Associations.*
* **NEGA:** - Ghaderi, S., Moreau, Y., & Ahookhosh, M. (2022). *Non-Euclidean Gradient Methods: Convergence, Complexity, and Applications*. JMLR, 23(2022):1-44.
* **GeneHound:** Yang, J. et al. (2022). *Gene Prioritization Using Bayesian Matrix Completion.*
