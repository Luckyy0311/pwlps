# PWLPS: Precision-Weighted Local Predictive Settling

[![PyPI version](https://badge.fury.io/py/pwlps.svg)](https://badge.fury.io/py/pwlps)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

**PWLPS (Precision-Weighted Local Predictive Settling)** is a forward-only local learning algorithm for deep neural networks.

Instead of propagating a single global gradient backward through the entire network, PWLPS trains each layer using:

1. **Feedforward weights** $W_l$ — transform the input representation.
2. **A local classifier head** $C_l$ — provides a local supervised learning signal.
3. **A precision parameter** $\pi_l$ — controls how strongly the layer trusts its local prediction error.

Each layer performs a short **settling process** in which its activity is adjusted toward a representation that both agrees with its feedforward prediction and improves local classification.

The resulting learning rule is local, forward-only, and does not require a global backward pass through the complete network.

---

## Table of Contents

- [Key Features](#-key-features)
- [How It Works](#-how-it-works)
- [The Three Components Per Layer](#the-three-components-per-layer)
- [The Two-Phase Cycle](#the-two-phase-cycle)
- [Mathematical Formulation](#-mathematical-formulation)
- [Full Algorithm](#-full-algorithm)
- [Solved Numerical Example](#-solved-numerical-example)
- [Installation](#-installation)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Experimental Results](#-experimental-results)
- [Repository Structure](#-repository-structure)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

# Key Features

| Feature | Description |
|---|---|
| **Forward-only training** | No global backward pass through the complete network |
| **Local learning** | Each layer learns using locally available signals |
| **Biologically inspired** | Uses local prediction errors and local supervised signals |
| **Memory efficient** | Layers can be processed sequentially without retaining the full computation graph |
| **Local classifier heads** | Every layer receives a direct classification signal |
| **Precision weighting** | Each layer can assign adaptive confidence to its prediction error |
| **Fixed or learned precision** | Precision can either remain fixed or be learned |
| **Architecture agnostic** | Can be applied to MLPs, CNNs, and other differentiable blocks |

---

# How It Works

## Standard Backpropagation: Global Credit Assignment

In standard deep learning, the network first performs a complete forward pass:

```text
Input
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Output
```

The loss is then propagated backward through the entire network:

```text
Output
  ↓
Layer 3
  ↓
Layer 2
  ↓
Layer 1
```

The gradient for an early layer therefore depends on computations performed by all subsequent layers.

For a deep network, this usually requires storing intermediate activations so that the backward pass can compute gradients.

---

## PWLPS: Local Self-Correction

PWLPS replaces global credit assignment with **layer-wise local learning**.

Each layer asks:

> Based on the representation I received, can I produce an internal representation that both agrees with my feedforward prediction and helps predict the correct class?

The learning process becomes:

```text
Input
  ↓
Layer 1 → local settling → local update
  ↓
Layer 2 → local settling → local update
  ↓
Layer 3 → local settling → local update
  ↓
Output
```

After a layer has been trained, its output is detached before being passed to the next layer.

Therefore, a later layer cannot send gradients back into earlier layers.

---

# The Three Components Per Layer

Every PWLPS layer contains three important components.

## 1. Feedforward weights

The feedforward weight matrix

$$
W_l \in \mathbb{R}^{d_l \times d_{l-1}}
$$

transforms the representation from the previous layer.

The feedforward prediction is

$$
z_l = W_l a_{l-1},
$$

followed by the activation function

$$
\hat{a}_l = \phi(z_l).
$$

---

## 2. Local classifier head

Each layer has its own classifier

$$
C_l \in \mathbb{R}^{K \times d_l}.
$$

The local classifier produces logits

$$
u_l = C_l h_l,
$$

and class probabilities

$$
p_l = \operatorname{softmax}(u_l).
$$

This provides a direct supervised learning signal to the layer.

---

## 3. Precision parameter

Each layer has a positive precision parameter

$$
\pi_l > 0.
$$

The precision controls how strongly the layer trusts the prediction error

$$
h_l - \hat{a}_l.
$$

A high value of $\pi_l$ means that the layer strongly penalizes deviations from its feedforward prediction.

---

# The Two-Phase Cycle

PWLPS training consists of two local phases.

## Phase 1: Inference / Settling

The layer activity $h_l$ is adjusted while the layer parameters are kept fixed.

The activity minimizes the local energy

$$
E_l(h_l)
=
\frac{\pi_l}{2}
\left\|
h_l-\hat{a}_l
\right\|^2
+
\lambda
\mathcal{L}_{CE}(C_lh_l,y).
$$

---

## Phase 2: Local Learning

After settling, the resulting activity

$$
h_l^*
$$

is treated as a target for the feedforward pathway.

The layer then updates:

- $W_l$
- $C_l$
- $\pi_l$ or its underlying parameter

using only locally available quantities.

---

# Mathematical Formulation

## 1. Problem Setup

Let the training dataset be

$$
\mathcal{D}
=
\left\{
(x_i,y_i)
\right\}_{i=1}^{N},
$$

where

$$
x_i \in \mathbb{R}^{d_0}
$$

is an input sample and

$$
y_i \in \{1,2,\ldots,K\}
$$

is its class label.

For MNIST,

$$
d_0 = 784,
\qquad
K = 10.
$$

The goal is to learn a function

$$
f_\theta(x)=\hat{y},
$$

where $\theta$ contains the trainable parameters.

---

# 2. Cross-Entropy Loss

For logits

$$
u \in \mathbb{R}^{K},
$$

the softmax probability of class $k$ is

$$
p_k
=
\frac{\exp(u_k)}
{\sum_{j=1}^{K}\exp(u_j)}.
$$

For the true class $y$, the cross-entropy loss is

$$
\mathcal{L}_{CE}(u,y)
=
-\log(p_y).
$$

Let $e_y$ be the one-hot vector representing the target class.

The softmax error is

$$
r
=
p-e_y.
$$

---

# 3. Standard Backpropagation Baseline

A conventional feedforward network computes

$$
z_l = W_l a_{l-1},
$$

and

$$
a_l = \phi(z_l),
$$

where:

- $a_0=x$
- $z_l$ is the pre-activation
- $a_l$ is the post-activation
- $W_l$ is the weight matrix
- $\phi$ is the activation function

For example, with ReLU,

$$
\phi(z)=\max(0,z).
$$

Backpropagation computes gradients through the global chain rule:

$$
\frac{\partial \mathcal{L}}
{\partial W_l}
=
\frac{\partial \mathcal{L}}
{\partial a_L}
\frac{\partial a_L}
{\partial a_{L-1}}
\cdots
\frac{\partial a_{l+1}}
{\partial a_l}
\frac{\partial a_l}
{\partial W_l}.
$$

The gradient for layer $l$ therefore depends on the computations of subsequent layers.

---

# 4. PWLPS Layer Components

For layer $l$, define:

| Symbol | Shape | Role |
|---|---|---|
| $W_l$ | $\mathbb{R}^{d_l \times d_{l-1}}$ | Feedforward weights |
| $C_l$ | $\mathbb{R}^{K \times d_l}$ | Local classifier |
| $\pi_l$ | scalar, $\pi_l>0$ | Prediction-error precision |

The feedforward pathway computes

$$
z_l = W_l a_{l-1},
$$

followed by

$$
\hat{a}_l=\phi(z_l).
$$

During settling, the layer maintains an activity variable $h_l$.

The local classifier computes

$$
u_l=C_lh_l,
$$

and

$$
p_l=\operatorname{softmax}(u_l).
$$

The corresponding local classification loss is

$$
\ell_l(h_l)
=
\mathcal{L}_{CE}(C_lh_l,y).
$$

---

# 5. Local Energy Function

The core PWLPS objective is a local energy combining prediction error and classification error:

$$
E_l(h_l)
=
\frac{\pi_l}{2}
\left\|
h_l-\hat{a}_l
\right\|^2
+
\lambda
\ell_l(h_l).
$$

Substituting the feedforward prediction

$$
\hat{a}_l
=
\phi(W_la_{l-1}),
$$

gives

$$
E_l(h_l)
=
\frac{\pi_l}{2}
\left\|
h_l-\phi(W_la_{l-1})
\right\|^2
+
\lambda
\mathcal{L}_{CE}(C_lh_l,y).
$$

Here:

- $\pi_l$ controls the strength of the prediction-error term.
- $\lambda$ controls the strength of the classification term.
- $h_l$ is the current layer activity.
- $\hat{a}_l$ is the feedforward prediction.

---

# 6. Activity Settling Dynamics

During settling, the weights are frozen and only the activity $h_l$ is updated.

The continuous-time formulation is

$$
\tau
\frac{dh_l}{dt}
=
-\nabla_{h_l}E_l(h_l).
$$

A discrete gradient-descent implementation is

$$
h_l^{(t+1)}
=
h_l^{(t)}
-
\gamma
\nabla_{h_l}
E_l(h_l^{(t)}),
$$

where:

- $\gamma$ is the settling step size.
- $t$ is the settling iteration.
- $T$ is the total number of settling iterations.

---

# 7. Gradient of the Local Energy

The prediction-error term is

$$
E_l^{pred}
=
\frac{\pi_l}{2}
\left\|
h_l-\hat{a}_l
\right\|^2.
$$

Its gradient with respect to $h_l$ is

$$
\nabla_{h_l}E_l^{pred}
=
\pi_l
(h_l-\hat{a}_l).
$$

For the classification term, define

$$
p_l
=
\operatorname{softmax}(C_lh_l),
$$

and

$$
r_l
=
p_l-e_y.
$$

Then

$$
\nabla_{h_l}
\mathcal{L}_{CE}(C_lh_l,y)
=
C_l^Tr_l.
$$

Therefore, the complete local energy gradient is

$$
\nabla_{h_l}E_l(h_l)
=
\pi_l(h_l-\hat{a}_l)
+
\lambda C_l^T(p_l-e_y).
$$

The complete settling update is therefore

$$
h_l^{(t+1)}
=
h_l^{(t)}
-
\gamma
\left[
\pi_l
\left(
h_l^{(t)}-\hat{a}_l
\right)
+
\lambda
C_l^T
\left(
p_l^{(t)}-e_y
\right)
\right].
$$

After the update, a projection operator can be applied:

$$
h_l^{(t+1)}
=
\mathcal{P}
\left(
h_l^{(t+1)}
\right),
$$

where $\mathcal{P}$ may implement ReLU and optional clipping.

---

# 8. Settled Activity

After $T$ settling iterations, the final activity is

$$
h_l^*
=
\operatorname{sg}
\left[
h_l^{(T)}
\right],
$$

where $\operatorname{sg}[\cdot]$ denotes the stop-gradient operation.

The settled activity is therefore treated as a local target rather than as part of a global computation graph.

---

# 9. Local Weight Learning Objective

After settling, PWLPS trains the current layer using a local objective:

$$
\mathcal{J}_l
=
\mathcal{J}_l^{pred}
+
\lambda
\mathcal{J}_l^{cls}
+
\lambda
\mathcal{J}_l^{settled}
+
\mathcal{R}_l^\pi.
$$

The individual terms are described below.

## 9.1 Predictive Matching Loss

The feedforward activity is

$$
a_l
=
\phi(W_la_{l-1}).
$$

The predictive matching loss is

$$
\mathcal{J}_l^{pred}
=
\frac{\tilde{\pi}_l}{2}
\left\|
h_l^*-a_l
\right\|^2,
$$

where

$$
\tilde{\pi}_l
=
\operatorname{sg}[\pi_l].
$$

This term teaches the feedforward pathway to reproduce the activity found during local settling.

## 9.2 Classification Loss on Feedforward Activity

The current feedforward activity is classified directly:

$$
\mathcal{J}_l^{cls}
=
\mathcal{L}_{CE}
(C_la_l,y).
$$

This ensures that the feedforward representation itself remains useful for classification.

## 9.3 Classification Loss on Settled Activity

The settled activity is also classified:

$$
\mathcal{J}_l^{settled}
=
\mathcal{L}_{CE}
(C_lh_l^*,y).
$$

This provides an additional supervised signal to the local classifier.

## 9.4 Precision Regularization

If precision is learned, parameterize it using

$$
\pi_l
=
\operatorname{softplus}(s_l)
+
\epsilon,
$$

which guarantees

$$
\pi_l>0.
$$

A regularization term can be written as

$$
\mathcal{R}_l^\pi
=
\frac{1}{2}
\pi_l
\operatorname{sg}
\left[
\left\|
h_l^*-a_l
\right\|^2
\right]
+
\rho
(\pi_l-\pi_0)^2.
$$

Here:

- $\rho$ controls the regularization strength.
- $\pi_0$ is the target precision.
- $\epsilon$ prevents the precision from becoming zero.

---

# 10. Gradient of the Predictive Matching Loss

Define the local prediction error

$$
\delta_l^{pred}
=
h_l^*-a_l.
$$

Since

$$
a_l=\phi(z_l),
\qquad
z_l=W_la_{l-1},
$$

the gradient of the predictive matching loss with respect to $W_l$ is

$$
\frac{\partial\mathcal{J}_l^{pred}}
{\partial W_l}
=
-\tilde{\pi}_l
\left(
\delta_l^{pred}
\odot
\phi'(z_l)
\right)
a_{l-1}^T.
$$

Therefore, the corresponding local update is

$$
W_l
\leftarrow
W_l
+
\eta\tilde{\pi}_l
\left(
\delta_l^{pred}
\odot
\phi'(z_l)
\right)
a_{l-1}^T.
$$

This is a local predictive-error learning rule.

---

# 11. Gradient of the Local Classification Loss

For the feedforward activity,

$$
p_l
=
\operatorname{softmax}(C_la_l),
$$

and

$$
r_l=p_l-e_y.
$$

The gradient with respect to the feedforward weights is

$$
\frac{\partial\mathcal{J}_l^{cls}}
{\partial W_l}
=
\left[
(C_l^Tr_l)
\odot
\phi'(z_l)
\right]
a_{l-1}^T.
$$

Therefore, its contribution to the weight update is

$$
W_l
\leftarrow
W_l
-
\eta\lambda
\left[
(C_l^Tr_l)
\odot
\phi'(z_l)
\right]
a_{l-1}^T.
$$

The complete weight gradient is obtained by combining the predictive and classification contributions:

$$
\nabla_{W_l}\mathcal{J}_l
=
\nabla_{W_l}\mathcal{J}_l^{pred}
+
\lambda
\nabla_{W_l}\mathcal{J}_l^{cls}.
$$

---

# 12. Gradient for the Local Classifier

For the current feedforward activity,

$$
\frac{\partial\mathcal{J}_l^{cls}}
{\partial C_l}
=
r_la_l^T.
$$

For the settled activity, define

$$
p_l^*
=
\operatorname{softmax}(C_lh_l^*),
$$

and

$$
r_l^*
=
p_l^*-e_y.
$$

Then

$$
\frac{\partial\mathcal{J}_l^{settled}}
{\partial C_l}
=
r_l^*
(h_l^*)^T.
$$

Therefore,

$$
\nabla_{C_l}\mathcal{J}_l
=
\lambda r_la_l^T
+
\lambda r_l^*
(h_l^*)^T.
$$

The local classifier is updated using

$$
C_l
\leftarrow
C_l
-
\eta
\nabla_{C_l}\mathcal{J}_l.
$$

---

# 13. Precision Update

For learned precision,

$$
\pi_l
=
\operatorname{softplus}(s_l)+\epsilon.
$$

Define the detached prediction error magnitude

$$
E_l^{detached}
=
\operatorname{sg}
\left[
\left\|
h_l^*-a_l
\right\|^2
\right].
$$

The derivative of the precision regularizer with respect to $\pi_l$ is

$$
\frac{\partial\mathcal{R}_l^\pi}
{\partial\pi_l}
=
\frac{1}{2}
E_l^{detached}
+
2\rho(\pi_l-\pi_0).
$$

Because

$$
\frac{\partial\pi_l}
{\partial s_l}
=
\sigma(s_l),
$$

the update becomes

$$
s_l
\leftarrow
s_l
-
\eta
\left[
\frac{1}{2}
E_l^{detached}
+
2\rho(\pi_l-\pi_0)
\right]
\sigma(s_l).
$$

---

# 14. Forward-Only Property

The defining property of PWLPS is that the output of a processed layer is detached before being passed to the next layer.

Specifically,

$$
a_l
=
\operatorname{sg}
\left[
\phi(W_la_{l-1})
\right].
$$

Consequently, the objective of layer $l+1$ cannot propagate gradients through layer $l$:

$$
\frac{\partial\mathcal{J}_{l+1}}
{\partial W_l}
=
0.
$$

Thus, there is no global gradient flow from deeper layers to earlier layers.

Each layer learns from locally available information.

---

# 15. Memory and Time Complexity

A simplified comparison is:

| Property | Backpropagation | PWLPS |
|---|---|---|
| Training direction | Forward + backward | Forward + local settling |
| Global backward pass | Required | Not required |
| Activation storage | Across the computation graph | Sequential/local |
| Approximate activation memory | $O(Ld)$ | $O(d)$ |
| Layer processing | Once per forward pass | $T$ local settling steps |
| Approximate layer computation | $O(L)$ | $O(LT)$ |

Here:

- $L$ = number of layers.
- $d$ = representative hidden width.
- $T$ = number of settling iterations.

The main trade-off is therefore:

$$
\boxed{
\text{Lower activation memory}
\quad\Longleftrightarrow\quad
\text{Higher computation from settling}
}
$$

---

# Full Algorithm

## Algorithm 1: Precision-Weighted Local Predictive Settling

**Input:**

- Minibatch $(x,y)$
- Number of layers $L$
- Weight learning rate $\eta$
- Settling learning rate $\gamma$
- Settling steps $T$
- Classification weight $\lambda$
- Precision regularization $\rho$
- Target precision $\pi_0$

**Output:**

Updated parameters

$$
\{W_l,C_l,s_l\}_{l=1}^{L}.
$$

### Step 1: Initialize the input

$$
a_0\leftarrow x.
$$

### Step 2: Process each layer

For

$$
l=1,\ldots,L,
$$

perform the following operations.

### Feedforward prediction

$$
z_l
\leftarrow
W_la_{l-1}
$$

and

$$
\hat{a}_l
\leftarrow
\phi(z_l).
$$

Initialize the settling activity:

$$
h_l^{(0)}
\leftarrow
\hat{a}_l.
$$

### Local settling

For

$$
t=0,\ldots,T-1,
$$

compute

$$
p_l^{(t)}
=
\operatorname{softmax}
\left(
C_lh_l^{(t)}
\right),
$$

then

$$
r_l^{(t)}
=
p_l^{(t)}-e_y.
$$

Compute the local activity gradient:

$$
g_h
=
\pi_l
\left(
h_l^{(t)}-\hat{a}_l
\right)
+
\lambda
C_l^Tr_l^{(t)}.
$$

Update the activity:

$$
h_l^{(t+1)}
=
h_l^{(t)}
-
\gamma g_h.
$$

Apply the projection:

$$
h_l^{(t+1)}
\leftarrow
\mathcal{P}
\left(
h_l^{(t+1)}
\right).
$$

### Step 3: Detach settled activity

After $T$ iterations,

$$
h_l^*
\leftarrow
\operatorname{sg}
\left[
h_l^{(T)}
\right].
$$

### Step 4: Compute the local learning objective

Compute

$$
a_l
=
\phi(W_la_{l-1}).
$$

Then compute

$$
\mathcal{J}_l
=
\mathcal{J}_l^{pred}
+
\lambda\mathcal{J}_l^{cls}
+
\lambda\mathcal{J}_l^{settled}
+
\mathcal{R}_l^\pi.
$$

### Step 5: Update local parameters

Update the feedforward weights:

$$
W_l
\leftarrow
W_l
-
\eta
\nabla_{W_l}\mathcal{J}_l.
$$

Update the local classifier:

$$
C_l
\leftarrow
C_l
-
\eta
\nabla_{C_l}\mathcal{J}_l.
$$

If precision is learned:

$$
s_l
\leftarrow
s_l
-
\eta
\nabla_{s_l}\mathcal{R}_l^\pi.
$$

### Step 6: Detach before passing to the next layer

$$
a_l
\leftarrow
\operatorname{sg}[a_l].
$$

The algorithm then proceeds to layer $l+1$.

---

# Solved Numerical Example

Consider a one-layer toy network.

## Setup

| Symbol | Value |
|---|---|
| Input $x$ | $[1,0]^T$ |
| True class $y$ | $1$ |
| One-hot target $e_y$ | $[0,1]^T$ |
| Weight $W$ | $[0.20,-0.10]$ |
| Classifier $C$ | $\begin{bmatrix}0\\1\end{bmatrix}$ |
| Activation $\phi$ | ReLU |
| Precision $\pi$ | $1$ |
| Classification weight $\lambda$ | $1$ |
| Settling rate $\gamma$ | $0.1$ |
| Weight learning rate $\eta$ | $0.1$ |
| Settling steps $T$ | $1$ |

## Step 1: Feedforward Prediction

The pre-activation is

$$
z
=
Wx.
$$

Therefore,

$$
z
=
0.20(1)+(-0.10)(0)
=
0.20.
$$

Using ReLU,

$$
\hat{a}
=
\operatorname{ReLU}(0.20)
=
0.20.
$$

Initialize

$$
h^{(0)}
=
\hat{a}
=
0.20.
$$

## Step 2: Local Classification

The classifier produces

$$
u
=
Ch^{(0)}.
$$

Therefore,

$$
u
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
(0.20)
=
\begin{bmatrix}
0\\
0.20
\end{bmatrix}.
$$

The softmax probabilities are approximately

$$
p_0
=
\frac{e^0}{e^0+e^{0.20}}
\approx
0.4502,
$$

and

$$
p_1
=
\frac{e^{0.20}}{e^0+e^{0.20}}
\approx
0.5498.
$$

The prediction error is

$$
r
=
p-e_y.
$$

Therefore,

$$
r
=
\begin{bmatrix}
0.4502\\
-0.4502
\end{bmatrix}.
$$

## Step 3: Activity Gradient

The prediction-error contribution is

$$
\pi(h^{(0)}-\hat{a})
=
1(0.20-0.20)
=
0.
$$

The classification contribution is

$$
C^Tr
=
\begin{bmatrix}
0 & 1
\end{bmatrix}
\begin{bmatrix}
0.4502\\
-0.4502
\end{bmatrix}
=
-0.4502.
$$

Therefore,

$$
\nabla_hE
=
0-0.4502
=
-0.4502.
$$

## Step 4: Settling Update

The activity update is

$$
h^{(1)}
=
h^{(0)}
-
\gamma\nabla_hE.
$$

Therefore,

$$
h^{(1)}
=
0.20
-
0.1(-0.4502)
=
0.2450.
$$

Since the value is positive, ReLU leaves it unchanged:

$$
h^*
=
0.2450.
$$

## Step 5: Predictive Weight Update

The predictive error is

$$
\delta^{pred}
=
h^*-a
=
0.2450-0.20
=
0.0450.
$$

Since

$$
z=0.20>0,
$$

the derivative of ReLU is

$$
\phi'(z)=1.
$$

Therefore,

$$
\nabla_W\mathcal{J}^{pred}
=
-\pi
\delta^{pred}
\phi'(z)
x^T.
$$

Substituting the values,

$$
\nabla_W\mathcal{J}^{pred}
=
-1(0.0450)(1)
[1,0].
$$

Thus,

$$
\nabla_W\mathcal{J}^{pred}
=
[-0.0450,0].
$$

## Step 6: Classification Weight Contribution

The classification contribution is

$$
\nabla_W\mathcal{J}^{cls}
=
(C^Tr)
\phi'(z)
x^T.
$$

Therefore,

$$
\nabla_W\mathcal{J}^{cls}
=
(-0.4502)(1)[1,0],
$$

giving

$$
\nabla_W\mathcal{J}^{cls}
=
[-0.4502,0].
$$

Because $\lambda=1$, the total gradient is

$$
\nabla_W\mathcal{J}
=
[-0.0450,0]
+
[-0.4502,0].
$$

Hence,

$$
\nabla_W\mathcal{J}
=
[-0.4952,0].
$$

The updated weight is

$$
W_{new}
=
W-\eta\nabla_W\mathcal{J}.
$$

Therefore,

$$
W_{new}
=
[0.20,-0.10]
-
0.1[-0.4952,0],
$$

giving

$$
\boxed{
W_{new}
=
[0.2495,-0.10]
}.
$$

## Step 7: Classifier Update

For the current activity,

$$
\nabla_C\mathcal{J}^{cls}
=
ra^T.
$$

Therefore,

$$
\nabla_C\mathcal{J}^{cls}
=
\begin{bmatrix}
0.4502\\
-0.4502
\end{bmatrix}
(0.20),
$$

which gives

$$
\nabla_C\mathcal{J}^{cls}
=
\begin{bmatrix}
0.0900\\
-0.0900
\end{bmatrix}.
$$

For the settled activity,

$$
h^*=0.2450.
$$

The settled probabilities are approximately

$$
p_0^*
\approx
0.4391,
$$

and

$$
p_1^*
\approx
0.5609.
$$

Therefore,

$$
r^*
=
\begin{bmatrix}
0.4391\\
-0.4391
\end{bmatrix}.
$$

The settled classification gradient is

$$
\nabla_C\mathcal{J}^{settled}
=
r^*(h^*)^T,
$$

giving approximately

$$
\nabla_C\mathcal{J}^{settled}
=
\begin{bmatrix}
0.1076\\
-0.1076
\end{bmatrix}.
$$

The total classifier gradient is therefore

$$
\nabla_C\mathcal{J}
=
\begin{bmatrix}
0.1976\\
-0.1976
\end{bmatrix}.
$$

The updated classifier becomes

$$
C_{new}
=
C
-
\eta\nabla_C\mathcal{J}.
$$

Thus,

$$
C_{new}
=
\begin{bmatrix}
0\\
1
\end{bmatrix}
-
0.1
\begin{bmatrix}
0.1976\\
-0.1976
\end{bmatrix},
$$

giving

$$
\boxed{
C_{new}
=
\begin{bmatrix}
-0.0198\\
1.0198
\end{bmatrix}
}.
$$

## Numerical Result Summary

| Stage | Probability of correct class |
|---|---:|
| Before settling | $0.5498$ |
| After settling | $0.5609$ |
| After local parameter update | Expected to increase on the next forward pass |

The example demonstrates the basic PWLPS mechanism:

$$
\boxed{
\text{Feedforward}
\rightarrow
\text{Local Settling}
\rightarrow
\text{Local Error}
\rightarrow
\text{Local Update}
}
$$

No global backward pass is required.

---

# Installation

## From PyPI

```bash
pip install pwlps
```

## From Source

```bash
git clone https://github.com/yourusername/pwlps.git
cd pwlps
pip install -e .
```

## Requirements

```text
Python >= 3.8
PyTorch >= 2.0.0
```

---

# Usage

## Basic MLP Example

```python
import torch
from pwlps import LocalPredictiveMLP

model = LocalPredictiveMLP(
    input_dim=784,
    hidden_dim=512,
    num_classes=10,
    num_layers=4,
    learn_precision=True,
)

optimizers = [
    torch.optim.Adam(
        layer.parameters(),
        lr=1e-3
    )
    for layer in model.layers
]

x = torch.randn(128, 784)
y = torch.randint(0, 10, (128,))

loss = model.local_train_step(
    x=x,
    y=y,
    optimizers=optimizers,
    T=5,
    step_size=0.05,
    cls_weight=1.0,
)

with torch.no_grad():
    logits = model(x)
    preds = logits.argmax(dim=1)
```

---

# Architecture-Agnostic Example

PWLPS can be applied to differentiable neural network blocks rather than being restricted to simple MLP layers.

For example, a CNN can be constructed from independent blocks:

```python
import torch
import torch.nn as nn

from pwlps import build_local_sequential

blocks = [
    nn.Sequential(
        nn.Conv2d(1, 32, 3),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),

    nn.Sequential(
        nn.Conv2d(32, 64, 3),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),

    nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 5 * 5, 256),
        nn.ReLU()
    ),
]

model = build_local_sequential(
    blocks=blocks,
    num_classes=10,
    sample_input=torch.randn(2, 1, 28, 28),
    pool_types=[
        "mean_spatial",
        "mean_spatial",
        "identity"
    ],
    learn_precision=True,
)

optimizers = [
    torch.optim.Adam(
        block.parameters(),
        lr=1e-3
    )
    for block in model.blocks
]

x = torch.randn(128, 1, 28, 28)
y = torch.randint(0, 10, (128,))

loss = model.local_train_step(
    x=x,
    y=y,
    optimizers=optimizers,
    T=3
)
```

---

# Hyperparameter Reference

| Parameter | Symbol | Default | Description |
|---|---|---:|---|
| Settling steps | $T$ | `5` | Number of local inference iterations per layer |
| Settling step size | $\gamma$ | `0.05` | Learning rate used during activity settling |
| Classification weight | $\lambda$ | `1.0` | Weight of the local classification objective |
| Fixed precision | $\pi_0$ | `1.0` | Precision value or regularization target |
| Precision regularization | $\rho$ | `0.01` | Stabilizes learned precision |
| Weight learning rate | $\eta$ | `0.001` | Learning rate for local parameter updates |

---

# Methodology

## Research Question

> Can a deep neural network be trained using local, forward-only predictive settling with precision weighting while maintaining competitive accuracy and reducing activation memory compared with standard backpropagation?

## Hypotheses

| Hypothesis | Description |
|---|---|
| **H1** | A network trained using local predictive settling can learn useful representations from scratch. |
| **H2** | Predictive settling improves accuracy compared with layer-wise local classification alone. |
| **H3** | Learned precision improves stability or accuracy compared with fixed precision. |
| **H4** | Local learning has slower memory growth with depth than standard backpropagation. |

---

# Experimental Setup

| Setting | Value |
|---|---|
| Dataset | MNIST |
| Hardware | Google Colab, NVIDIA T4 GPU |
| Batch size | `128` |
| Optimizer | Adam |
| Learning rate | `0.001` |
| Hidden width | `512` |
| Activation | ReLU |
| Random seed | `0` |
| Settling steps | `5` |
| Settling step size | `0.05` |
| Classification weight | `1.0` |

---

# Experimental Conditions

## 1. Backpropagation Baseline

Standard global gradient descent through the entire network.

## 2. Local Learning with Fixed Precision

PWLPS with

$$
\pi_l=1.0.
$$

## 3. Local Learning with Learned Precision

PWLPS where

$$
\pi_l
=
\operatorname{softplus}(s_l)+\epsilon
$$

is learned during training.

---

# Reproduction Commands

## Four-layer backpropagation baseline

```bash
python main.py \
    --method bp \
    --epochs 5 \
    --num-layers 4 \
    --hidden-dim 512
```

## Four-layer local method with fixed precision

```bash
python main.py \
    --method local \
    --epochs 5 \
    --num-layers 4 \
    --hidden-dim 512 \
    --T 5
```

## Four-layer local method with learned precision

```bash
python main.py \
    --method local \
    --learn-precision \
    --epochs 5 \
    --num-layers 4 \
    --hidden-dim 512 \
    --T 5
```

## Sixteen-layer backpropagation baseline

```bash
python main.py \
    --method bp \
    --epochs 3 \
    --num-layers 16 \
    --hidden-dim 512
```

## Sixteen-layer local method

```bash
python main.py \
    --method local \
    --epochs 3 \
    --num-layers 16 \
    --hidden-dim 512 \
    --T 5
```

---

# Experimental Results

## Four-Layer Network

Training for five epochs produced the following results.

| Method | Parameters | Final Test Accuracy | Best Test Accuracy | Peak Memory | Time/Epoch |
|---|---:|---:|---:|---:|---:|
| Backpropagation | 1,195,018 | 97.54% | 97.66% | 41.12 MB | 14.87 s |
| Local Fixed Precision | 1,210,408 | 97.50% | 97.55% | 38.35 MB | 30.30 s |
| Local Learned Precision | 1,210,412 | 97.52% | 97.83% | 38.36 MB | 29.25 s |

### Observations

The fixed-precision local method achieves accuracy close to the backpropagation baseline.

The measured memory reduction is approximately

$$
\frac{41.12-38.35}{41.12}
\times100
\approx
6.74\%.
$$

The local method requires additional computation because every layer performs multiple settling iterations.

For the reported experiment, the approximate time overhead is

$$
\frac{30.30}{14.87}
\approx
2.04\times.
$$

---

# Deep Network Experiment

A sixteen-layer network was also evaluated.

| Method | Parameters | Test Accuracy | Peak Memory | Time/Epoch |
|---|---:|---:|---:|---:|
| Backpropagation | 4,346,890 | 82.34% | 101.24 MB | 16.83 s |
| Local Fixed Precision | 4,423,840 | 96.89% | 87.41 MB | 70.77 s |

The measured memory reduction is

$$
\frac{101.24-87.41}{101.24}
\times100
\approx
13.66\%.
$$

The reported accuracy difference is

$$
96.89-82.34
=
14.55
\text{ percentage points}.
$$

These results suggest that the local classifier heads can provide direct supervised signals to intermediate layers, which may help deep networks learn useful representations without relying on a long chain of backpropagated gradients.

---

# Key Findings

The experiments indicate three important observations:

### 1. Local learning can approach backpropagation accuracy

On the four-layer MNIST experiment, the local methods achieved approximately the same test accuracy as the backpropagation baseline.

### 2. Memory savings increase with depth

The measured memory reduction increased from approximately

$$
6.74\%
$$

for the four-layer experiment to

$$
13.66\%
$$

for the sixteen-layer experiment.

### 3. Settling introduces a computation trade-off

PWLPS requires multiple local inference iterations:

$$
T>1.
$$

Therefore, the method trades additional computation for reduced activation storage.

---

# Repository Structure

```text
pwlps/
├── pyproject.toml
├── README.md
├── LICENSE
│
├── pwlps/
│   ├── __init__.py
│   ├── core.py
│   └── universal.py
│
├── main.py
│
├── examples/
│   ├── mnist_mlp.py
│   └── mnist_cnn.py
│
└── notebooks/
    └── ResearchPro.ipynb
```

---

# Citation

If you use PWLPS in your research, please cite:

```bibtex
@misc{pwlps2026,
  title        = {PWLPS: Precision-Weighted Local Predictive Settling},
  author       = {Your Name},
  year         = {2026},
  howpublished = {\url{https://github.com/yourusername/pwlps}},
  note         = {A forward-only local learning algorithm for deep neural networks}
}
```

---

# License

This project is licensed under the MIT License.

See the `LICENSE` file for the complete license text.

---

# Acknowledgments

PWLPS draws inspiration from several areas of research:

- Predictive coding theories of cortical computation
- Local learning rules
- Biologically plausible credit assignment
- Feedback Alignment
- Target Propagation
- Equilibrium Propagation
- Forward-Forward learning

---

# Summary

PWLPS replaces global backpropagation with a layer-wise learning procedure:

$$
\boxed{
\text{Forward Prediction}
\rightarrow
\text{Local Settling}
\rightarrow
\text{Prediction Error}
\rightarrow
\text{Local Parameter Update}
}
$$

For every layer $l$, the three core components are:

$$
\boxed{
W_l
\qquad
C_l
\qquad
\pi_l
}
$$

where:

- $W_l$ transforms the incoming representation.
- $C_l$ provides a local supervised signal.
- $\pi_l$ controls the confidence assigned to the prediction error.

The central idea is to allow every layer to **predict, settle, classify, and learn locally**, eliminating the need for a global backward pass through the complete network.
