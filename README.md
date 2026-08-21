# PWLPS: Precision-Weighted Local Predictive Settling

<p align="center">
  <a href="https://badge.fury.io/py/pwlps"><img src="https://badge.fury.io/py/pwlps.svg" alt="PyPI version"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-red.svg" alt="PyTorch"></a>
</p>

**PWLPS** is a forward-only, local learning algorithm for deep neural networks. It replaces global backpropagation with **layer-wise predictive settling**, **local classification heads**, and **precision-weighted prediction errors**.

Unlike standard backpropagation, PWLPS does **not** propagate gradients through the entire network. Instead, each layer learns using only **local information**, resulting in **lower activation memory** and a **more biologically plausible** learning rule.

---

## Table of Contents

- [How It Works (Intuition)](#how-it-works-intuition)
- [Mathematical Formulation](#mathematical-formulation)
- [Full Algorithm](#full-algorithm)
- [Solved Numerical Example](#solved-numerical-example)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Experimental Results](#experimental-results)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## ✨ Key Features

- 🔄 **Forward-only training** — no global backward pass through the full network
- 🧠 **Biologically plausible** — each layer uses only local signals
- 💾 **Memory-efficient** — does not store the full computation graph across layers
- 🎯 **Local classifier heads** — every layer gets its own supervised signal
- ⚖️ **Precision weighting** — adaptive confidence per layer (fixed or learned)
- 🏗️ **Architecture-agnostic** — works with MLPs, CNNs, Transformers, RNNs, and more

---

## 🧠 How It Works (Intuition)

### Standard Backpropagation: "Global Blame"

In normal deep learning, the output makes a prediction. If it is wrong, a global error signal travels **backward** through every layer:

```text
Input → Layer 1 → Layer 2 → Layer 3 → Output
Output → Layer 3 → Layer 2 → Layer 1   (backward pass)
```

This requires storing activations from **all layers** so gradients can be computed later. Memory grows with depth.

### PWLPS: "Local Self-Correction"

In PWLPS, each layer does **not wait** for a global error. Instead, each layer asks:

> *"Based on the input I received, can I produce a representation that helps predict the correct label right now?"*

Each layer runs a small **local optimization** (called **settling**) to find a good internal representation, then updates its own weights to make that representation automatic.

```text
Input → Layer 1 settles & learns locally
      → Layer 2 settles & learns locally
      → Layer 3 settles & learns locally
      → Output
```

### The Three Components Per Layer

Every layer has:

1. **Feedforward weights** $W_l$ — transform the input.
2. **A local classifier head** $C_l$ — provides a local supervised signal.
3. **A precision parameter** $\pi_l$ — controls how much to trust the prediction error.

### The Two-Phase Cycle

For each layer, PWLPS alternates between:

- **Phase 1 — Inference (Settling):** Adjust the layer's *activity* $h_l$ to minimize a local energy (prediction error + classification loss). Weights are frozen.
- **Phase 2 — Learning:** Update the *weights* $W_l$ and *classifier* $C_l$ so the feedforward path produces the settled activity.

---

## 🧮 Mathematical Formulation

### 1. Problem Setup

Let the training dataset be:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}
$$

where:

- $x_i \in \mathbb{R}^{d_0}$ is the input sample
- $y_i \in \{1, 2, \dots, K\}$ is the class label

For MNIST: $d_0 = 784$ and $K = 10$.

We learn a function $f_\theta(x) = \hat{y}$, where $\theta$ denotes all trainable parameters.

---

### 2. Cross-Entropy Loss

Given logits $u \in \mathbb{R}^K$, the softmax probability for class $k$ is:

$$
p_k = \frac{e^{u_k}}{\sum_{j=1}^{K} e^{u_j}}
$$

The cross-entropy loss for true class $y$ is:

$$
\mathcal{L}_{CE}(u, y) = -\log(p_y)
$$

Let $e_y \in \mathbb{R}^K$ be the one-hot vector for class $y$. Define:

$$
p = \text{softmax}(u), \qquad r = p - e_y
$$

The term $r$ is the **softmax error** and appears throughout the update rules.

---

### 3. Standard Backpropagation Baseline

A feedforward network with $L$ layers computes:

$$
z_l = W_l a_{l-1}, \qquad a_l = \phi(z_l)
$$

where:

- $a_0 = x$
- $z_l$ is the pre-activation of layer $l$
- $a_l$ is the post-activation of layer $l$
- $W_l$ is the weight matrix
- $\phi$ is the activation function (e.g., ReLU)

Backpropagation computes the gradient via the global chain rule:

$$
\frac{\partial \mathcal{L}}{\partial W_l} = \frac{\partial \mathcal{L}}{\partial a_L} \frac{\partial a_L}{\partial a_{L-1}} \cdots \frac{\partial a_{l+1}}{\partial a_l} \frac{\partial a_l}{\partial W_l}
$$

This requires storing activations from **all** layers, giving memory cost:

$$
M_{BP} = O(Ld)
$$

where $L$ is depth and $d$ is hidden width.

---

### 4. Layer Components in PWLPS

For layer $l$, we define:

- Feedforward weights: $W_l \in \mathbb{R}^{d_l \times d_{l-1}}$
- Local classifier head: $C_l \in \mathbb{R}^{K \times d_l}$
- Precision parameter: $\pi_l > 0$

**Feedforward prediction:**

$$
z_l = W_l a_{l-1}, \qquad \hat{a}_l = \phi(z_l)
$$

**Local classifier:**

$$
u_l = C_l h_l, \qquad p_l = \text{softmax}(u_l)
$$

$$
\ell_l(h_l) = \mathcal{L}_{CE}(C_l h_l, y)
$$

---

### 5. Local Energy Function

The core idea of predictive coding is that a layer minimizes a **local prediction error**. We define the local energy:

$$
E_l(h_l) = \frac{\pi_l}{2} \left\| h_l - \hat{a}_l \right\|^2 + \lambda \, \ell_l(h_l)
$$

Substituting $\hat{a}_l = \phi(W_l a_{l-1})$:

$$
E_l(h_l) = \frac{\pi_l}{2} \left\| h_l - \phi(W_l a_{l-1}) \right\|^2 + \lambda \, \mathcal{L}_{CE}(C_l h_l, y)
$$

where:

- $\pi_l$ is the precision (confidence weight)
- $\lambda$ is the classification loss weight
- $h_l$ is the layer activity during settling
- $\hat{a}_l$ is the feedforward prediction

---

### 6. Activity Settling Dynamics

We perform local gradient descent on $h_l$, keeping weights frozen.

**Continuous form:**

$$
\tau \frac{d h_l}{dt} = -\nabla_{h_l} E_l(h_l)
$$

**Discrete form:**

$$
h_l^{(t+1)} = h_l^{(t)} - \gamma \, \nabla_{h_l} E_l\left(h_l^{(t)}\right)
$$

where $\gamma$ is the settling step size and $T$ is the total number of settling steps.

---

### 7. Gradient of the Local Energy

The energy is:

$$
E_l(h_l) = \frac{\pi_l}{2}\left\|h_l - \hat{a}_l\right\|^2 + \lambda \, \ell_l(h_l)
$$

**Prediction term gradient:**

$$
\nabla_{h_l} \left[ \frac{\pi_l}{2}\left\|h_l - \hat{a}_l\right\|^2 \right] = \pi_l (h_l - \hat{a}_l)
$$

**Classification term gradient:** Let $p_l = \text{softmax}(C_l h_l)$ and $r_l = p_l - e_y$. Then:

$$
\nabla_{h_l} \ell_l(h_l) = C_l^T r_l
$$

**Combined:**

$$
\nabla_{h_l} E_l(h_l) = \pi_l (h_l - \hat{a}_l) + \lambda C_l^T (p_l - e_y)
$$

**Settling update:**

$$
h_l^{(t+1)} = h_l^{(t)} - \gamma \left[ \pi_l \left(h_l^{(t)} - \hat{a}_l\right) + \lambda C_l^T \left(p_l^{(t)} - e_y\right) \right]
$$

After each update, apply a projection:

$$
h_l^{(t+1)} = \mathcal{P}\left(h_l^{(t+1)}\right)
$$

In the implementation, $\mathcal{P}$ is ReLU plus clamping.

---

### 8. Settled Activity

After $T$ settling steps:

$$
h_l^* = h_l^{(T)}
$$

This is detached from the computation graph:

$$
h_l^* = \text{sg}\left[h_l^{(T)}\right]
$$

where $\text{sg}[\cdot]$ denotes stop-gradient.

---

### 9. Local Weight Learning Objective

The local learning objective is:

$$
\mathcal{J}_l = \mathcal{J}_l^{pred} + \lambda \mathcal{J}_l^{cls} + \lambda \mathcal{J}_l^{settled} + \mathcal{R}_l^{\pi}
$$

**(a) Predictive matching loss:**

$$
\mathcal{J}_l^{pred} = \frac{\tilde{\pi}_l}{2} \left\| h_l^* - a_l \right\|^2
$$

where $\tilde{\pi}_l = \text{sg}[\pi_l]$ and $a_l = \phi(W_l a_{l-1})$.

**(b) Local classification loss on current feedforward activity:**

$$
\mathcal{J}_l^{cls} = \mathcal{L}_{CE}(C_l a_l, y)
$$

**(c) Local classification loss on settled activity:**

$$
\mathcal{J}_l^{settled} = \mathcal{L}_{CE}(C_l h_l^*, y)
$$

**(d) Precision regularization:** If precision is learned, with $\pi_l = \text{softplus}(s_l) + \epsilon$:

$$
\mathcal{R}_l^{\pi} = \frac{1}{2} \pi_l \cdot \text{sg}\left[\left\|h_l^* - a_l\right\|^2\right] + \rho (\pi_l - \pi_0)^2
$$

---

### 10. Gradient of the Predictive Matching Loss

Define the prediction error:

$$
\delta_l^{pred} = h_l^* - a_l
$$

Then:

$$
\frac{\partial \mathcal{J}_l^{pred}}{\partial W_l} = -\tilde{\pi}_l \left( \delta_l^{pred} \odot \phi'(z_l) \right) a_{l-1}^T
$$

The weight update is:

$$
W_l \leftarrow W_l + \eta \tilde{\pi}_l \left( \delta_l^{pred} \odot \phi'(z_l) \right) a_{l-1}^T
$$

This is a **local Hebbian-style predictive error update**.

---

### 11. Gradient of the Local Classification Loss

Let $r_l = p_l - e_y$. Then:

$$
\frac{\partial \mathcal{J}_l^{cls}}{\partial W_l} = \left[ (C_l^T r_l) \odot \phi'(z_l) \right] a_{l-1}^T
$$

The classification contribution to the weight update is:

$$
W_l \leftarrow W_l - \eta \lambda \left[ (C_l^T r_l) \odot \phi'(z_l) \right] a_{l-1}^T
$$

---

### 12. Gradient for the Local Classifier Head

For the current feedforward activity:

$$
\frac{\partial \mathcal{J}_l^{cls}}{\partial C_l} = r_l a_l^T
$$

For the settled activity, let $p_l^* = \text{softmax}(C_l h_l^*)$ and $r_l^* = p_l^* - e_y$:

$$
\frac{\partial \mathcal{J}_l^{settled}}{\partial C_l} = r_l^* (h_l^*)^T
$$

Total classifier gradient:

$$
\nabla_{C_l} \mathcal{J}_l = \lambda r_l a_l^T + \lambda r_l^* (h_l^*)^T
$$

Update:

$$
C_l \leftarrow C_l - \eta \nabla_{C_l} \mathcal{J}_l
$$

---

### 13. Precision Update

With $\pi_l = \text{softplus}(s_l) + \epsilon$:

$$
\frac{\partial \mathcal{R}_l^{\pi}}{\partial \pi_l} = \frac{1}{2} E_l^{detached} + 2\rho(\pi_l - \pi_0)
$$

where $E_l^{detached} = \text{sg}\left[\|h_l^* - a_l\|^2\right]$.

Since $\frac{\partial \pi_l}{\partial s_l} = \sigma(s_l)$, where $\sigma$ is the sigmoid:

$$
s_l \leftarrow s_l - \eta \left[ \frac{1}{2} E_l^{detached} + 2\rho(\pi_l - \pi_0) \right] \sigma(s_l)
$$

---

### 14. Forward-Only Property

After layer $l$ is processed, its output is detached before being passed to layer $l+1$:

$$
a_l = \text{sg}[\phi(W_l a_{l-1})]
$$

Therefore:

$$
\frac{\partial \mathcal{J}_{l+1}}{\partial W_l} = 0
$$

There is **no gradient flow** from layer $l+1$ back to layer $l$. This is the fundamental difference from backpropagation.

---

### 15. Memory and Time Complexity

**Memory:**

Backpropagation stores activations from all layers:

$$
M_{BP} \approx O(Ld)
$$

PWLPS processes layers sequentially and discards previous graphs:

$$
M_{local} \approx O(d)
$$

So for deep networks:

$$
M_{local} \ll M_{BP}
$$

**Time:**

Backpropagation per minibatch:

$$
T_{BP} \approx O(L)
$$

PWLPS performs $T$ settling steps per layer:

$$
T_{local} \approx O(LT)
$$

This explains why PWLPS is slower but more memory-efficient.

---

## 📜 Full Algorithm

```text
Algorithm 1: Precision-Weighted Local Predictive Settling (PWLPS)

Input:  Minibatch (x, y), number of layers L
        Hyperparameters: η (weight lr), γ (settle lr), T (settle steps),
                         λ (cls weight), ρ (precision reg), π₀ (target precision)

Output: Updated weights {W_l, C_l, s_l} for all layers

─────────────────────────────────────────────────────────────────
a₀ ← x

for l = 1 to L do

    ── Step 1: Feedforward prediction ─────────────────────────
    z_l ← W_l · a_{l-1}
    â_l ← φ(z_l)
    h_l^(0) ← â_l

    ── Step 2: Local settling (Phase 1 — Inference) ──────────
    for t = 0 to T-1 do
        p_l^(t) ← softmax(C_l · h_l^(t))
        r_l^(t) ← p_l^(t) − e_y
        g_h ← π_l·(h_l^(t) − â_l) + λ·C_l^T·r_l^(t)
        h_l^(t+1) ← h_l^(t) − γ·g_h
        h_l^(t+1) ← 𝒫(h_l^(t+1))          # ReLU + clamp
    end for

    h_l* ← sg[h_l^(T)]                    # settled activity

    ── Step 3: Local loss (Phase 2 — Learning) ───────────────
    a_l ← φ(W_l · a_{l-1})                # current feedforward
    Compute 𝒥_l = 𝒥_l^pred + λ·𝒥_l^cls + λ·𝒥_l^settled + 𝒭_l^π

    ── Step 4: Update local parameters ───────────────────────
    W_l ← W_l − η · ∇_{W_l} 𝒥_l
    C_l ← C_l − η · ∇_{C_l} 𝒥_l
    if precision is learned then
        s_l ← s_l − η · ∇_{s_l} 𝒭_l^π
    end if

    ── Step 5: Pass activity forward (detached) ──────────────
    a_l ← sg[a_l]

end for
─────────────────────────────────────────────────────────────────
```

---

## 🔢 Solved Numerical Example

We demonstrate the full algorithm on a **one-layer toy network**.

### Setup

| Symbol | Value |
|--------|-------|
| Input $x$ | $\begin{bmatrix} 1 \\ 0 \end{bmatrix}$ |
| True class $y$ | $1$, so $e_y = \begin{bmatrix} 0 \\ 1 \end{bmatrix}$ |
| Weight $W$ | $\begin{bmatrix} 0.20 & -0.10 \end{bmatrix}$ |
| Classifier $C$ | $\begin{bmatrix} 0.00 \\ 1.00 \end{bmatrix}$ |
| Activation $\phi$ | ReLU |
| $\pi$ | $1$ |
| $\lambda$ | $1$ |
| $\gamma$ (settle lr) | $0.1$ |
| $\eta$ (weight lr) | $0.1$ |
| $T$ (settle steps) | $1$ |

---

### Step 1: Feedforward Prediction

$$
z = Wx = 0.20(1) + (-0.10)(0) = 0.20
$$

$$
\hat{a} = \phi(z) = \text{ReLU}(0.20) = 0.20
$$

Initialize:

$$
h^{(0)} = \hat{a} = 0.20
$$

---

### Step 2: Local Settling

Compute local logits:

$$
u = C h^{(0)} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}(0.20) = \begin{bmatrix} 0 \\ 0.20 \end{bmatrix}
$$

Compute softmax (using $e^{0.20} \approx 1.2214$):

$$
p_0 = \frac{e^0}{e^0 + e^{0.20}} = \frac{1}{1 + 1.2214} \approx 0.4502
$$

$$
p_1 = \frac{e^{0.20}}{e^0 + e^{0.20}} = \frac{1.2214}{2.2214} \approx 0.5498
$$

Compute softmax error:

$$
r = p - e_y = \begin{bmatrix} 0.4502 \\ 0.5498 \end{bmatrix} - \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.4502 \\ -0.4502 \end{bmatrix}
$$

Classification gradient with respect to activity:

$$
\nabla_h \ell = C^T r = \begin{bmatrix} 0 & 1 \end{bmatrix} \begin{bmatrix} 0.4502 \\ -0.4502 \end{bmatrix} = -0.4502
$$

Prediction error term:

$$
\pi(h^{(0)} - \hat{a}) = 1(0.20 - 0.20) = 0
$$

Total activity gradient:

$$
\nabla_h E = 0 + 1(-0.4502) = -0.4502
$$

Update activity:

$$
h^{(1)} = h^{(0)} - \gamma \nabla_h E = 0.20 - 0.1(-0.4502) = 0.2450
$$

Apply ReLU:

$$
h^* = 0.2450
$$

---

### Step 3: Predictive Weight Update

Current feedforward activity: $a = 0.20$. Settled activity: $h^* = 0.2450$.

Prediction error:

$$
\delta^{pred} = h^* - a = 0.2450 - 0.20 = 0.0450
$$

Since $z = 0.20 > 0$, for ReLU: $\phi'(z) = 1$.

Prediction gradient:

$$
\nabla_W \mathcal{J}^{pred} = -\pi \delta^{pred} \phi'(z) x^T = -1(0.0450)(1)\begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} -0.0450 & 0 \end{bmatrix}
$$

---

### Step 4: Classification Weight Update

From earlier, $C^T r = -0.4502$.

$$
\nabla_W \mathcal{J}^{cls} = (C^T r)\phi'(z) x^T = -0.4502(1)\begin{bmatrix} 1 & 0 \end{bmatrix} = \begin{bmatrix} -0.4502 & 0 \end{bmatrix}
$$

Total weight gradient:

$$
\nabla_W \mathcal{J} = \begin{bmatrix} -0.0450 & 0 \end{bmatrix} + \begin{bmatrix} -0.4502 & 0 \end{bmatrix} = \begin{bmatrix} -0.4952 & 0 \end{bmatrix}
$$

Update weights:

$$
W \leftarrow W - \eta \nabla_W \mathcal{J} = \begin{bmatrix} 0.20 & -0.10 \end{bmatrix} - 0.1 \begin{bmatrix} -0.4952 & 0 \end{bmatrix} = \begin{bmatrix} 0.2495 & -0.10 \end{bmatrix}
$$

---

### Step 5: Classifier Head Update

Current gradient:

$$
\nabla_C \mathcal{J}^{cls} = r a^T = \begin{bmatrix} 0.4502 \\ -0.4502 \end{bmatrix}(0.20) = \begin{bmatrix} 0.0900 \\ -0.0900 \end{bmatrix}
$$

Settled logits: $u^* = C h^* = \begin{bmatrix} 0 \\ 0.2450 \end{bmatrix}$. Using $e^{0.2450} \approx 1.2776$:

$$
p_0^* = \frac{1}{1 + 1.2776} \approx 0.4391, \qquad p_1^* = \frac{1.2776}{2.2776} \approx 0.5609
$$

$$
r^* = \begin{bmatrix} 0.4391 \\ 0.5609 \end{bmatrix} - \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0.4391 \\ -0.4391 \end{bmatrix}
$$

Settled gradient:

$$
\nabla_C \mathcal{J}^{settled} = r^* (h^*)^T = \begin{bmatrix} 0.4391 \\ -0.4391 \end{bmatrix}(0.2450) = \begin{bmatrix} 0.1076 \\ -0.1076 \end{bmatrix}
$$

Total classifier gradient:

$$
\nabla_C \mathcal{J} = \begin{bmatrix} 0.0900 \\ -0.0900 \end{bmatrix} + \begin{bmatrix} 0.1076 \\ -0.1076 \end{bmatrix} = \begin{bmatrix} 0.1976 \\ -0.1976 \end{bmatrix}
$$

Update classifier:

$$
C \leftarrow C - \eta \nabla_C \mathcal{J} = \begin{bmatrix} 0 \\ 1 \end{bmatrix} - 0.1 \begin{bmatrix} 0.1976 \\ -0.1976 \end{bmatrix} = \begin{bmatrix} -0.0198 \\ 1.0198 \end{bmatrix}
$$

---

### Interpretation

| Stage | Probability of correct class $p_1$ |
|-------|--------------------------------------|
| Before update | $0.5498$ |
| After settling | $0.5609$ |
| After weight update | Higher on next forward pass |

After one local update, the network becomes **more confident about the correct class**. The layer locally adjusted its activity and weights to improve classification — **without any global backward pass**.

---

## 🚀 Installation

### From PyPI

```bash
pip install pwlps
```

### From Source

```bash
git clone https://github.com/yourusername/pwlps.git
cd pwlps
pip install -e .
```

### Requirements

```text
torch>=2.0.0
```

---

## 💡 Usage

### Basic Example

```python
import torch
from pwlps import LocalPredictiveMLP

# 1. Define the model
model = LocalPredictiveMLP(
    input_dim=784,
    hidden_dim=512,
    num_classes=10,
    num_layers=4,
    learn_precision=True,
)

# 2. Create one optimizer per layer
optimizers = [
    torch.optim.Adam(layer.parameters(), lr=1e-3)
    for layer in model.layers
]

# 3. Training step
x = torch.randn(128, 784)
y = torch.randint(0, 10, (128,))

loss = model.local_train_step(
    x=x,
    y=y,
    optimizers=optimizers,
    T=5,             # settling steps
    step_size=0.05,  # settling learning rate
    cls_weight=1.0,  # classification loss weight
)

# 4. Inference
with torch.no_grad():
    logits = model(x)
    preds = logits.argmax(dim=1)
```

### Architecture-Agnostic Example (Universal Wrapper)

PWLPS works with **any differentiable block**, not just MLPs.

```python
import torch
import torch.nn as nn
from pwlps import build_local_sequential

# CNN example
blocks = [
    nn.Sequential(nn.Conv2d(1, 32, 3), nn.ReLU(), nn.MaxPool2d(2)),
    nn.Sequential(nn.Conv2d(32, 64, 3), nn.ReLU(), nn.MaxPool2d(2)),
    nn.Sequential(nn.Flatten(), nn.Linear(64 * 5 * 5, 256), nn.ReLU()),
]

model = build_local_sequential(
    blocks=blocks,
    num_classes=10,
    sample_input=torch.randn(2, 1, 28, 28),
    pool_types=["mean_spatial", "mean_spatial", "identity"],
    learn_precision=True,
)

optimizers = [
    torch.optim.Adam(block.parameters(), lr=1e-3)
    for block in model.blocks
]

x = torch.randn(128, 1, 28, 28)
y = torch.randint(0, 10, (128,))

loss = model.local_train_step(x=x, y=y, optimizers=optimizers, T=3)
```

### Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Settling steps | $T$ | 5 | Local inference iterations per layer |
| Settling step size | $\gamma$ | 0.05 | Learning rate for activity settling |
| Classification weight | $\lambda$ | 1.0 | Weight of local classification loss |
| Fixed precision | $\pi_0$ | 1.0 | Precision value / regularization target |
| Precision regularization | $\rho$ | 0.01 | Keeps learned precision stable |
| Weight learning rate | $\eta$ | 0.001 | Optimizer learning rate |

---

## 🔬 Methodology

### Research Question

> Can a deep neural network be trained using local, forward-only predictive settling with precision weighting, achieving competitive accuracy while reducing activation memory compared to standard backpropagation?

### Hypotheses

1. **H1:** A network trained with local predictive settling can learn useful representations from scratch.
2. **H2:** Predictive settling improves accuracy compared to layer-wise local classification alone ($T > 0$ vs $T = 0$).
3. **H3:** Learned precision improves stability or accuracy compared to fixed precision.
4. **H4:** The local method has slower memory growth with depth than backpropagation.

### Experimental Setup

| Setting | Value |
|---------|-------|
| Dataset | MNIST |
| Hardware | Google Colab, NVIDIA T4 GPU |
| Batch size | 128 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Hidden width | 512 |
| Activation | ReLU |
| Seed | 0 |
| Settling steps $T$ | 5 |
| Settling step size $\gamma$ | 0.05 |
| Classification weight $\lambda$ | 1.0 |

### Conditions Compared

1. **Backpropagation baseline** — standard global gradient descent
2. **Local + fixed precision** — PWLPS with $\pi_l = 1.0$
3. **Local + learned precision** — PWLPS with adaptive $\pi_l$

### Metrics

- Training loss
- Test accuracy
- Peak GPU memory (MB)
- Time per epoch (s)
- Parameter count

### Reproduction Commands

```bash
# Backpropagation baseline (4 layers)
python main.py --method bp --epochs 5 --num-layers 4 --hidden-dim 512

# Local method with fixed precision (4 layers)
python main.py --method local --epochs 5 --num-layers 4 --hidden-dim 512 --T 5

# Local method with learned precision (4 layers)
python main.py --method local --learn-precision --epochs 5 --num-layers 4 --hidden-dim 512 --T 5

# Deep backpropagation baseline (16 layers)
python main.py --method bp --epochs 3 --num-layers 16 --hidden-dim 512

# Deep local method (16 layers)
python main.py --method local --epochs 3 --num-layers 16 --hidden-dim 512 --T 5
```

---

## 📊 Experimental Results

### 4-Layer Network

| Method | Parameters | Final Test Acc | Best Test Acc | Peak Memory | Avg Time/Epoch |
|--------|-----------|----------------|---------------|-------------|----------------|
| Backprop | 1,195,018 | 97.54% | 97.66% | 41.12 MB | 14.87 s |
| Local Fixed Precision | 1,210,408 | 97.50% | 97.55% | 38.35 MB | 30.30 s |
| Local Learned Precision | 1,210,412 | 97.52% | 97.83% | 38.36 MB | 29.25 s |

**Findings:**

- Local method matches backpropagation accuracy (within $0.04\%$).
- Memory reduction:

$$
\frac{41.12 - 38.35}{41.12} \times 100 \approx 6.74\%
$$

- Time overhead $\approx 2.04\times$ (expected due to settling steps).

### 16-Layer Network (Deep)

| Method | Parameters | Test Acc (3 epochs) | Peak Memory | Avg Time/Epoch |
|--------|-----------|---------------------|-------------|----------------|
| Backprop | 4,346,890 | 82.34% | 101.24 MB | 16.83 s |
| Local Fixed Precision | 4,423,840 | 96.89% | 87.41 MB | 70.77 s |

**Findings:**

- Memory reduction:

$$
\frac{101.24 - 87.41}{101.24} \times 100 \approx 13.66\%
$$

- Accuracy improvement after 3 epochs:

$$
96.89\% - 82.34\% = 14.55\%
$$

The memory advantage **grows with depth**, and local classifier heads help deep networks converge faster by providing direct supervised signals to early layers (mitigating weak gradient flow).

---

## 📁 Repository Structure

```text
pwlps/
│
├── pyproject.toml          # Build & packaging config
├── README.md               # This file
├── LICENSE                 # MIT License
│
├── pwlps/
│   ├── __init__.py         # Package exports
│   ├── core.py             # MLP implementation
│   └── universal.py        # Architecture-agnostic implementation
│
├── main.py                 # Experiment runner
│
├── examples/
│   ├── mnist_mlp.py        # MLP example
│   └── mnist_cnn.py        # CNN example
│
└── notebooks/
    └── ResearchPro.ipynb   # Colab experiment notebook
```

---

## 📖 Citation

If you use this code in your research, please cite:

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

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This project draws inspiration from:

- Predictive coding theories of cortical computation
- Local learning rules and biologically plausible credit assignment
- Research on alternatives to backpropagation (Feedback Alignment, Target Propagation, Equilibrium Propagation, Forward-Forward)
