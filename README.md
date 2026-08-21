# PWLPS: Precision-Weighted Local Predictive Settling

[![PyPI version](https://badge.fury.io/py/pwlps.svg)](https://badge.fury.io/py/pwlps)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)

**PWLPS** is a forward-only, local learning algorithm for deep neural networks. It replaces global backpropagation with **layer-wise predictive settling**, **local classification heads**, and **precision-weighted prediction errors**.

Unlike standard backpropagation, PWLPS does **not** propagate gradients through the entire network. Instead, each layer learns using only **local information**, resulting in **lower activation memory** and a **more biologically plausible** learning rule.

---

## Table of Contents

- [Key Features](#-key-features)
- [How It Works](#-how-it-works-intuition)
- [Mathematical Formulation](#-mathematical-formulation)
- [Full Algorithm](#-full-algorithm)
- [Solved Numerical Example](#-solved-numerical-example)
- [Experimental Results](#-experimental-results)
- [Citation](#-citation)
- [License](#-license)

---

##  Key Features

| Feature | Description |
|---|---|
|  **Forward-only training** | No global backward pass through the full network |
|  **Biologically plausible** | Each layer uses only local signals |
|  **Memory-efficient** | Does not store the full computation graph across layers |
|  **Local classifier heads** | Every layer gets its own supervised signal |
|  **Precision weighting** | Adaptive confidence per layer (fixed or learned) |
|  **Architecture-agnostic** | Works with MLPs, CNNs, Transformers, RNNs, and more |

---

##  How It Works (Intuition)

### Standard Backpropagation — "Global Blame"

In normal deep learning, the output makes a prediction. If it is wrong, a global error signal travels **backward** through every layer:

```
Input → Layer 1 → Layer 2 → Layer 3 → Output
Output → Layer 3 → Layer 2 → Layer 1   (backward pass)
```

This requires storing activations from **all layers** so gradients can be computed later. Memory grows with depth.

### PWLPS — "Local Self-Correction"

In PWLPS, each layer does **not wait** for a global error. Instead, each layer asks:

> *"Based on the input I received, can I produce a representation that helps predict the correct label right now?"*

Each layer runs a small **local optimization** (called **settling**) to find a good internal representation, then updates its own weights to make that representation automatic.

```
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

##  Mathematical Formulation

### 1. Problem Setup

Let the training dataset be $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$, where:

- $x_i \in \mathbb{R}^{d_0}$ is the input sample
- $y_i \in \{1, 2, \dots, K\}$ is the class label

For MNIST: $d_0 = 784$ and $K = 10$.

We learn a function $f_\theta(x) = \hat{y}$, where $\theta$ denotes all trainable parameters.

---

### 2. Cross-Entropy Loss

Given logits $u \in \mathbb{R}^K$, the softmax probability for class $k$ is:

$$p_k = \frac{e^{u_k}}{\sum_{j=1}^{K} e^{u_j}}$$

The cross-entropy loss for true class $y$ is:

$$\mathcal{L}_{CE}(u, y) = -\log(p_y)$$

Let $e_y \in \mathbb{R}^K$ be the one-hot vector for class $y$. The **softmax error** is:

$$p = \text{softmax}(u), \qquad r = p - e_y$$

---

### 3. Standard Backpropagation Baseline

A feedforward network with $L$ layers computes:

$$z_l = W_l a_{l-1}, \qquad a_l = \phi(z_l)$$

where $a_0 = x$, $z_l$ is the pre-activation, $a_l$ is the post-activation, $W_l$ is the weight matrix, and $\phi$ is the activation function (e.g., ReLU).

Backpropagation computes the gradient via the global chain rule:

$$\frac{\partial \mathcal{L}}{\partial W_l} = \frac{\partial \mathcal{L}}{\partial a_L} \frac{\partial a_L}{\partial a_{L-1}} \cdots \frac{\partial a_{l+1}}{\partial a_l} \frac{\partial a_l}{\partial W_l}$$

This requires storing activations from **all** layers, giving memory cost $M_{BP} = O(Ld)$, where $L$ is depth and $d$ is hidden width.

---

### 4. Layer Components in PWLPS

For layer $l$, we define:

| Symbol | Shape | Role |
|---|---|---|
| $W_l$ | $\mathbb{R}^{d_l \times d_{l-1}}$ | Feedforward weights |
| $C_l$ | $\mathbb{R}^{K \times d_l}$ | Local classifier head |
| $\pi_l > 0$ | scalar | Precision parameter |

**Feedforward prediction:**

$$z_l = W_l a_{l-1}, \qquad \hat{a}_l = \phi(z_l)$$

**Local classifier:**

$$u_l = C_l h_l, \qquad p_l = \text{softmax}(u_l), \qquad \ell_l(h_l) = \mathcal{L}_{CE}(C_l h_l, y)$$

---

### 5. Local Energy Function

The core idea of predictive coding is that a layer minimizes a **local prediction error**:

$$E_l(h_l) = \frac{\pi_l}{2} \left\| h_l - \hat{a}_l \right\|^2 + \lambda \ell_l(h_l)$$

Substituting $\hat{a}\_l = \phi(W\_l a\_{l-1})$:

$$E_l(h_l) = \frac{\pi_l}{2} \left\| h_l - \phi(W_l a_{l-1}) \right\|^2 + \lambda \mathcal{L}_{CE}(C_l h_l, y)$$

where $\pi_l$ is the precision (confidence weight), $\lambda$ is the classification loss weight, $h_l$ is the layer activity during settling, and $\hat{a}_l$ is the feedforward prediction.

---

### 6. Activity Settling Dynamics

We perform local gradient descent on $h_l$, keeping weights frozen.

**Continuous form:**

$$\tau \frac{d h_l}{dt} = -\nabla_{h_l} E_l(h_l)$$

**Discrete update:**

$$h_l^{(t+1)} = h_l^{(t)} - \gamma \, \nabla_{h_l} E_l\!\left(h_l^{(t)}\right)$$

where $\gamma$ is the settling step size and $T$ is the total number of settling steps.

---

### 7. Gradient of the Local Energy

**Prediction term:**

$$\nabla_{h_l} \left[ \frac{\pi_l}{2}\left\|h_l - \hat{a}_l\right\|^2 \right] = \pi_l (h_l - \hat{a}_l)$$

**Classification term** — let $p_l = \text{softmax}(C_l h_l)$ and $r_l = p_l - e_y$:

$$\nabla_{h_l} \ell_l(h_l) = C_l^T r_l$$

**Combined gradient:**

$$\nabla_{h_l} E_l(h_l) = \pi_l (h_l - \hat{a}_l) + \lambda C_l^T (p_l - e_y)$$

**Full settling update:**

$$h_l^{(t+1)} = h_l^{(t)} - \gamma \left[ \pi_l \left(h_l^{(t)} - \hat{a}_l\right) + \lambda C_l^T \left(p_l^{(t)} - e_y\right) \right]$$

After each update, apply projection $\mathcal{P}$ (ReLU + clamp):

$$h_l^{(t+1)} = \mathcal{P}\!\left(h_l^{(t+1)}\right)$$

---

### 8. Settled Activity

After $T$ settling steps, the settled activity is detached from the computation graph:

$$h_l^* = \text{sg}\!\left[h_l^{(T)}\right]$$

where $\text{sg}[\cdot]$ denotes stop-gradient.

---

### 9. Local Weight Learning Objective

$$\mathcal{J}_l = \mathcal{J}_l^{pred} + \lambda \mathcal{J}_l^{cls} + \lambda \mathcal{J}_l^{settled} + \mathcal{R}_l^{\pi}$$

**(a) Predictive matching loss — where $\tilde{\pi}\_l = \text{sg}[\pi\_l]$ and $a\_l = \phi(W\_l a\_{l-1})$:**

$$\mathcal{J}_l^{pred} = \frac{\tilde{\pi}_l}{2} \left\| h_l^* - a_l \right\|^2$$

**(b) Classification loss on current feedforward activity:**

$$\mathcal{J}_l^{cls} = \mathcal{L}_{CE}(C_l a_l, y)$$

**(c) Classification loss on settled activity:**

$$\mathcal{J}_l^{settled} = \mathcal{L}_{CE}(C_l h_l^*, y)$$

**(d) Precision regularization** — if precision is learned, with $\pi_l = \text{softplus}(s_l) + \epsilon$:

$$\mathcal{R}_l^{\pi} = \frac{1}{2} \pi_l \cdot \text{sg}\!\left[\left\|h_l^* - a_l\right\|^2\right] + \rho (\pi_l - \pi_0)^2$$

---

### 10. Gradient of the Predictive Matching Loss

Define the prediction error $\delta_l^{pred} = h_l^* - a_l$. Then:

$$\frac{\partial \mathcal{J}_l^{pred}}{\partial W_l} = -\tilde{\pi}_l \left( \delta_l^{pred} \odot \phi'(z_l) \right) a_{l-1}^T$$

Weight update (local Hebbian-style predictive error update):

$$W_l \leftarrow W_l + \eta \tilde{\pi}_l \left( \delta_l^{pred} \odot \phi'(z_l) \right) a_{l-1}^T$$

---

### 11. Gradient of the Local Classification Loss

Let $r_l = p_l - e_y$:

$$\frac{\partial \mathcal{J}_l^{cls}}{\partial W_l} = \left[ (C_l^T r_l) \odot \phi'(z_l) \right] a_{l-1}^T$$

Classification contribution to the weight update:

$$W_l \leftarrow W_l - \eta \lambda \left[ (C_l^T r_l) \odot \phi'(z_l) \right] a_{l-1}^T$$

---

### 12. Gradient for the Local Classifier Head

For current feedforward activity:

$$\frac{\partial \mathcal{J}_l^{cls}}{\partial C_l} = r_l a_l^T$$

For settled activity, let

$$p_l^{\ast} = \text{softmax}(C_l h_l^{\ast}) \quad \text{and} \quad r_l^{\ast} = p_l^{\ast} - e_y$$

$$\frac{\partial \mathcal{J}_l^{settled}}{\partial C_l} = r_l^* (h_l^*)^T$$

Total classifier gradient and update:

$$\nabla_{C_l} \mathcal{J}_l = \lambda r_l a_l^T + \lambda r_l^* (h_l^*)^T$$

$$C_l \leftarrow C_l - \eta \nabla_{C_l} \mathcal{J}_l$$

---

### 13. Precision Update

With $\pi_l = \text{softplus}(s_l) + \epsilon$ and $E_l^{detached} = \text{sg}\!\left[\|h_l^* - a_l\|^2\right]$:

$$\frac{\partial \mathcal{R}_l^{\pi}}{\partial \pi_l} = \frac{1}{2} E_l^{detached} + 2\rho(\pi_l - \pi_0)$$

Since $\partial \pi_l / \partial s_l = \sigma(s_l)$ (sigmoid):

$$s_l \leftarrow s_l - \eta \left[ \frac{1}{2} E_l^{detached} + 2\rho(\pi_l - \pi_0) \right] \sigma(s_l)$$

---

### 14. Forward-Only Property

After layer $l$ is processed, its output is detached before being passed to layer $l+1$:

$$a_l = \text{sg}[\phi(W_l a_{l-1})]$$

Therefore $\partial \mathcal{J}_{l+1} / \partial W_l = 0$. There is **no gradient flow** from layer $l+1$ back to layer $l$.

---

### 15. Memory and Time Complexity

| Property | Backpropagation | PWLPS |
|---|---|---|
| **Memory** | $O(Ld)$ — stores all activations | $O(d)$ — processes layers sequentially |
| **Time per minibatch** | $O(L)$ | $O(LT)$ — $T$ settling steps per layer |

For deep networks: $M_{local} \ll M_{BP}$.

The time overhead from settling steps ($T$) is the trade-off for memory efficiency.

---

##  Full Algorithm

```
Algorithm 1: Precision-Weighted Local Predictive Settling (PWLPS)

Input:  Minibatch (x, y), number of layers L
        Hyperparameters: η (weight lr), γ (settle lr), T (settle steps),
                         λ (cls weight), ρ (precision reg), π₀ (target precision)

Output: Updated weights {W_l, C_l, s_l} for all layers

────────────────────────────────────────────────────────────────
a₀ ← x

for l = 1 to L do

  ── Step 1: Feedforward prediction ──────────────────────────
  z_l ← W_l · a_{l-1}
  â_l ← φ(z_l)
  h_l^(0) ← â_l

  ── Step 2: Local settling (Phase 1 — Inference) ────────────
  for t = 0 to T-1 do
      p_l^(t) ← softmax(C_l · h_l^(t))
      r_l^(t) ← p_l^(t) − e_y
      g_h    ← π_l·(h_l^(t) − â_l) + λ·C_l^T·r_l^(t)
      h_l^(t+1) ← h_l^(t) − γ·g_h
      h_l^(t+1) ← 𝒫(h_l^(t+1))          # ReLU + clamp
  end for

  h_l* ← sg[h_l^(T)]                    # settled activity (detached)

  ── Step 3: Local loss (Phase 2 — Learning) ─────────────────
  a_l ← φ(W_l · a_{l-1})                # current feedforward
  Compute 𝒥_l = 𝒥_l^pred + λ·𝒥_l^cls + λ·𝒥_l^settled + 𝒭_l^π

  ── Step 4: Update local parameters ─────────────────────────
  W_l ← W_l − η · ∇_{W_l} 𝒥_l
  C_l ← C_l − η · ∇_{C_l} 𝒥_l
  if precision is learned then
      s_l ← s_l − η · ∇_{s_l} 𝒭_l^π
  end if

  ── Step 5: Pass activity forward (detached) ─────────────────
  a_l ← sg[a_l]

end for
────────────────────────────────────────────────────────────────
```

---

##  Solved Numerical Example

A complete single-step walkthrough on a **one-layer toy network**.

### Setup

| Symbol | Value |
|---|---|
| Input $x$ | $[1,\ 0]^T$ |
| True class $y$ | $1$, so $e_y = [0,\ 1]^T$ |
| Weight $W$ | $[0.20,\ {-0.10}]$ |
| Classifier $C$ | $[0.00,\ 1.00]^T$ |
| Activation $\phi$ | ReLU |
| Precision $\pi$ | $1$ |
| Classification weight $\lambda$ | $1$ |
| Settling lr $\gamma$ | $0.1$ |
| Weight lr $\eta$ | $0.1$ |
| Settling steps $T$ | $1$ |

---

### Step 1 — Feedforward Prediction

$$z = Wx = 0.20(1) + (-0.10)(0) = 0.20$$

$$\hat{a} = \phi(z) = \text{ReLU}(0.20) = 0.20, \qquad h^{(0)} = \hat{a} = 0.20$$

---

### Step 2 — Local Settling

Compute local logits and softmax (using $e^{0.20} \approx 1.2214$):

$$u = C h^{(0)} = [0,\ 1]^T \cdot 0.20 = [0,\ 0.20]^T$$

$$p_0 = \frac{1}{1 + 1.2214} \approx 0.4502, \qquad p_1 = \frac{1.2214}{2.2214} \approx 0.5498$$

Softmax error and activity gradient:

$$r = p - e_y = [0.4502,\ -0.4502]^T$$

$$\nabla_h \ell = C^T r = [0,\ 1] \cdot [0.4502,\ -0.4502]^T = -0.4502$$

$$\nabla_h E = \underbrace{\pi(h^{(0)} - \hat{a})}_{= 0} + \lambda(-0.4502) = -0.4502$$

Settling update with ReLU projection:

$$h^{(1)} = 0.20 - 0.1(-0.4502) = 0.2450 \implies h^* = 0.2450$$

---

### Step 3 — Predictive Weight Update

Prediction error: $\delta^{pred} = h^* - a = 0.2450 - 0.20 = 0.0450$

Since $z = 0.20 > 0$: $\phi'(z) = 1$

$$\nabla_W \mathcal{J}^{pred} = -\pi \cdot \delta^{pred} \cdot \phi'(z) \cdot x^T = -1(0.0450)(1)[1,\ 0] = [-0.0450,\ 0]$$

---

### Step 4 — Classification Weight Update

$$\nabla_W \mathcal{J}^{cls} = (C^T r)\phi'(z)\, x^T = (-0.4502)(1)[1,\ 0] = [-0.4502,\ 0]$$

Total gradient and weight update:

$$\nabla_W \mathcal{J} = [-0.0450,\ 0] + [-0.4502,\ 0] = [-0.4952,\ 0]$$

$$W \leftarrow [0.20,\ {-0.10}] - 0.1 \cdot [-0.4952,\ 0] = [0.2495,\ {-0.10}]$$

---

### Step 5 — Classifier Head Update

**From current activity** ($a = 0.20$):

$$\nabla_C \mathcal{J}^{cls} = r\, a^T = [0.4502,\ {-0.4502}]^T \cdot 0.20 = [0.0900,\ {-0.0900}]^T$$

**From settled activity** ($h^* = 0.2450$, $e^{0.2450} \approx 1.2776$):

$$p_0^* \approx 0.4391, \qquad p_1^* \approx 0.5609$$

$$r^* = [0.4391,\ {-0.4391}]^T$$

$$\nabla_C \mathcal{J}^{settled} = r^* (h^*)^T = [0.1076,\ {-0.1076}]^T$$

Total update:

$$\nabla_C \mathcal{J} = [0.0900,\ {-0.0900}]^T + [0.1076,\ {-0.1076}]^T = [0.1976,\ {-0.1976}]^T$$

$$C \leftarrow [0,\ 1]^T - 0.1 \cdot [0.1976,\ {-0.1976}]^T = [{-0.0198},\ 1.0198]^T$$

---

### Result Summary

| Stage | $p_1$ (probability of correct class) |
|---|---|
| Before update | $0.5498$ |
| After settling | $0.5609$ |
| After weight update | Higher on next forward pass |

After one local update, the network becomes more confident about the correct class — **without any global backward pass**.



### Experimental Setup

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
| Settling steps $T$ | `5` |
| Settling step size $\gamma$ | `0.05` |
| Classification weight $\lambda$ | `1.0` |

### Conditions

1. **Backpropagation baseline** — standard global gradient descent
2. **Local + fixed precision** — PWLPS with $\pi_l = 1.0$
3. **Local + learned precision** — PWLPS with adaptive $\pi_l$

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

##  Experimental Results

### 16-Layer Network (3 epochs)

| Method | Parameters | Test Acc | Peak Memory | Time/Epoch |
|---|---|---|---|---|
| Backprop | 4,346,890 | 82.34% | 101.24 MB | 16.83 s |
| Local Fixed Precision | 4,423,840 | 96.89% | 87.41 MB | 70.77 s |

**Key findings:**
- Memory reduction: **(101.24 − 87.41) / 101.24 × 100 ≈ 13.66%**
- Accuracy improvement after 3 epochs: **+14.55 pp**

The memory advantage **grows with depth**, and local classifier heads help deep networks converge faster by providing direct supervised signals to early layers — mitigating weak gradient flow in very deep networks.

---



##  Citation

If you use this code in your research, please cite:

```bibtex
@misc{pwlps2026,
  title        = {PWLPS: Precision-Weighted Local Predictive Settling},
  author       = {Abdul Mofique Siddiqui},
  year         = {2026},
  howpublished = {\url{https://github.com/Luckyy0311/pwlps}},
  note         = {A forward-only local learning algorithm for deep neural networks}
}
```

---

##  License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

##  Acknowledgments

This project draws inspiration from:

- Predictive coding theories of cortical computation
- Local learning rules and biologically plausible credit assignment
- Research on alternatives to backpropagation: Feedback Alignment, Target Propagation, Equilibrium Propagation, Forward-Forward
