---
title: Neural Network Fundamentals
description: From the perceptron to multi-layer networks, loss functions, backpropagation with shapes, activation functions, vanishing gradients and weight initialization.
act: 2
priority: Critical
tags:
  - ML Engineer
  - Applied Scientist
  - AI Engineer
---

# Neural Network Fundamentals

!!! abstract "TL;DR"
    A neural network stacks linear layers with non-linear **activations**, so hidden layers learn their own features. **Backpropagation** is the chain rule applied backward through the computation graph, giving every parameter's gradient in about the cost of two forward passes. Deep networks were hard to train because gradients **vanish or explode** through many layers; ReLU-family activations, careful **initialization**, normalization and residual connections fixed that.

!!! evolution "Where this fits"
    **Before:** [logistic regression](../classical-ml/linear-models.md) is a one-layer network; [SVMs](../classical-ml/knn-nb-svm.md) need a hand-picked kernel. **This page:** learning features with layers. **Next:** [Training Deep Networks](training.md) covers optimizers, normalization and regularization.

## From perceptron to multi-layer network

**The problem:** linear models can't learn XOR or any interaction unless someone engineers the feature.

<div class="metro act-2" markdown>

- **McCulloch–Pitts neuron** <span class="yr">1943</span><br>A threshold unit as a model of a neuron. No learning rule.
- **Perceptron (Rosenblatt)** <span class="yr">1958</span><br>Learns weights with an error-driven rule. Only linear boundaries.
- **Perceptrons (Minsky & Papert)** <span class="yr">1969</span><br>Showed single layers can't learn XOR; funding and interest collapsed.
- **Backpropagation popularized** <span class="yr">1986</span><br>Multi-layer networks become trainable.
- **Deep learning revival** <span class="yr">2006–2012</span><br>Layer-wise pretraining, GPUs, ReLU, big data, then AlexNet.

</div>

**Core idea:** stack layers. Each layer computes $h = \phi(Wx + b)$, a linear map followed by a non-linearity $\phi$. Hidden layers transform the input into features that make the final (linear) layer's job easy.

Why the non-linearity is essential: without it, $W_2(W_1x) = (W_2W_1)x$ collapses into a single linear layer, however many layers you stack.

**Universal approximation theorem** (Cybenko 1989; Hornik 1991): a network with one hidden layer and enough units can approximate any continuous function on a bounded domain to any accuracy. It says nothing about how many units you need, whether gradient descent can find the weights, or how well it generalizes. In practice **depth** is far more parameter-efficient than width: deep networks compose simple features into complex ones (edges → textures → parts → objects).

### Shapes and parameter counts

For a batch $X$ of shape $(n, d_{in})$, a dense layer with weights $W$ of shape $(d_{in}, d_{out})$ and bias $(d_{out},)$ produces $(n, d_{out})$ and has $d_{in}d_{out} + d_{out}$ parameters. An MLP $784 \to 256 \to 10$ has $784\cdot256 + 256 + 256\cdot10 + 10 = 203{,}530$ parameters.

## Loss functions

| Task | Output layer | Loss |
|---|---|---|
| Regression | Linear | MSE (or MAE, Huber) |
| Binary classification | One logit, sigmoid | Binary cross-entropy |
| Multiclass (one label) | $K$ logits, softmax | Categorical cross-entropy |
| Multi-label | $K$ logits, independent sigmoids | Sum of binary cross-entropies |
| Ranking / retrieval | Similarity scores | Contrastive (InfoNCE), triplet, pairwise ranking |

!!! tip "Numerical stability"
    Compute losses from **logits**, not probabilities: frameworks fuse sigmoid with BCE (`BCEWithLogitsLoss`) and softmax with cross-entropy using the **log-sum-exp trick**, $\log\sum_j e^{z_j} = m + \log\sum_j e^{z_j - m}$ with $m = \max_j z_j$. This avoids overflow in $e^{z}$ and $\log 0$.

## Backpropagation

**The problem:** a network has millions of parameters; computing each gradient separately (for example by finite differences) would need millions of forward passes.

**Core idea:** the chain rule, applied backward from the loss through the computation graph, reusing every intermediate result. One backward pass gives the gradient for all parameters at roughly two to three times the cost of a forward pass.

### A two-layer network with shapes

Forward pass for input $X$ $(n, d)$, hidden size $h$, $k$ classes, one-hot labels $Y$ $(n, k)$:

$$
Z_1 = XW_1 + b_1,\quad A_1 = \text{ReLU}(Z_1),\quad Z_2 = A_1W_2 + b_2,\quad P = \text{softmax}(Z_2),\quad L = -\tfrac{1}{n}\textstyle\sum Y \odot \log P
$$

Backward pass (each gradient has the shape of the thing it's the gradient of):

| Gradient | Formula | Shape |
|---|---|---|
| $\partial L / \partial Z_2$ | $(P - Y)/n$ | $(n, k)$ |
| $\partial L / \partial W_2$ | $A_1^\top\, \partial Z_2$ | $(h, k)$ |
| $\partial L / \partial b_2$ | sum of $\partial Z_2$ over rows | $(k,)$ |
| $\partial L / \partial A_1$ | $\partial Z_2\, W_2^\top$ | $(n, h)$ |
| $\partial L / \partial Z_1$ | $\partial A_1 \odot \mathbb{1}[Z_1 > 0]$ | $(n, h)$ |
| $\partial L / \partial W_1$ | $X^\top\, \partial Z_1$ | $(d, h)$ |
| $\partial L / \partial b_1$ | sum of $\partial Z_1$ over rows | $(h,)$ |

The pattern to remember: for $Z = AW$, the weight gradient is $A^\top \partial Z$ and the gradient passed back is $\partial Z\, W^\top$.

### Automatic differentiation

- **Reverse mode** (backprop) computes the gradient of one scalar output with respect to all inputs in one backward pass. Ideal for training, where the loss is a scalar and parameters number in the billions.
- **Forward mode** propagates derivatives alongside values, one input direction per pass. Efficient when there are few inputs and many outputs.
- PyTorch builds the graph dynamically as code runs ("define by run") and frees it after `backward()`.
- **Memory:** the backward pass needs the activations saved during the forward pass, so activation memory grows with batch size, sequence length and depth. **Gradient checkpointing** stores only some activations and recomputes the rest during backward, trading about 30% more compute for much less memory.
- **Gradient checking:** compare analytic gradients with finite differences $\frac{f(\theta + \epsilon) - f(\theta - \epsilon)}{2\epsilon}$ on a tiny network to catch bugs.

## Activation functions

![Sigmoid, tanh, ReLU and GELU with their derivatives](../assets/img/activations.svg#only-light){ loading=lazy }
![Sigmoid, tanh, ReLU and GELU with their derivatives](../assets/img/activations-dark.svg#only-dark){ loading=lazy }

| Activation | Formula | Pros | Cons | Where used |
|---|---|---|---|---|
| **Sigmoid** | $\frac{1}{1+e^{-x}}$ | Output in (0, 1) | Saturates; derivative ≤ 0.25; not zero-centered | Binary outputs, gates |
| **Tanh** | $\tanh x$ | Zero-centered, range (−1, 1) | Saturates | RNN and LSTM states |
| **ReLU** | $\max(0, x)$ | No saturation for $x > 0$; cheap; sparse | "Dying ReLU": units stuck at 0 | CNNs, MLPs |
| **Leaky ReLU** | $\max(\alpha x, x)$, $\alpha \approx 0.01$ | Keeps a gradient for $x < 0$ | Extra hyperparameter | Fixes dying units |
| **ELU / SELU** | Smooth exponential for $x < 0$ | Smooth, mean near zero | Costlier | Some MLPs |
| **GELU** | $x\,\Phi(x)$ | Smooth, probabilistic gating | Slightly costlier | BERT, GPT |
| **Swish / SiLU** | $x\,\sigma(x)$ | Smooth, non-monotonic | Slightly costlier | EfficientNet, inside SwiGLU |
| **SwiGLU** | $(xW) \otimes \text{SiLU}(xV)$ | Gated feed-forward layers train better | A third weight matrix | LLaMA-style LLMs |
| **Softmax** | $\frac{e^{z_k}}{\sum_j e^{z_j}}$ | Probabilities over classes | Not used between hidden layers | Classification heads, attention |

## Vanishing and exploding gradients

**The problem:** deep networks and RNNs stopped learning in their early layers.

**Why:** backprop multiplies one factor per layer (weights times activation derivatives). If those factors are mostly below 1, the gradient shrinks exponentially with depth; above 1, it explodes. Sigmoid's derivative is at most 0.25, so ten sigmoid layers can shrink a gradient by a factor of about $10^{-6}$.

| Fix | How it helps |
|---|---|
| ReLU-family activations | Derivative 1 for active units |
| Xavier / He initialization | Keeps activation and gradient variance stable across layers at the start |
| Normalization (BatchNorm, LayerNorm) | Keeps activations in a healthy range during training |
| Residual connections | An identity path adds a direct route for gradients: $\frac{\partial}{\partial x}(x + F(x)) = I + F'(x)$ |
| Gated RNNs (LSTM, GRU) | Additive memory updates instead of repeated multiplication |
| Gradient clipping | Caps the gradient norm to stop explosions (standard for RNNs and Transformers) |

## Weight initialization

**Why not zeros?** Every unit in a layer would compute the same output and receive the same gradient, so they'd stay identical forever (the symmetry problem). Random initialization breaks the symmetry, and its **scale** determines whether signals vanish or explode.

| Scheme | Variance | Designed for |
|---|---|---|
| **Xavier / Glorot** (2010) | $\operatorname{Var}(W) = \frac{2}{n_{in} + n_{out}}$ | Tanh, sigmoid |
| **He / Kaiming** (2015) | $\operatorname{Var}(W) = \frac{2}{n_{in}}$ | ReLU (which zeroes half the inputs, hence the factor 2) |
| **Transformers** | Normal with std about 0.02; output projections of residual branches scaled down by depth | Stable deep residual stacks |

The idea behind both: a sum of $n_{in}$ terms has variance $n_{in}\operatorname{Var}(w)\operatorname{Var}(x)$, so choosing $\operatorname{Var}(w) \propto 1/n_{in}$ keeps the output variance close to the input variance.

## What came next, and why

Fundamentals made deep networks possible; training them well needed better **optimizers** (momentum, Adam), **normalization**, **regularization** (dropout, augmentation) and **schedules**. See [Training Deep Networks](training.md). Architectures then specialized: **CNNs** for grids ([CNNs & Computer Vision](cnn-vision.md)) and **RNNs** for sequences ([Sequence Models](sequence-models.md)).

## Interview questions

??? question "Why can't a single perceptron learn XOR?"
    A perceptron's decision boundary is a single line (hyperplane). XOR's positive points (0,1) and (1,0) sit on one diagonal and the negatives (0,0) and (1,1) on the other, so no single line separates them. One hidden layer fixes it: two hidden units can compute OR and NAND (or AND), and the output combines them, which is exactly the feature engineering a linear model can't do for itself.

??? question "Why do neural networks need non-linear activations?"
    Without them, a stack of linear layers is just one linear map, $W_n \cdots W_1 x$, so depth adds no expressive power and the network can only represent linear functions. The non-linearity between layers lets each layer bend and fold the space so later layers can separate classes that weren't linearly separable.

??? question "What does the universal approximation theorem say, and what doesn't it say?"
    One hidden layer with enough units and a suitable non-linearity can approximate any continuous function on a compact domain arbitrarily well. It doesn't say how many units are needed (possibly exponentially many), that gradient descent will find the weights, or that the network will generalize from finite data. Deep networks achieve the same with far fewer parameters by composing features.

??? question "Derive backpropagation for a two-layer network."
    With $Z_1 = XW_1 + b_1$, $A_1 = \text{ReLU}(Z_1)$, $Z_2 = A_1W_2 + b_2$ and softmax cross-entropy: $\partial Z_2 = (P - Y)/n$; $\partial W_2 = A_1^\top \partial Z_2$; $\partial b_2 = \sum_{\text{rows}} \partial Z_2$; $\partial A_1 = \partial Z_2 W_2^\top$; $\partial Z_1 = \partial A_1 \odot \mathbb{1}[Z_1 > 0]$; $\partial W_1 = X^\top \partial Z_1$; $\partial b_1 = \sum_{\text{rows}} \partial Z_1$. State shapes as you go; a shape mismatch is the fastest way to catch an error.

??? question "Why is backpropagation efficient compared with computing each gradient separately?"
    It reuses intermediate results. The gradient flowing into each layer is computed once and shared by all parameters below it, so the total cost is a small constant multiple (about 2–3×) of a forward pass, regardless of how many parameters there are. Finite differences would need one or two forward passes per parameter, which is impossible for millions or billions of parameters.

??? question "Why does backprop need to store activations, and what is gradient checkpointing?"
    The gradient of a layer's weights depends on that layer's input ($\partial W = A^\top \partial Z$) and activation derivatives depend on pre-activations, so the forward values must be kept until the backward pass reaches that layer. Memory grows with depth, batch size and sequence length. Gradient checkpointing stores activations only at some layers and recomputes the others during the backward pass, trading extra compute for a large memory saving; it's standard in LLM training.

??? question "Why does sigmoid cause vanishing gradients?"
    Its derivative $\sigma(x)(1 - \sigma(x))$ is at most 0.25 (at $x = 0$) and close to 0 when the unit saturates. Backprop multiplies one such factor per layer, so gradients shrink exponentially with depth; the early layers barely learn. It's also not zero-centered, which makes gradient updates zig-zag. Hidden layers therefore use ReLU-family activations, and sigmoid stays at outputs and inside gates.

??? question "Why ReLU over sigmoid in hidden layers?"
    ReLU's gradient is exactly 1 for positive inputs, so it doesn't shrink gradients through active units; it's very cheap to compute; and it produces sparse activations. That made deep networks much faster to train (AlexNet reported several times faster convergence than tanh). Its weakness is dying units; Leaky ReLU, ELU or GELU address that.

??? question "What is the dying ReLU problem and how do you fix it?"
    If a unit's pre-activation becomes negative for every input (for example after a large update or with a large negative bias), its output and gradient are zero forever, so it never recovers. Fixes: Leaky ReLU or PReLU (small slope for negative inputs), ELU or GELU, lower learning rates, He initialization, and normalization layers.

??? question "Why do LLMs use GELU or SwiGLU instead of ReLU?"
    Smooth activations give smoother loss surfaces and keep small gradients for negative inputs, and empirically they train Transformers to lower loss. SwiGLU, a gated linear unit using SiLU, adds a multiplicative gate inside the feed-forward layer, which performed best in Shazeer's comparison of GLU variants and was adopted by PaLM and LLaMA. To keep parameter count equal, the hidden width is usually reduced from $4d$ to about $\tfrac{8}{3}d$.

??? question "Why not initialize all weights to zero?"
    All units in a layer would receive identical inputs, compute identical outputs and get identical gradients, so they would remain identical: the layer behaves like a single unit (the symmetry problem). Random initialization breaks the symmetry. Biases can start at zero.

??? question "Xavier vs He initialization."
    Both pick the weight variance so activations and gradients keep roughly constant variance across layers. Xavier uses $\frac{2}{n_{in} + n_{out}}$, derived for symmetric activations like tanh. He uses $\frac{2}{n_{in}}$, accounting for ReLU zeroing about half of its inputs, which halves the variance. Using Xavier with deep ReLU networks makes activations shrink layer by layer.

??? question "Why subtract the maximum before computing softmax?"
    $e^{z}$ overflows for large logits (above about 88 in float32). Softmax is unchanged by subtracting a constant from every logit, so subtracting the maximum makes the largest exponent $e^0 = 1$ and all others smaller, avoiding overflow. For the loss, use log-softmax with the log-sum-exp trick so you never take the log of an underflowed zero.

??? question "How do you detect and fix exploding gradients?"
    Symptoms: loss suddenly becomes NaN or spikes, very large gradient norms, weights blowing up. Monitor the global gradient norm per step. Fixes: gradient clipping by global norm (commonly 1.0), a lower learning rate or warmup, better initialization, normalization layers, residual connections, and checking the data for extreme values or bad labels.

??? question "Reverse-mode vs forward-mode automatic differentiation."
    Reverse mode (backprop) computes the gradient of one output with respect to all inputs in one backward pass, ideal when the output is a scalar loss and the inputs are billions of parameters. Forward mode carries derivatives forward alongside values, one input direction per pass, which is efficient for functions with few inputs and many outputs. Jacobian–vector products use forward mode and vector–Jacobian products use reverse mode.

!!! pitfall "Common mistakes"
    - Saying the universal approximation theorem means one layer is enough in practice.
    - Applying softmax and then a cross-entropy that applies softmax again.
    - Computing losses from probabilities instead of logits.
    - Forgetting the factor 2 in He initialization, or why it's there.
    - Confusing vanishing gradients (early layers stop learning) with overfitting.

## Revision checklist

- [ ] Explain why XOR needs a hidden layer and why activations must be non-linear
- [ ] Count the parameters of an MLP
- [ ] Derive backprop for a two-layer network with shapes
- [ ] Compare sigmoid, tanh, ReLU, GELU and SwiGLU
- [ ] Explain vanishing and exploding gradients and five fixes
- [ ] State Xavier and He initialization and the reasoning behind them

## Go deeper

- [Karpathy: Neural Networks, Zero to Hero](https://karpathy.ai/zero-to-hero.html), starting with micrograd
- [3Blue1Brown: Neural networks](https://www.3blue1brown.com/topics/neural-networks)
- [Dive into Deep Learning](https://d2l.ai/), chapters 4 and 5
