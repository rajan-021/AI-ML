---
title: Training Deep Networks
description: The training loop, optimizers from SGD to AdamW, learning-rate schedules and warmup, batch size, BatchNorm vs LayerNorm vs RMSNorm, regularization, and how to debug training.
act: 2
priority: Critical
tags:
  - ML Engineer
  - Applied Scientist
---

# Training Deep Networks

!!! abstract "TL;DR"
    Training is a loop: forward pass, loss, backward pass, optimizer step. **Adam/AdamW** adapt the step size per parameter and are the default for Transformers; **SGD with momentum** is still strong for CNNs. The **learning rate** matters most: use **warmup** then decay (often cosine). **Normalization** (BatchNorm for CNNs, LayerNorm or RMSNorm for Transformers) stabilizes training. **Regularization** (weight decay, dropout, augmentation, early stopping) controls overfitting. To debug, overfit one batch first and check the loss at initialization.

!!! evolution "Where this fits"
    **Before:** [Neural Network Fundamentals](neural-networks.md): layers, backprop, activations, initialization. **This page:** making optimization fast and stable. **Next:** architectures for images ([CNNs](cnn-vision.md)) and sequences ([Sequence Models](sequence-models.md)).

## The training loop

```python
model.train()
for epoch in range(num_epochs):
    for x, y in train_loader:
        logits = model(x)                 # forward
        loss = loss_fn(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                   # backprop
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                  # update
        scheduler.step()                  # per-step learning-rate schedule
    model.eval()                          # dropout off, BatchNorm uses running stats
    with torch.no_grad():
        val_loss = evaluate(model, val_loader)
    model.train()
```

Things interviewers check: zeroing gradients each step (PyTorch accumulates them), `model.eval()` and `torch.no_grad()` for evaluation, where the scheduler steps, and gradient clipping for Transformers and RNNs.

## Optimizers

![Paths of SGD, SGD with momentum and Adam on a long, narrow valley](../assets/img/optimizer-paths.svg#only-light){ loading=lazy }
![Paths of SGD, SGD with momentum and Adam on a long, narrow valley](../assets/img/optimizer-paths-dark.svg#only-dark){ loading=lazy }

On a long, narrow valley, plain SGD zig-zags across the steep direction while crawling along the flat one. Momentum builds speed along the consistent direction. Adam rescales each coordinate by its gradient size, so it moves in both directions at similar speed.

| Optimizer | Update | What it fixed | Weakness |
|---|---|---|---|
| **SGD** | $\theta \leftarrow \theta - \eta g$ | Full-batch gradient descent is too slow per step | Noisy; one learning rate for every parameter; slow in ravines |
| **Momentum** (1964) | $v \leftarrow \beta v + g$; $\theta \leftarrow \theta - \eta v$ | Damps oscillation, speeds up consistent directions | Can overshoot; still one global rate |
| **Nesterov** | Gradient evaluated at the look-ahead point | Slightly better-behaved momentum | Marginal gains |
| **AdaGrad** (2011) | Divide by $\sqrt{\sum g^2}$ | Per-parameter rates; great for sparse features | Rate decays toward zero on long runs |
| **RMSProp** (2012) | Divide by $\sqrt{\text{EMA}(g^2)}$ | Fixes AdaGrad's decay | No momentum term |
| **Adam** (2014) | Momentum + RMSProp + bias correction | Robust default with little tuning | L2 penalty interacts badly with adaptive scaling |
| **AdamW** (2017) | Adam with **decoupled** weight decay | Proper regularization; standard for Transformers and LLMs | Two extra states per parameter (memory) |

### Adam step by step

$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t &&\text{(average gradient: momentum)}\\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 &&\text{(average squared gradient: scale)}\\
\hat m_t &= \frac{m_t}{1 - \beta_1^t}, \quad \hat v_t = \frac{v_t}{1 - \beta_2^t} &&\text{(bias correction)}\\
\theta_t &= \theta_{t-1} - \eta\left(\frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda\,\theta_{t-1}\right) &&\text{(AdamW's decoupled weight decay)}
\end{aligned}
$$

Typical defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$ (often 0.95 for LLMs), $\epsilon = 10^{-8}$, weight decay around 0.01–0.1.

- **Why bias correction?** $m$ and $v$ start at zero, so early averages are biased toward zero; dividing by $1 - \beta^t$ fixes that for the first steps.
- **Why AdamW?** With plain Adam, an L2 term added to the loss gets divided by $\sqrt{\hat v}$ like any gradient, so parameters with large gradients are barely regularized. Decoupling applies the same proportional shrinkage to every weight.
- **Memory:** Adam keeps $m$ and $v$ for every parameter, so optimizer state is twice the model size in the state's precision. That matters for LLMs; see [memory math](../llms/fine-tuning.md).
- **Other optimizers you may hear about:** LARS/LAMB for very large batches, Adafactor (factorized second moments to save memory), Lion, and newer orthogonalized-update methods such as Muon. AdamW remains the common baseline.

## Learning rate, schedules and batch size

The learning rate is the most important hyperparameter. Too high: loss diverges or spikes. Too low: slow training, or stuck in a poor region.

- **Learning-rate range test:** increase the rate exponentially over a few hundred steps and pick a value a bit below where the loss starts rising.
- **Warmup:** start near zero and increase linearly over the first few hundred or thousand steps. Early on, Adam's second-moment estimates are unreliable and the weights are random, so large steps can destabilize training, especially for Transformers.
- **Decay:** step decay, exponential, **cosine** (smooth decay to a small floor), **one-cycle** (up then down), or **warmup–stable–decay** (hold a constant rate, then decay quickly at the end; convenient when you don't know the training length in advance).

![Learning rate rising linearly during 2,000 warmup steps then decaying along a cosine curve to 10% of its peak](../assets/img/lr-schedule.svg#only-light){ loading=lazy }
![Learning rate rising linearly during 2,000 warmup steps then decaying along a cosine curve to 10% of its peak](../assets/img/lr-schedule-dark.svg#only-dark){ loading=lazy }

**Batch size.** Larger batches give less noisy gradients and better hardware use, but each example contributes less learning per step. The **linear scaling rule** (multiply the learning rate by $k$ when the batch grows by $k$, with warmup) works for SGD up to a point; adaptive optimizers often scale closer to $\sqrt{k}$. Very large batches can generalize slightly worse without retuning. If memory is the limit, use **gradient accumulation**: sum gradients over several micro-batches before one optimizer step.

## Normalization layers

**The problem:** as weights change, the distribution of each layer's inputs shifts, forcing small learning rates and careful initialization.

**Core idea:** normalize activations to zero mean and unit variance, then learn a scale $\gamma$ and shift $\beta$ so the layer can still represent what it needs: $y = \gamma \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$.

| | Statistics computed over | Train vs inference | Best for |
|---|---|---|---|
| **BatchNorm** (2015) | The batch (and spatial positions), per channel | Training: batch stats. Inference: running averages | CNNs with reasonable batch sizes |
| **LayerNorm** (2016) | The features of each token or sample | Identical | Transformers, RNNs, any batch size |
| **GroupNorm** | Groups of channels per sample | Identical | CNNs with tiny batches (detection, segmentation) |
| **RMSNorm** (2019) | Root mean square of features; no mean subtraction | Identical | LLMs (LLaMA and most successors): cheaper, as good |

**Why it helps:** it smooths the loss landscape and keeps activations in a range where gradients flow, allowing higher learning rates; BatchNorm also adds a little regularizing noise.

**BatchNorm's problems:** it depends on batch size (noisy statistics with small batches), behaves differently in training and inference (a classic bug is forgetting `model.eval()`), and is awkward for variable-length sequences and autoregressive generation. Hence LayerNorm for Transformers.

**Pre-LN vs Post-LN.** The original Transformer normalized after adding the residual (Post-LN), which needed careful warmup to train deep stacks. **Pre-LN** normalizes the input of each sublayer, $x + F(\text{Norm}(x))$, keeping a clean identity path for gradients; GPT-2 onward uses it.

## Regularization in deep learning

| Technique | How | Notes |
|---|---|---|
| **Weight decay** | Shrink weights each step (AdamW) | The default regularizer |
| **Dropout** (2014) | Zero each activation with probability $p$ during training, scale survivors by $1/(1-p)$ | Acts like an ensemble of sub-networks; turned off at inference. Little or no dropout in LLM pretraining, where data is huge and seen about once. |
| **Data augmentation** | Images: crops, flips, color jitter, RandAugment. Text: back-translation, token masking. Audio: SpecAugment. | Often the biggest win for vision |
| **Mixup / CutMix** | Blend two images and their labels, or paste a patch from one into another | Smoother decision boundaries; better calibration |
| **Label smoothing** | Replace one-hot targets with $1 - \epsilon$ for the true class and spread $\epsilon$ over the rest | Reduces over-confidence |
| **Early stopping** | Stop when validation loss stops improving; keep the best checkpoint | Simple and effective |
| **Stochastic depth** | Randomly skip residual blocks during training | Very deep networks, ViTs |

## Debugging training

A reliable order of checks:

1. **Check the loss at initialization.** With $K$ balanced classes and a fresh model it should be close to $\ln K$ (2.30 for 10 classes; about 10.8 for a 50,000-token vocabulary). Far above that means bad initialization or a bug.
2. **Overfit one small batch.** A correct model and pipeline should drive training loss near zero on 10–100 examples. If it can't, the bug is in the model, loss or labels, not in regularization.
3. **Visualize the data right before it enters the model**, after augmentation and batching: label alignment, normalization, padding and masks.
4. **Start with a known-good setup** (AdamW, a standard learning rate, no fancy tricks), then add one change at a time.
5. **Watch gradient norms, activation statistics and the learning rate** per step, not just the loss.

| Symptom | Likely causes |
|---|---|
| Loss is NaN or explodes | Learning rate too high, no warmup, no gradient clipping, log of zero, division by zero, bad data, FP16 overflow |
| Loss doesn't decrease at all | Learning rate too low or too high, frozen parameters, wrong labels, a bug in the loss, gradients not flowing (detached tensor) |
| Training loss falls, validation doesn't | Overfitting, leakage between splits in reverse (validation from another distribution), augmentation only at validation |
| Validation better than training | Dropout and augmentation active only during training; that's normal early on |
| Loss plateaus early | Learning rate schedule decaying too fast, dead units, poor initialization |
| Big gap between `train()` and `eval()` results | BatchNorm running statistics, dropout, different preprocessing |

## Interview questions

??? question "Explain Adam step by step."
    Adam keeps two exponential moving averages per parameter: $m$ of the gradient (momentum, direction) and $v$ of the squared gradient (scale). Both start at zero, so it divides them by $1 - \beta_1^t$ and $1 - \beta_2^t$ to correct the bias in early steps. The update is $\eta\,\hat m / (\sqrt{\hat v} + \epsilon)$: parameters with consistently large gradients take relatively smaller steps, and rarely updated ones take larger steps. Defaults are $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

??? question "Why AdamW instead of Adam with L2 regularization?"
    In Adam, an L2 penalty's gradient $\lambda\theta$ is added to the loss gradient and then divided by $\sqrt{\hat v}$, so weights with large gradient history are regularized less, which isn't what weight decay is supposed to do. AdamW decouples it: it applies $\theta \leftarrow \theta - \eta\lambda\theta$ directly, independent of the adaptive scaling. This generalizes better and makes the learning rate and decay strength easier to tune separately.

??? question "Momentum: what problem does it solve?"
    In a long, narrow valley, gradients point mostly across the valley, so SGD bounces between the walls while making slow progress along the floor. Momentum keeps a running average of gradients: oscillating components cancel and the consistent component accumulates, so the optimizer speeds up along the valley and damps the zig-zag. It also helps pass through small bumps and plateaus.

??? question "Why use learning-rate warmup for Transformers?"
    At the start, weights are random and Adam's second-moment estimate $v$ is based on very few gradients, so the effective per-parameter step sizes can be huge and unstable. Large early updates, combined with LayerNorm and attention softmax, can push training into divergence or a bad region. Ramping the learning rate up over the first steps lets the statistics stabilize. Pre-LN architectures need less warmup than Post-LN, but warmup is still standard.

??? question "What is the relationship between batch size and learning rate?"
    Bigger batches give lower-variance gradient estimates, so you can take larger steps. For SGD, the linear scaling rule (scale the rate with the batch, plus warmup) works up to a critical batch size; beyond it, returns diminish. Adaptive optimizers often scale closer to the square root. Very large batches may need retuning (LARS/LAMB) to avoid a generalization gap. When you change one, retune the other.

??? question "How much memory does Adam need for a 7B-parameter model?"
    In mixed-precision training with fp32 optimizer states: BF16 weights (2 bytes) + BF16 gradients (2) + an fp32 master copy of weights (4) + Adam's $m$ (4) and $v$ (4) = about 16 bytes per parameter, so roughly 112 GB for 7B parameters, before activations. That's why full fine-tuning needs multiple GPUs or sharding (ZeRO/FSDP), and why LoRA and 8-bit optimizers are popular.

??? question "BatchNorm vs LayerNorm: why do Transformers use LayerNorm?"
    BatchNorm normalizes each feature across the batch, so its output depends on other examples in the batch, it needs running statistics at inference, and it breaks down with small batches and variable-length sequences. LayerNorm normalizes across the features of each token independently: no batch dependence, identical behavior in training and inference, and it works for autoregressive generation one token at a time. Many LLMs use RMSNorm, a cheaper LayerNorm without mean subtraction.

??? question "How does BatchNorm behave at inference time?"
    During training it normalizes with the current batch's mean and variance and updates exponential moving averages of them. At inference (`model.eval()`) it uses those running averages, so outputs don't depend on the batch. Forgetting to switch to eval mode, or training with tiny batches so the running statistics are poor, causes a training–inference mismatch.

??? question "Pre-LN vs Post-LN Transformers."
    Post-LN applies normalization after the residual addition: $\text{Norm}(x + F(x))$. Gradients near the output are large and those to early layers can be unstable, so deep Post-LN models need careful warmup. Pre-LN normalizes the sublayer input: $x + F(\text{Norm}(x))$, leaving an un-normalized identity path from input to output, which keeps gradients well-behaved and trains deep stacks more reliably. Most modern LLMs use Pre-LN (often with RMSNorm), plus a final normalization before the output layer.

??? question "How does dropout work at training vs test time?"
    During training, each unit is zeroed with probability $p$ and the survivors are scaled by $1/(1-p)$ (inverted dropout), so expected activations stay the same. At test time dropout is disabled and no scaling is needed. It prevents co-adaptation of units and approximates averaging an ensemble of thinned networks. Monte Carlo dropout keeps it on at inference to estimate uncertainty.

??? question "Why do large LLMs use little or no dropout?"
    Pretraining datasets are so large that the model sees most tokens about once, so classic overfitting is rarely the bottleneck; underfitting is. Dropout also slows convergence and wastes compute at that scale. Fine-tuning on small datasets may still use dropout, along with weight decay and early stopping.

??? question "What is label smoothing and why does it help?"
    Instead of training toward a one-hot target, you train toward $1 - \epsilon$ for the correct class and $\epsilon/(K-1)$ for the others. It discourages logits from growing without bound, reduces over-confidence, improves calibration and sometimes accuracy. Downsides: it can hurt knowledge distillation (smoothed teachers carry less information) and it slightly changes the meaning of predicted probabilities.

??? question "Your training loss is NaN. How do you debug it?"
    Find the first step where it happens and log the loss, gradient norms and activations around it. Common causes: learning rate too high or no warmup, no gradient clipping, FP16 overflow (use BF16 or loss scaling), log of zero or division by zero in a custom loss, exploding attention logits, and bad input values (NaN or infinities in data). Fix by lowering the rate, adding warmup and clipping, using numerically stable functions (log-softmax, BCE with logits), and validating the data pipeline.

??? question "How do you know your model can learn at all?"
    Check that the initial loss is near $\ln K$ for $K$ classes, then try to overfit a tiny batch (10–100 examples) to near-zero training loss with regularization off. If it can't memorize a few examples, there's a bug: labels misaligned, the loss applied to the wrong tensor, gradients not flowing, or a learning rate that's badly off. Only after that should you tune regularization for generalization.

??? question "SGD vs Adam: when would you choose SGD?"
    Adam converges faster with less tuning and is the default for Transformers, sparse features and NLP. Well-tuned SGD with momentum and a good schedule often matches or slightly beats Adam on image classification with CNNs, sometimes generalizing better, and uses less memory (one state per parameter instead of two). Choose SGD for CNN vision baselines with established recipes or tight memory; Adam/AdamW otherwise.

??? question "What is gradient accumulation?"
    When a desired batch doesn't fit in memory, run several smaller micro-batches, sum (or average) their gradients without stepping the optimizer, then take one step. Mathematically it equals training with the larger batch, except for batch-dependent layers like BatchNorm, which still see the small batches. It trades time for memory.

!!! pitfall "Common mistakes"
    - Forgetting `optimizer.zero_grad()` or `model.eval()`.
    - Adding L2 to the loss with Adam and calling it weight decay.
    - Changing batch size without retuning the learning rate.
    - Applying weight decay to biases and normalization parameters (usually excluded).
    - Tuning regularization before verifying the model can overfit a small batch.

## Revision checklist

- [ ] Write the training loop and the evaluation-mode details
- [ ] Write Adam's update with bias correction, and explain AdamW
- [ ] Explain warmup and cosine decay and why Transformers need warmup
- [ ] Compare BatchNorm, LayerNorm, GroupNorm and RMSNorm; explain Pre-LN
- [ ] Explain dropout's train/test behavior and label smoothing
- [ ] Debug a NaN loss and a loss that won't decrease

## Go deeper

- [Adam: A Method for Stochastic Optimization (2014)](https://arxiv.org/abs/1412.6980) and [Decoupled Weight Decay Regularization (AdamW, 2017)](https://arxiv.org/abs/1711.05101)
- [Batch Normalization (2015)](https://arxiv.org/abs/1502.03167) and [Layer Normalization (2016)](https://arxiv.org/abs/1607.06450)
- [Andrej Karpathy: A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/)
- [Google's Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook)
