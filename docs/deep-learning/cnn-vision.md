---
title: CNNs & Computer Vision
description: Convolutions and their arithmetic, the architecture line from LeNet to ResNet and ConvNeXt, detection and segmentation, Vision Transformers, and self-supervised and contrastive learning including CLIP.
act: 2
priority: High
tags:
  - ML Engineer
  - Applied Scientist
---

# CNNs & Computer Vision

!!! abstract "TL;DR"
    **Convolutions** slide small shared filters over an image, giving translation equivariance with few parameters. **AlexNet** (2012) proved deep CNNs on GPUs beat hand-made features; **ResNet** (2015) made very deep networks trainable with skip connections. Detection (Faster R-CNN, YOLO) and segmentation (U-Net, Mask R-CNN) build on CNN backbones. The **Vision Transformer** treats image patches as tokens and scales better with data. **Contrastive learning** (SimCLR, CLIP) learns from unlabeled images or image–caption pairs, and CLIP-style encoders now feed multimodal LLMs.

!!! evolution "Where this fits"
    **Before:** [MLPs and backprop](neural-networks.md), [training techniques](training.md). **This page:** the first domain where deep learning won decisively. **Next:** [Sequence Models](sequence-models.md) for text and time; the [Transformer](../llms/transformer.md) later unifies vision and language.

## Convolutions

**The problem:** an MLP on a 224×224×3 image needs about 150,000 weights per hidden unit and has no idea that a cat in the top-left corner is the same as a cat in the bottom-right.

**Core idea:** use small filters (for example 3×3) whose weights are **shared** across all positions. Each filter detects one pattern wherever it appears.

| Concept | Meaning |
|---|---|
| **Kernel / filter** | A $k \times k \times C_{in}$ weight tensor; a layer has $C_{out}$ of them |
| **Stride** | Step between filter positions; stride 2 halves the resolution |
| **Padding** | Zeros around the border; "same" padding keeps the size |
| **Output size** | $\left\lfloor \frac{W - k + 2p}{s} \right\rfloor + 1$ |
| **Parameters** | $(k \cdot k \cdot C_{in} + 1) \cdot C_{out}$, independent of image size |
| **Receptive field** | The input region that affects one output; grows with depth, stride and pooling |
| **Pooling** | Max or average over small windows; downsamples and adds small translation invariance |
| **1×1 convolution** | Mixes channels at each position; cheap way to change channel count |
| **Depthwise separable** | A per-channel spatial filter, then a 1×1 to mix channels; about $k^2$ times cheaper (MobileNet) |
| **Dilated (atrous)** | Gaps between kernel taps; bigger receptive field without more parameters (DeepLab) |
| **Transposed convolution** | Learned upsampling (decoders, segmentation) |

??? example "Worked example: shapes and parameters"
    A 3×3 convolution from 64 to 128 channels, stride 1, padding 1, on a 56×56 input: output is 56×56×128 and parameters are $3\cdot3\cdot64\cdot128 + 128 = 73{,}856$. With stride 2 the output becomes 28×28×128 and the parameter count doesn't change.

    Two stacked 3×3 layers see a 5×5 region with $2 \cdot 9C^2 = 18C^2$ weights instead of $25C^2$ for one 5×5 layer, plus an extra non-linearity. That's why VGG used only 3×3 filters.

**Inductive biases:** locality (nearby pixels matter together), weight sharing (the same pattern can appear anywhere) and a hierarchy of features. Convolution is **translation-equivariant** (shift the input, the feature map shifts the same way); pooling and global averaging add approximate **invariance**. These biases make CNNs data-efficient, and they're exactly what ViTs give up in exchange for flexibility.

## The architecture line

<div class="metro act-2" markdown>

- **LeNet-5** <span class="yr">1998</span><br>Convolution, pooling, then dense layers for digits. Small images, little compute.
- **AlexNet** <span class="yr">2012</span><br>8 layers, ReLU, dropout, data augmentation, two GPUs. ImageNet top-5 error 15.3% vs about 26% for the runner-up.
- **VGG** <span class="yr">2014</span><br>Uniform stacks of 3×3 convolutions, 16–19 layers. Simple but heavy (about 138M parameters).
- **GoogLeNet / Inception** <span class="yr">2014</span><br>Parallel 1×1, 3×3, 5×5 branches with 1×1 bottlenecks: wide and efficient.
- **ResNet** <span class="yr">2015</span><br>Residual blocks $y = F(x) + x$; 152 layers; 3.57% top-5 error. Deeper finally meant better.
- **DenseNet** <span class="yr">2016</span><br>Each layer receives all previous feature maps; strong reuse of features.
- **MobileNet · EfficientNet** <span class="yr">2017–2019</span><br>Depthwise separable convolutions; compound scaling of depth, width and resolution.
- **ConvNeXt** <span class="yr">2022</span><br>A CNN modernized with Transformer-era choices (large kernels, LayerNorm, GELU) that matches ViTs.

</div>

### Why ResNet works

**The degradation problem:** adding layers to a plain deep network made *training* error worse, not just test error. That isn't overfitting; it's an optimization failure. A deeper network could in principle copy the shallower one and set the extra layers to identity, but plain layers find identity hard to learn.

A residual block learns only the change: $y = F(x) + x$. If identity is best, the block pushes $F$ toward zero, which is easy. The skip connection also gives gradients a direct path: $\frac{\partial y}{\partial x} = I + \frac{\partial F}{\partial x}$, so signals reach early layers even when $F$'s gradient is small. **Bottleneck blocks** (1×1 reduce → 3×3 → 1×1 expand) keep very deep ResNets affordable. Every Transformer block uses the same residual idea.

### Transfer learning in vision

Use a backbone pretrained on a large dataset (ImageNet, or a self-supervised or CLIP model), then:

- **Small data, similar domain:** freeze the backbone, train a new head (linear probing).
- **More data or a different domain:** fine-tune the whole network with a lower learning rate for early layers.
- Match the pretraining preprocessing (resolution, normalization) and use augmentation.

## Object detection

**The problem:** classification says *what*; detection also says *where*, as boxes with classes, often many per image.

| Family | Examples | Idea | Trade-off |
|---|---|---|---|
| **Two-stage** | R-CNN (2014) → Fast R-CNN → Faster R-CNN (2015) | Propose regions (a region proposal network), then classify and refine each | Accurate, slower |
| **One-stage** | YOLO (2015 onward), SSD, RetinaNet (2017) | Predict classes and boxes densely in one pass | Real-time; RetinaNet's focal loss closed the accuracy gap |
| **Anchor-free** | FCOS, CenterNet | Predict box centers or distances directly, no predefined anchor boxes | Simpler, fewer hyperparameters |
| **Transformer-based** | DETR (2020) | Learned object queries plus set matching (Hungarian) to ground truth; no NMS | Elegant; slower to train originally |

Key pieces:

- **IoU** (intersection over union) measures box overlap: $\frac{|A \cap B|}{|A \cup B|}$.
- **Non-max suppression** removes duplicate boxes: keep the highest-scoring box, drop others overlapping it above an IoU threshold, repeat.
- **mAP:** average precision per class from the precision–recall curve at an IoU threshold (for example 0.5, or averaged over 0.5–0.95 in COCO), then averaged over classes.
- **Feature Pyramid Networks** combine coarse, semantic features with fine, high-resolution ones to detect objects at many scales.
- **Focal loss:** $FL(p_t) = -(1 - p_t)^\gamma \log p_t$ down-weights the flood of easy background examples so training focuses on hard ones.

## Segmentation

| Task | Output | Examples |
|---|---|---|
| **Semantic** | A class per pixel | FCN, U-Net (2015), DeepLab |
| **Instance** | A mask per object instance | Mask R-CNN (2017) |
| **Panoptic** | Both: every pixel gets a class and, for objects, an instance | Panoptic FPN, Mask2Former |
| **Promptable** | A mask for whatever you point at or describe | Segment Anything (2023) |

**U-Net** uses an encoder–decoder with **skip connections** from each encoder level to the matching decoder level, so fine spatial detail lost during downsampling is restored when upsampling. It's still the standard in medical imaging and is the backbone of many diffusion models. Metrics: mean IoU per class and the **Dice coefficient** $\frac{2|A \cap B|}{|A| + |B|}$.

## The Vision Transformer

**The problem:** CNNs' locality is helpful with limited data, but it limits global context and seemed to scale less well than Transformers did in language.

**Core idea:** split the image into patches (for example 16×16 pixels, so a 224×224 image becomes 196 tokens), linearly embed each patch, add position embeddings and a [CLS] token, and run a standard Transformer encoder.

- Weak built-in assumptions, so ViT needs **lots of data**: it beat CNNs only when pretrained on very large datasets (JFT-300M). **DeiT** showed strong augmentation and distillation make it competitive on ImageNet alone.
- Attention cost grows with the square of the number of patches, a problem at high resolution. **Swin Transformer** uses shifted local windows and a hierarchy, like a CNN, to fix this.
- One architecture for text and images made multimodal models natural.

| | CNN | ViT |
|---|---|---|
| Built-in assumptions | Locality, translation equivariance, hierarchy | Almost none; learns them from data |
| Data efficiency | High | Low without huge pretraining or heavy augmentation |
| Global context | Grows slowly with depth | From the first layer |
| Scaling with data and compute | Good | Better at large scale |
| High-resolution cost | Linear in pixels | Quadratic in patches (unless windowed) |

## Self-supervised and contrastive learning

**The problem:** labels are expensive, but unlabeled images and image–caption pairs are abundant.

| Method | Year | Idea |
|---|---|---|
| **SimCLR** | 2020 | Two augmentations of the same image are positives; other images in the batch are negatives; InfoNCE loss. Needs large batches. |
| **MoCo** | 2019 | A momentum-updated encoder and a queue of negatives, so batches can be small |
| **BYOL, DINO** | 2020, 2021 | No negatives: a student matches a slowly updated teacher's output on another view |
| **MAE** | 2021 | Mask 75% of patches and reconstruct them; very efficient for ViTs |
| **CLIP** | 2021 | Image and text encoders trained contrastively on 400M image–caption pairs |

**The InfoNCE loss** for a positive pair $(q, k^+)$ among negatives $k_j$: $L = -\log\frac{\exp(q \cdot k^+/\tau)}{\sum_j \exp(q \cdot k_j/\tau)}$, with temperature $\tau$. It's a softmax classification of "which of these is my partner", which is why more negatives (bigger batches) help.

**CLIP** trains with a symmetric version of that loss over a batch of N image–caption pairs: the N matching pairs should score highest among all N² combinations. **Zero-shot classification:** embed prompts like "a photo of a {label}" for every class and pick the class whose text embedding is closest to the image embedding. CLIP's zero-shot ImageNet accuracy matched a fully supervised ResNet-50. Its text encoder conditions text-to-image diffusion models, and CLIP or SigLIP vision encoders feed most [multimodal LLMs](../llms/generative-models.md).

## Interview questions

??? question "Compute the output size and parameter count of a convolutional layer."
    Output width $= \lfloor (W - k + 2p)/s \rfloor + 1$ (same for height); output depth is the number of filters $C_{out}$. Parameters $= (k \cdot k \cdot C_{in} + 1)\cdot C_{out}$ with biases. Example: 224×224×3 input, 64 filters of 7×7, stride 2, padding 3 → 112×112×64 output with $7\cdot7\cdot3\cdot64 + 64 = 9{,}472$ parameters. The count doesn't depend on image size; the compute does.

??? question "Why share weights in a CNN?"
    It encodes the assumption that a useful pattern (an edge, an eye) can appear anywhere, so the same detector should be applied everywhere. That cuts parameters by orders of magnitude, improves data efficiency because every position provides training signal for the same filter, and makes the layer translation-equivariant.

??? question "What does a 1×1 convolution do?"
    At each spatial position it computes a linear combination of the input channels, like a dense layer applied per pixel. It changes the number of channels cheaply (bottlenecks in Inception and ResNet), mixes information across channels, and adds a non-linearity when followed by an activation. In depthwise separable convolutions it's the "pointwise" step.

??? question "Equivariance vs invariance."
    A function is **equivariant** to translation if shifting the input shifts the output the same way: convolution feature maps move with the object. It's **invariant** if shifting the input leaves the output unchanged: a classifier's "cat" score. CNNs are equivariant through convolutions and become approximately invariant through pooling, striding and global average pooling at the end.

??? question "What is a receptive field and how do you grow it?"
    The region of the input that can influence a given output unit. It grows with depth (each 3×3 layer adds 2 pixels), with stride and pooling (which multiply the growth of later layers), and with dilated convolutions. Large receptive fields are needed for context, such as recognizing large objects or segmenting with scene context. The effective receptive field is usually smaller than the theoretical one and concentrated near the center.

??? question "Why did AlexNet matter so much?"
    In 2012 it cut ImageNet top-5 error from about 26% to 15.3%, a far larger jump than incremental improvements to hand-crafted features. It combined a deep CNN, ReLU (much faster training than tanh), dropout, data augmentation, and training on GPUs with a large labeled dataset. It convinced the field that learned features beat engineered ones given enough data and compute, which started the deep-learning era.

??? question "Explain the degradation problem and how ResNet solves it."
    Plain networks deeper than about 20 layers had higher *training* error than shallower ones, an optimization problem, not overfitting. ResNet adds identity skip connections so each block learns a residual $F(x)$ and outputs $F(x) + x$. Learning identity becomes trivial (drive $F$ toward zero), and gradients flow directly through the skip paths ($\partial y/\partial x = I + \partial F/\partial x$). This made 100- to 1000-layer networks trainable.

??? question "Why use two 3×3 convolutions instead of one 5×5?"
    Same receptive field (5×5), fewer parameters ($18C^2$ vs $25C^2$), and an extra non-linearity between them, which makes the function more expressive. Three 3×3 layers similarly replace a 7×7 ($27C^2$ vs $49C^2$).

??? question "How much does a depthwise separable convolution save?"
    A standard convolution costs about $k^2 C_{in} C_{out}$ multiply-adds per position. Depthwise separable splits it into a per-channel $k \times k$ filter ($k^2 C_{in}$) plus a 1×1 pointwise convolution ($C_{in} C_{out}$). The ratio is $\frac{1}{C_{out}} + \frac{1}{k^2}$, so about 8–9 times cheaper for 3×3 kernels, with a small accuracy cost. It's the core of MobileNet and EfficientNet.

??? question "One-stage vs two-stage object detectors."
    Two-stage detectors (Faster R-CNN) first propose candidate regions, then classify and refine each one: more accurate, especially on small objects, but slower. One-stage detectors (YOLO, SSD, RetinaNet) predict boxes and classes densely over the image in a single pass: fast enough for real time. RetinaNet's focal loss addressed the extreme foreground–background imbalance that had held one-stage accuracy back.

??? question "Explain IoU, non-max suppression and mAP."
    **IoU** is the overlap between two boxes divided by their union; a detection usually counts as correct if its IoU with a ground-truth box exceeds 0.5. **Non-max suppression** keeps the highest-scoring box and removes boxes that overlap it above an IoU threshold, repeating until none remain. **mAP** computes average precision (the area under the precision–recall curve) per class and averages over classes; COCO averages it over IoU thresholds from 0.5 to 0.95 too.

??? question "Why does U-Net use skip connections?"
    The encoder downsamples to capture context but loses fine spatial detail. Skip connections pass high-resolution feature maps from each encoder level directly to the matching decoder level, where they're concatenated with the upsampled features. The decoder can then produce precise boundaries using both "what" (deep features) and "where" (shallow features). It works well even with small datasets, which is why it's standard in medical imaging.

??? question "Why does focal loss help RetinaNet?"
    Dense detectors evaluate tens of thousands of candidate locations, nearly all easy background. With cross-entropy, these easy negatives dominate the total loss and gradients. Focal loss multiplies cross-entropy by $(1 - p_t)^\gamma$ (often $\gamma = 2$), shrinking the loss for well-classified examples so training concentrates on hard, misclassified ones. Combined with class-balancing weights, it let one-stage detectors match two-stage accuracy.

??? question "How does ViT turn an image into tokens, and why does it need so much data?"
    It cuts the image into fixed-size patches (for example 16×16), flattens each and projects it linearly to the model dimension, adds learned position embeddings, prepends a [CLS] token, and feeds the sequence to a Transformer encoder. Unlike a CNN, it has no built-in locality or translation equivariance, so it must learn those regularities from data. With limited data CNNs win; with very large pretraining datasets (or strong augmentation and distillation) ViT matches or beats them.

??? question "ViT vs CNN: which would you choose?"
    For small or medium datasets, edge devices or high-resolution dense prediction with tight compute, a CNN (or a hybrid) is often the better choice. With large pretraining data or a strong pretrained checkpoint, ViTs scale better and integrate naturally with language models. In practice you usually fine-tune a pretrained backbone, and modern CNNs (ConvNeXt) and ViTs perform similarly; choose based on available checkpoints, latency and deployment tooling.

??? question "Explain the contrastive loss used by SimCLR and CLIP."
    InfoNCE treats each example as a classification problem: among one positive and many negatives, pick the positive. $L = -\log\frac{\exp(\text{sim}(q, k^+)/\tau)}{\sum_j \exp(\text{sim}(q, k_j)/\tau)}$ with cosine similarity and temperature $\tau$. SimCLR's positives are two augmented views of one image; CLIP's are an image and its caption, with the loss applied symmetrically in both directions. More negatives give a tighter bound and better representations, hence large batches.

??? question "How does CLIP do zero-shot classification?"
    Embed each class name in a prompt such as "a photo of a {class}" with the text encoder, embed the image with the image encoder, and predict the class whose text embedding has the highest cosine similarity with the image embedding. No task-specific training is needed, and prompt wording and ensembling prompts change accuracy. Limits: it inherits web-data biases and struggles with counting, fine-grained distinctions and spatial relations.

!!! pitfall "Common mistakes"
    - Forgetting that the parameter count of a convolution doesn't depend on input size.
    - Saying ResNet solved overfitting (it solved an optimization problem).
    - Confusing semantic and instance segmentation.
    - Claiming ViTs are always better than CNNs.
    - Mismatching preprocessing between pretraining and fine-tuning.

## Revision checklist

- [ ] Compute conv output sizes and parameter counts
- [ ] Explain receptive fields, 1×1 and depthwise separable convolutions
- [ ] Walk the line from LeNet to ResNet and explain the degradation problem
- [ ] Explain IoU, NMS, mAP and focal loss
- [ ] Explain U-Net's skip connections
- [ ] Describe ViT's patching and its data needs
- [ ] Explain InfoNCE and CLIP zero-shot classification

## Go deeper

- [Stanford CS231n notes](https://cs231n.github.io/)
- [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
- [An Image is Worth 16x16 Words (ViT, 2020)](https://arxiv.org/abs/2010.11929)
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP, 2021)](https://arxiv.org/abs/2103.00020)
