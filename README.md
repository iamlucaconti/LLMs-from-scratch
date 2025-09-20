# LLMs-from-scratch

This repository contains an implementation of a **Large Language Model (LLM)** developed entirely *from scratch*.
The goals are to understand the theoretical principles underlying modern language architectures and to gain practical experience by writing the code that allows building and training a functioning LLM.

> The project follows step by step the book [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by *Sebastian Raschka (2024)*, enriching the concepts with mathematical explanations, technical notes, and didactic experiments. You can find the original repository [here](https://github.com/rasbt/LLMs-from-scratch).

Specifically, it covers:

* the **theoretical foundations** of the transformer (self-attention, multi-head attention, feed-forward networks, normalization, regularization),
* the construction of a **causal transformer** capable of generating text autoregressively,
* different **sampling strategies** for generation (greedy decoding, temperature scaling, top-k sampling),
* practical aspects of **training and evaluating** the model.


# Theoretical background 

## Transformer block

In transformers, the **multi-head attention (MHA)** layer is always used inside a larger structure called the *transformer block*. Originally, this was composed of a MHA layer, two layer normalization operations, two residual connections, and a so-called position-wise network.  
The intermediate MLP is typically designed as a 2-layer MLP, with hidden dimension an integer
multiple of the input dimension (e.g., 3x, 4x; here we use 4x), and no biases.

---
### Self-Attention

Given $n$ input vectors $\{\mathbf{x}_i\}$ stacked into a matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$, where $d$ is the input dimension. The query, key, and value matrices are then defined as

$$
\mathbf{Q} = \mathbf{X}\mathbf{W}_q, 
\quad 
\mathbf{K} = \mathbf{X}\mathbf{W}_k, 
\quad 
\mathbf{V} = \mathbf{X}\mathbf{W}_v,
$$

with learned projection matrices $\mathbf{W}_q, \mathbf{W}_k \in \mathbb{R}^{d \times q}$ and $\mathbf{W}_v \in \mathbb{R}^{d \times v}$.
Hence, $\mathbf{Q}, \mathbf{K} \in \mathbb{R}^{n \times q}$ and $\mathbf{V} \in \mathbb{R}^{n \times v}$.

The **scaled dot-product self-attention** [[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)] is then given by

$$
\mathbf{H} = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{q}}\right)\mathbf{V},
$$

where the hyperparameters are the projection dimensions $q$ and $v$.

* The product $\mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{n \times n}$ computes pairwise attention scores between tokens in the sequence.
* The row-wise softmax normalizes these scores into attention weights.
* Multiplication by $\mathbf{V}$ yields the output representations $\mathbf{H} \in \mathbb{R}^{n \times v}$.

When applied to a batch of sequences (e.g., sentences), the self-attention function is computed **independently for each sequence**, so each token only attends to tokens within the same sequence.

---
### Causal Model

For a model to perform **autoregressive prediction**, it must satisfy causality:
given a temporal sequence $\mathbf{X} \in \mathbb{R}^{n \times c}$, a causal model $f$ produces $\mathbf{H} = f(\mathbf{X}) \in \mathbb{R}^{n \times c'}$ such that each output $\mathbf{H}_i$ depends **only** on inputs $\mathbf{X}_j$ with $j \leq i$.

This ensures that information flows only from past and present tokens, not from future ones.

To build a **causal transformer**, we use a *masked* self-attention mechanism:

$$
\mathbf{H} = \operatorname{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^\top \odot \mathbf{M}}{\sqrt{q}}\right)\mathbf{V},
$$

where the mask $\mathbf{M} \in \mathbb{R}^{n \times n}$ has a lower-triangular structure:

$$
M_{ij} =
\begin{cases}
0 & \text{if } j \leq i, \\[6pt]
-\infty & \text{if } j > i,
\end{cases}
$$

This forces each position to attend only to itself and previous positions.

---
### Multi-Head Attention (MHA)

A common generalization is **multi-head attention (MHA)**. Instead of a single set of queries, keys, and values, we compute $h$ independent attention “heads.” For each head $t \in {1,\dots,h}$:

$$
\mathbf{Q}_t = \mathbf{X}\mathbf{W}_{q,t}, 
\quad 
\mathbf{K}_t = \mathbf{X}\mathbf{W}_{k,t}, 
\quad 
\mathbf{V}_t = \mathbf{X}\mathbf{W}_{v,t},
$$

and

$$
\mathbf{H}_t = \operatorname{softmax}\!\left(\frac{\mathbf{Q}_t \mathbf{K}_t^\top \odot \mathbf{M}}{\sqrt{q}}\right)\mathbf{V}_t.
$$

This introduces $3h$ trainable projection matrices (or equivalently a $3 \times h \times q$ tensor if $q=v$).

The outputs from all heads are then concatenated along the feature dimension and projected back:

$$
\mathbf{H} = \big[\,\mathbf{H}_1 \;\; \cdots \;\; \mathbf{H}_h \,\big]\mathbf{W}_o,
$$

where $\mathbf{W}_o \in \mathbb{R}^{(hv) \times o}$ is a learned output projection.

In practice, we typically choose an embedding dimension $m$, an output dimension $o$, and a number of heads $h$, setting

$$
q = v = \frac{m}{h}
$$

for all heads.

---
### Dropout Regularization

Let  
$$
\mathbf{H} \in \mathbb{R}^{b \times f}
$$  
be the output of a fully connected layer with $f$ units, given a mini-batch of $b$ inputs.  

We use **inverted dropout**, defined as:  
$$
\tilde{\mathbf{H}} = \frac{1}{1-p}\,\mathbf{H} \odot \mathbf{M},
$$  
where $\mathbf{M} \in \{0,1\}^{b \times f}$ is a binary mask. Each entry $M_{i,j}$ is drawn from a Bernoulli distribution with parameter $(1-p)$.

Thus:
- with probability $p$, $M_{i,j} = 0$ and the corresponding element $H_{i,j}$ is dropped,  
- with probability $(1-p)$, $M_{i,j} = 1$ and the element is kept but rescaled by $\tfrac{1}{1-p}$.  

When applying (inverted) dropout inside **multi-head attention**, the mask is applied to the **attention weights**.  

If the attention matrix has dimension:  
$$
\mathbf{A} \in \mathbb{R}^{b \times h \times n \times n},
$$  
where:
- $b$ = batch size,  
- $h$ = number of attention heads,  
- $n$ = context length,  

then the corresponding dropout mask is:  
$$
\mathbf{M} \in \{0,1\}^{b \times h \times n \times n}.
$$  

The dropout operation is:  
$$
\tilde{\mathbf{A}} = \frac{1}{1-p}\,\mathbf{A} \odot \mathbf{M}.
$$  

This ensures that some attention links are randomly suppressed during training, improving generalization.

---
### Layer Normalization

Let
$$
\mathbf{H} \in \mathbb{R}^{b \times f}
$$
be the output of a fully connected layer with $f$ units, given a mini-batch of $b$ inputs.  

In **layer normalization**, the mean and variance are computed independently for each row (i.e., for each input vector of dimension $f$):

$$
\tilde{\mu}_i = \frac{1}{f} \sum_{j=1}^{f} H_{i,j},
\qquad 
\tilde{\sigma}_i^2 = \frac{1}{f} \sum_{j=1}^{f} \big(H_{i,j} - \tilde{\mu}_i\big)^2
$$

This formulation works with any batch size (even $b=1$), since normalization is performed along the **feature dimension** only, without introducing inter-batch dependencies.  
We then standardize the output so that each row has mean 0 and standard deviation 1:
$$
[\mathbf{H}^\prime]_{i,j} = \frac{H_{i,j} - \tilde{\mu}_i}{\sqrt{\tilde{\sigma}_i^2 + \epsilon}}
$$

The normalized output is defined as:

$$
\operatorname{LN}(\mathbf{H})_{i,j} = 
\alpha_j \cdot [\mathbf{H}^\prime ]_{i,j} + \beta_j,
$$

where:
- $\alpha, \beta \in \mathbb{R}^f$ are learnable parameters (one per feature dimension),
- $\epsilon > 0$ is a small constant added for numerical stability.  

Thus, layer normalization re-centers each row to mean $0$ and variance $1$, then rescales and shifts it using trainable affine parameters $\alpha$ and $\beta$.

---

In the case where the input is a 3D tensor:

$$
\mathbf{X} \in \mathbb{R}^{b \times t \times d_{\text{in}}},
$$

- $b$: batch size,  
- $t$: context length (number of tokens),  
- $d_{\text{in}}$: embedding dimension (features per token),  

layer normalization is applied **per token vector**, i.e. along the last axis of size $d_{\text{in}}$.  

For each token $\mathbf{x}_{i,k,:} \in \mathbb{R}^{d_{\text{in}}}$, we compute:

$$
\mu_{i,k} = \frac{1}{d_{\text{in}}} \sum_{j=1}^{d_{\text{in}}} X_{i,k,j}, 
\qquad 
\sigma_{i,k}^2 = \frac{1}{d_{\text{in}}} \sum_{j=1}^{d_{\text{in}}} \big(X_{i,k,j} - \mu_{i,k}\big)^2,
$$

and the normalized output:

$$
\operatorname{LN}(\mathbf{X})_{i,k,j} =
\alpha_j \cdot \frac{X_{i,k,j} - \mu_{i,k}}{\sqrt{\sigma_{i,k}^2 + \epsilon}} + \beta_j,
$$

with learnable parameters $\alpha, \beta \in \mathbb{R}^{d_{\text{in}}}$.

---
### GELU

The **Gaussian Error Linear Unit (GELU)** activation is defined as:

$$
\operatorname{GELU}(x) = 0.5x \left[ 1 + \tanh\!\left( \frac{\sqrt{2}}{\pi} \left( x + 0.044715x^{3} \right) \right) \right].
$$

Unlike ReLU, which abruptly sets negative inputs to zero, GELU smoothly weights inputs according to their magnitude. Small negative values are only partially suppressed, while large positive values pass through almost unchanged.  

The smooth nonlinearity improves optimization stability and allows gradients to propagate more effectively during training.  

Transformer-based LLMs (BERT, GPT, etc.) consistently achieve better results with GELU than with ReLU or tanh, especially in language understanding and generation tasks.  

In practice, GELU provides a balance between the sparsity of ReLU and the smoothness of sigmoid/tanh, making it well-suited for large-scale deep learning architectures like transformers.

---
### FeedForward

The **feedforward network (FFN)** is applied independently to each token embedding after the attention layer.  

The intermediate MLP is typically designed as a **2-layer MLP**, where the hidden dimension is an integer multiple of the input dimension  (e.g., 3x, 4x; here we use 4x).  
A nonlinearity is applied in between (commonly GELU in modern LLMs) and **no biases** are used in the linear layers (as per standard transformer implementations). 

---
### Transformer Block

A commonly used variant, used here, is the **pre-normalized transformer block**, which tends to be easier to train.  
Its structure is as follows:

1. Start with a layer normalization operation: $\mathbf{H}=\operatorname{LayerNorm}(\mathbf{X})$
2. Apply a MHA layer: $\mathbf{H}=\mathrm{MHA}(\mathbf{H})$.
3. Add a residual connection: $\mathbf{H}=\mathbf{H}+\mathbf{X}$.
4. Apply a layer normalization operation: $\mathbf{F}=\operatorname{LayerNorm}(\mathbf{F})$
5. Apply a fully-connected model $g(\cdot)$ on each row $\mathbf{F}=g(\mathbf{F})$.
6. Add a residual connection: $\mathbf{F}=\mathbf{F}+\mathbf{H}$.

## GPT Model
The **GPT architecture** was introduced by [Radford et al. (2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pd) as an autoregressive
transformer trained to predict the next token in a sequence.  


Let  
$$
\mathbf{X} \in \mathbb{R}^{b \times n \times vocab}
$$  
be the input token indices, where:
- $b$ = batch size,  
- $n$ = context length (number of tokens)
- $vocab$ = vocabulary size

Since each element of $\mathbf{X}$ is an integer index in the vocabulary, the first step is to map tokens into continuous vectors using **token embeddings**:  

$$
\mathbf{T} \in \mathbb{R}^{b \times n \times d},
$$  

where $d$ is the embedding dimension.  

We then add **positional embeddings**  
$$
\mathbf{P} \in \mathbb{R}^{n \times d}
$$  
to encode the order of tokens in the sequence:

$$
\mathbf{E} = \mathbf{T} + \mathbf{P}, \qquad 
\mathbf{E} \in \mathbb{R}^{b \times n \times d}.
$$

A dropout layer is applied for regularization:

$$
\mathbf{E}' = \operatorname{Dropout}(\mathbf{E}).
$$


The embedded sequence $\mathbf{E}'$ is then passed through a stack of $L$ **transformer blocks**:

$$
\mathbf{H} = \operatorname{TransformerBlocks}(\mathbf{E}'),
\qquad \mathbf{H} \in \mathbb{R}^{b \times n \times d}.
$$

Finally, we apply **layer normalization** followed by a **linear output head** that projects embeddings into vocabulary logits:

$$
\mathbf{Z} = \operatorname{LN}(\mathbf{H}), \qquad
\mathbf{Y} = \mathbf{Z} W,
$$

where $W \in \mathbb{R}^{d \times vocab}$.  

Thus, the model produces logits:

$$
\mathbf{Y} \in \mathbb{R}^{b \times n \times vocab},
$$

which can be converted into probabilities over the vocabulary using softmax for next-token prediction.

## Generate New Text and Sampling Strategies

Text can be generated *autoregressively* using a language model, meaning each new token is predicted based on the previously generated tokens. Several decoding strategies can be applied, each with its own trade-offs.

The simplest strategy is **greedy decoding**, which always selects the token with the highest probability at each step:

$$
t^* = \arg \max_t P(t \mid \text{context})
$$

This method is fast and deterministic but may produce repetitive or generic text since it never explores alternative tokens. In other words, greedy decoding always picks the maximum probability token, making multiple runs with the same start context yield identical results.

Instead of always choosing the most likely token, we can sample from the full probability distribution given by the softmax:

$$
P(t) = \frac{\exp\left(z_t \right)}{\sum_j \exp\left(z_j\right)}
$$

where $z_t$ is the logit (unnormalized score) for token $t$. This introduces randomness and diversity into text generation.

**Temperature scaling** modifies the sharpness of the probability distribution before sampling. The logits are divided by a positive value $T$:

$$
P(t) = \frac{\exp\left(z_t/T \right)}{\sum_j \exp\left(z_j/T\right)}
$$

where $z_t$ is the logit for token $t$ and $T$ is the temperature. Using a temperature of 1 is the same as not using any temperature scaling. In this case, the tokens are selected with a probability equal to the original softmax probability scores. Applying very small temperatures will result in sharper distributions such that the behavior of the multinomial function selects the most likely token almost 100% of the time, approaching the behavior of the argmax function. Likewise, a high temperature results in a more uniform distribution where other tokens are selected more often. This can add more variety to the generated texts but also more often results in nonsensical text.

**Top-k sampling** combines well with temperature scaling. Instead of sampling from the full vocabulary, only the $k$ most probable tokens are considered. All others are discarded by setting their logits to $-\infty$. This strategy guarantees that unlikely tokens are excluded and maintains diversity by allowing randomness among the top candidates. It often leads to more fluent and coherent text compared to unconstrained sampling.  

# Training



# Limitations

While this project is highly valuable for learning purposes, it has several important limitations compared to modern large-scale LLMs:

* **Reduced scale**: the model is trained on a very small English dataset. This makes it prone to **overfitting**, as seen in the divergence between training and validation loss curves. As a result, the model struggles to generalize beyond the training data.
* **Computational simplicity**: it is designed to run on standard hardware, which restricts both dataset size and model complexity. Consequently, its capabilities cannot be compared to large models such as GPT or LLaMA.
* **Lack of efficiency optimizations**: features like distributed training, advanced mixed-precision computation, and GPU/TPU-optimized kernels are not implemented.
* **Limited linguistic ability**: due to the small dataset size, the model does not capture the diversity, robustness, and nuanced semantics found in LLMs trained on billions of tokens.
* **Restricted applicability**: while the model is a solid prototype and an excellent teaching tool, it is not suitable for real-world or production scenarios.
* **No advanced alignment techniques**: modern methods such as RLHF (Reinforcement Learning with Human Feedback), instruction fine-tuning, or retrieval-augmented generation (RAG) are intentionally omitted.

These constraints are not drawbacks but **deliberate trade-offs** to keep the implementation transparent, lightweight, and focused on educational clarity.


# References

* Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention is All You Need*. Advances in Neural Information Processing Systems (NeurIPS). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
* Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., & Sutskever, I. (2019). *Language Models are Unsupervised Multitask Learners* (GPT-2). OpenAI. [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)  
* Raschka, S. (2024). *Build a Large Language Model (From Scratch)*. Manning Publications. [Book page](https://www.manning.com/books/build-a-large-language-model-from-scratch)  


## Additional Readings

* Bishop, C. M. (2025). *Deep Learning: Foundations and Concepts*. Springer. [Book page](https://link.springer.com/book/10.1007/978-3-031-45468-4)  
* Scardapane, S. (2023). *Alice's Adventures in a Differentiable Wonderland*. Springer. [Book page](https://www.sscardapane.it/alice-book/)  


