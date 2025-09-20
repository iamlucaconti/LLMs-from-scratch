# LLMs-from-scratch

This repository contains an implementation of a **Large Language Model (LLM)** developed entirely *from scratch*.
The goals are to understand the theoretical principles underlying modern language architectures and to gain practical experience by writing the code that allows building and training a functioning LLM.

> The project follows step by step the book [*Build a Large Language Model (From Scratch)*](https://www.manning.com/books/build-a-large-language-model-from-scratch) by *Sebastian Raschka (2024)*, enriching the concepts with mathematical explanations, technical notes, and didactic experiments. You can find the original repository [here](https://github.com/rasbt/LLMs-from-scratch).

Specifically, it covers:

* the **theoretical foundations** of the transformer (self-attention, multi-head attention, feed-forward networks, normalization, regularization),
* the construction of a **causal transformer** capable of generating text autoregressively,
* different **sampling strategies** for generation (greedy decoding, temperature scaling, top-k sampling),
* practical aspects of **training and evaluating** the model.


# Dataset and Training
The training corpus for this LLM consists of “The Verdict” by Edith Wharton, a short story published in 1908 and now in the public domain. The text is openly available on Wikisource [here](https://en.wikisource.org/wiki/The_Verdict).
Because the dataset is relatively small in scale, it primarily serves as a didactic resource for experimenting with transformer-based language models rather than for building a model with strong generalization capabilities. In practice, training on such a limited corpus makes the model prone to overfitting, as it may memorize sentence patterns instead of learning broadly applicable linguistic structures. This behavior can be observed by comparing the training and validation loss curves, where validation loss begins to diverge once overfitting occurs.


# How to Run

To **pre-train the model**, open and execute the notebook:

```
pretrain_llm.ipynb
```

Within this notebook, you can adjust key **hyperparameters** (such as learning rate, batch size, context length, and number of epochs) in the **Global** configuration section.

For a more didactic, step-by-step walkthrough, the directory `book_chapters/` contains Jupyter notebooks corresponding to the main chapters of *Build a Large Language Model (From Scratch)* by Sebastian Raschka. Each notebook introduces concepts incrementally.

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


