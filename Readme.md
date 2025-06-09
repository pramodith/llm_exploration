# Summary
This repository consists of various projects/ideas that I've tinkered with and blogged about on my [website](https://pramodith.github.io) or [Medium](https://medium.com/@buildingblocks).

Most of the projects are structured as jupyter notebooks and are self-contained. Here's a brief overview of each of the projects and the accompanied blog posts:

## 1. Simple GRPO Trainer

This repository provides an implementation of the GRPO (Group Relative Policy Optimization) Algorithm and a Trainer for training reasoning Large Language Models (LLMs) without using any libraries like TRL, veRL etc. It includes dataset processing, reward functions, and training logic using PyTorch Lightning and Hugging Face Transformers.

Blog Post: [Simple GRPO Trainer](https://pramodith.github.io/posts/grpo-trainer/)

Instructions to run/install can be found in the [readme](https://github.com/pramodith/llm_exploration/blob/main/simple_grpo_trainer/Readme.md) of the sub-project.

## 2. Mechanistic Interpretability: Superposition 101

Mechanistic interpretability is an emerging area of research in AI focused on understanding the inner workings of neural networks. LLMs and Diffusion models have taken the world by storm in the past couple of years but despite they're jaw dropping capabilities very little is known about how and why these deep neural networks generate these outputs.

In this notebook we'll attempt to breakdown some of the key ideas of Mechanistic Interpretability (mech interp.). We haven't found many good resources to understand the fundamentals of mech interp, there's a sense of irony in how dense the literature on a field that aims to make it easier to understand neural nets is 😅. 

As a novice diving into this area of research my goal is to improve my understanding of the topic as I learn and hopefully make it easier for others to learn too. The initial articles will be heavily based on a [blog](https://transformer-circuits.pub/2022/toy_model/index.html) released by Chris Olah and team in 2022, the [code](https://colab.research.google.com/github/anthropics/toy-models-of-superposition/blob/main/toy_models.ipynb) encompassed in this series largely derives from it too. 

I found the blog quite dense for a newbie to follow so my aim is to dumb it down as much as possible. Word of caution, that even my series expects readers to have a good understanding of ML and how to train Deep Neural Networks. If you've completed a ML 101 class in your schooling you should have no trouble following these articles.

Link to [blog](https://pramodith.github.io/posts/superposition/)

## 3. Exploring Sink Tokens
This notebook is associated with our Medium Article here, where we detail our exploration of encoder and decoder-only Transformer
models and the existence of *sink tokens* in them.

At a high level *sink tokens* are a small group of tokens that transformer models use to offload a very high proportion of attention scores to. For more, please read our [article](https://medium.com/@buildingblocks/llms-may-not-need-dense-self-attention-1fa3bf47522e?source=friends_link&sk=7f278a97d7f236c8b7bf9782cb75b035).

In this notebook we'll visualize the attention scores of various models and identify the tokens which are allocated the highest amount of attention, on a layer-by-layer basis. We'll show how sink tokens are prominent among encoder and decoder models of all sizes.

## 4. Sparse Attention Leveraging Sink Tokens
In our prior notebook, we found that both encoder-only and decoder-only models offload a significant portion of their attention scores to **sink tokens**. We identified that these sink tokens tend to be either special tokens like **[CLS], [SEP]** or tokens corresponding to **punctuations**. The consistence display of this phenomenon across model architectures and inputs makes one question the relevance of dense self-attention.

In this notebook we'll explore the performance of BERT by creating custom attention masks, which will be sparse in nature. We'll create a unique mask per each token, where all tokens attend to special tokens and the k tokens in their neighborhood. When visualized the tokens along a diagonal of size 2*k+1 and the first anad last tokens (in the case of BERT) being attended to. We'll also explore the effects of allowing dense attention in some layers and sparse attention in the rest.

We'll assess the downstream performance of the models that leverage this type of custom attention mask on some commonly used datasets for benchmarking like [TBD]. Our resulting article can be found [here](https://medium.com/towards-artificial-intelligence/comparing-dense-attention-vs-sparse-sliding-window-attention-6cd5b2e7420f).

## 5. Riddle Reasoning Model
This notebook trains a reasoning model using GRPO on a dataset of riddles via Unsloth and TRL.