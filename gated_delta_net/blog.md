# Gated Delta Net Blog

The recently released Qwen3.5, Nemotron-3-Super and GLM-5 [models](https://sebastianraschka.com/llm-architecture-gallery/#card-minimax-m2-5-230b) are all hybrid attention models, with interleaved linear/sparse and dense/self attention layers.

While standalone linear attention models don't perform as well as dense attention models on recall oriented tasks,
interspersing the two bridges the gap, and allows for much more efficient inference.

Linear attention solves 2 key problems associated with dense attention:

1. **Growing KV-cache**: In dense attention, the key-value cache grows linearly with the sequence length, couple this with the fact that we need to save the KV-cache for each layer, it's easy to see how we can quickly run of memory on long sequence tasks. Self-attention is also a memory bound operation on the GPU because of the need to transfer the KV-cache from GPU to CPU memory, and back again. Linear attention on the other hand has a fixed memory footprint, and can be computed on the fly without needing to save the KV-cache.
2. **Quadratic compute**: Dense attention has a quadratic compute complexity with respect to the sequence length, whereas linear attention has a linear compute complexity.

The outsized inference time benefits that comes with linear attention and the growing number of mainstream model providers adopting it, makes it seem like this is going to be one of those architectural paradigms like MoE, GQA etc. that will likely stick around for a while, making it a worthwhile area to understand deeply.

While there are countless blogs on the workings of dense attention, I didn't find many that explained the Gated Delta Net Attention mechanism used in Qwen3.5 with the same level of detail, so I thought it would be useful to write one myself.

This blog will help you understand the intuition behind linear attention, what Gated Delta Net Attention is, the math that underpins it, some of the geometric interpretations and pytorch code to implement it from scratch.

## From Dense to Linear Attention
The intuition behind linear attention is explained brilliantly in the [Smol Training Playbook](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#excursion-hybrid-models) and I'm just going to paraphrase them here.

The simplified formula of dense attention can be written as:
Now drop the softmax:

$$\mathbf{o}_t = \sum_{j=1}^{t} (q_t^\top k_j) \, v_j$$

Reordering gives:

$$\sum_{j=1}^{t} (q_t^\top k_j) \, v_j = \left( \sum_{j=1}^{t} v_j k_j^\top \right) q_t$$

We define the running state:

$$S_t \triangleq \sum_{j=1}^{t} k_j v_j^\top = K_{1:t}^\top V_{1:t} \in \mathbb{R}^{d \times d}$$

with the simple update:

$$S_t = S_{t-1} + k_t v_t^\top$$

### References
* (LLM Architecture Gallery)[https://sebastianraschka.com/llm-architecture-gallery/#card-minimax-m2-5-230b]
* (A Systematic Analysis of Hybrid Linear Attention)[https://arxiv.org/pdf/2507.06457]