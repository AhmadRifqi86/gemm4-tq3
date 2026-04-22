---
language:
- en
- zh
- ko
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
tags:
- unsloth
- qwen
- qwen3.5
- reasoning
- chain-of-thought
- lora
pipeline_tag: image-text-to-text
datasets:
- nohurry/Opus-4.6-Reasoning-3000x-filtered
- Jackrong/Qwen3.5-reasoning-700x
- Roman1111111/claude-opus-4.6-10000x
---

# 🌟 Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2

## 📢 Announcement

> **v2 Update:**
> This iteration is powered by **14,000+ premium Claude 4.6 Opus-style general reasoning samples**, with a major focus on **optimizing reasoning economy and structural efficiency**. 
>
> v2 introduces a **refined reasoning scaffold** designed to eliminate redundant internal loops, significantly improving the model's **cross-task generalization** from logic and math into specialized fields like programming. Compared to the original model, **autonomy and stability are significantly improved**, ensuring the model remains robust and self-consistent during complex, multi-step problem solving. v2 is built to **think smarter, not longer**, ensuring high-quality analytical depth with a much better reasoning-cost-to-quality ratio.

![HCaJnUQaoAAaMIc](https://cdn-uploads.huggingface.co/production/uploads/66309bd090589b7c65950665/k4JsVit-31Fw8RZ4pFkun.jpeg)

## 💡 Model Introduction
**Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2** is the second iteration of this reasoning-focused Qwen3.5-4B fine-tune, built to improve the *efficiency* of chain-of-thought generation while preserving strong general reasoning behavior.

Compared with the earlier version, **v2 was trained with 14,000 Claude 4.6 Opus-style general reasoning samples**, with a stronger emphasis on transferring concise, reusable reasoning patterns rather than only maximizing raw benchmark scores. The goal of v2 is not simply to make the model "think more," but to help it **think more economically**: reducing unnecessarily long internal chains, avoiding verbose over-analysis on easy problems, and producing answers with a better reasoning-cost-to-quality ratio.

A key design choice in v2 is that the distillation data is **primarily general-domain reasoning data**—specifically focused on mathematics, word problems, logical deduction, and a balanced mix of general knowledge and instructions—rather than specialized code-heavy supervision. Consequently, **HumanEval and HumanEval+ are employed here to evaluate cross-task generalization and capability transfer**, rather than serving as direct optimization targets. High performance on these benchmarks, despite the lack of code-centric training, confirms that the model's **reasoning scaffold** has become more robust and transferable, proving that fundamental reasoning logic can effectively power specialized tasks like programming.

### Why v2 matters

Relative to the official Qwen3.5-4B baseline, the fine-tuned v2 model still trails slightly in absolute HumanEval accuracy after fair rescoring, but it shows **substantial gains in reasoning efficiency**:

| Metric | Official Qwen3.5-4B | v2 Fine-tuned Model | Change |
|---|---:|---:|---:|
| Average think length | 2829 chars | **1874 chars** | **🟢 -33.77%** |
| HumanEval base passes per 10k think chars | 3.104 | **4.393** | **🟢 +41.54%** |
| HumanEval+ passes per 10k think chars | 2.910 | **4.165** | **🟢 +43.15%** |
| Think chars needed per HumanEval base pass | 3222 | **2276** | **🟢 -29.35%** |
| Think chars needed per HumanEval+ pass | 3437 | **2401** | **🟢 -30.14%** |

At the same time, the official model remains stronger in absolute benchmark score:

| Fairly Recomputed Benchmark | Official Qwen3.5-4B | v2 Fine-tuned Model | Gap |
|---|---:|---:|---:|
| HumanEval (base tests) pass@1 | **0.7683** | 0.7317 | **🔴 -3.66 pts** |
| HumanEval+ (base + extra tests) pass@1 | **0.7256** | 0.6951 | **🔴 -3.05 pts** |

This trade-off is important to understand correctly.

For users who care only about the **highest possible benchmark accuracy**, the official model is still the stronger option. However, for users who care about **reasoning efficiency per unit of inference budget**, v2 is meaningfully improved.

That matters especially for:

- **Resource-constrained local deployment**: On consumer GPUs or lower-memory local setups, shorter and cleaner reasoning traces can reduce latency, memory pressure, and the effective cost of generation.
- **Agentic workflows**: In multi-step agents, the model often solves many *easy* or *medium* subtasks. In those settings, excessively elaborate chain-of-thought can become a tax on throughput. A model that reaches a workable answer with fewer reasoning tokens can improve end-to-end agent speed and lower cumulative inference cost.
- **Open-source tool use and emerging agent stacks**: For users building with lightweight open reasoning systems, browser-use agents, terminal agents, or projects in the "OpenClaw / local autonomous agent" style ecosystem, a model that sacrifices a small amount of peak accuracy for much better reasoning economy can be more practical in real-world loops.
- **Simple problems at scale**: One common issue with strong reasoning-tuned base models is that they sometimes produce very elaborate internal traces even for simple prompts. While that can look impressive, it is often inefficient in practice. v2 is explicitly aimed at trimming this overhead.

In short, **v2 does not claim to beat the official model on absolute coding benchmark score**. Instead, it demonstrates a more deployment-oriented optimization target: **faster, shorter, more economical reasoning** with still-competitive generalization. For many local users, agent builders, and cost-sensitive applications, this can be a highly favorable trade.

## 🗺️ Training Pipeline Overview

```text
Base Model (Qwen3.5-4B)
 │
 ▼
Qwen3.5-4B fine-tuned with Unsloth
 │
 ▼
Supervised Fine-Tuning (SFT) + LoRA
(Response-Only Training masked on "<|im_start|>assistant\n<think>")
 │
 ▼
Jackrong/Qwen3.5-4B-Claude-4.6-Opus-Reasoning-Distilled-v2
```

### 🧠 Example of Learned Reasoning Scaffold（Example）

The model includes targeted optimizations addressing Qwen3.5’s tendency toward excessive transitional or repetitive reasoning on simple queries. Through deep distillation and structural imitation of Claude-4.6-Opus reasoning chains, the model adopts a more efficient structured thinking pattern:  
**“Let me analyze this request carefully: 1..2..3...”.**  
This streamlined reasoning paradigm significantly reduces redundant cognitive loops while preserving deep analytical capacity, resulting in substantially improved inference efficiency.

```text
Let me analyze this request carefully:

1. Identify the core objective of the problem.
2. Break the task into clearly defined subcomponents.
3. Evaluate constraints and edge cases.
4. Formulate a step-by-step solution plan.
5. Execute the reasoning sequentially and verify consistency.
            .
            .
            .
```

### 📚 All Datasets Used
The dataset consists of high-quality, filtered reasoning distillation data:

| Dataset Name | Description / Purpose |
|--------------|-----------------------|
| [nohurry/Opus-4.6-Reasoning-3000x-filtered](https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered) | Provides comprehensive Claude 4.6 Opus reasoning trajectories. |
| [Roman1111111/claude-opus-4.6-10000x](https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x) | Large-scale public Claude 4.6 Opus distillation data used to strengthen general reasoning transfer in v2. |
| [TeichAI/claude-4.5-opus-high-reasoning-250x](https://huggingface.co/datasets/TeichAI/claude-4.5-opus-high-reasoning-250x) | Injecting high-intensity, structured reasoning instances. |
| [Jackrong/Qwen3.5-reasoning-700x](https://huggingface.co/datasets/Jackrong/Qwen3.5-reasoning-700x) | Additional curated reasoning samples designed to strengthen structured step-by-step problem solving and improve reasoning diversity. |


## ⚠️ Limitations & Intended Use
- **Hallucination Risk:** While reasoning is strong, the model remains an autoregressive LLM; external facts provided during the thinking sequence may occasionally contain hallucinations if verifying real-world events.
- **Intended Scenario:** Best suited for offline analytical tasks, coding, math, and heavy logic-dependent prompting where the user needs to transparently follow the AI's internal logic.
- This model is a test version intended solely for learning and demonstration purposes, and is for academic research and technical exploration use only.

## 🙏 Acknowledgements
Significant thanks to the [Unsloth AI](https://unsloth.ai/) team for making rapid fine-tuning of large LLM models accessible. Additionally, we acknowledge Qwen internally, and the open-source community developers producing exceptional distilled datasets.