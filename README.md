# Alias Optimization for SAM3

This repo is a **small exploratory experiment**, not a polished or supported library.

The goal is to test a simple question:

> *Can we find prompt token sequences that cause SAM3 to segment certain out-of-distribution objects better than their natural / ground-truth labels?*

In some cases, the answer appears to be **yes**.

---

## What this is

* A minimal pipeline for **token-level prompt search** against SAM3
* Evaluates segmentation quality (IoU-style metrics) against labeled masks
* Uses naive local / evolutionary search starting from random token sequences
* Intended to explore **latent prompt-conditioned behavior**, not to ship a product

This was written to probe a phenomenon, not to be maximally efficient or general.

---

## What this is *not*

* ❌ Not a production-ready system
* ❌ Not an optimized search implementation
* ❌ Not a claim that this works universally
* ❌ Not an endorsement of adversarial prompting as a general solution

Most concepts fail. A small subset show surprising gains. That’s the point.

---

## Why this might be interesting

For some out-of-distribution objects:

* Natural prompts fail
* Fine-tuning can struggle due to semantic collisions in label space
* But **unnatural token sequences** can sometimes unlock better segmentation

This suggests SAM3 may contain **latent, prompt-inaccessible capabilities** that can be surfaced via search.

A natural next step (not implemented here) would be to use such discovered aliases as a **policy or teacher signal** during adapter fine-tuning.

---

## Rough structure

* `main.py` – entry point for running evaluations
* `optimizer.py` – high-level optimization loop
* `soft_prompt.py` – optional soft-prompt optimization
* `discrete_search.py` – local / evolutionary token search
* `embeddings.py` – vocab + embedding utilities
* `metrics.py` – segmentation evaluation

Expect rough edges.

---

## Performance notes

This code does **not** implement obvious optimizations such as:

* caching image embeddings
* batching evaluations aggressively
* text prefix KV caching

Those would significantly reduce cost if someone wanted to scale this up.

---

## Status

This repo is provided as a **research artifact** / reference implementation.

I don’t plan to actively maintain it, but I’m happy if it’s useful as a starting point or a pointer to an interesting direction.

---

## License

MIT
