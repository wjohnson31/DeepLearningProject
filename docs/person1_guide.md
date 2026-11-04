# Person 1: what you are responsible for

Your handoff is a working, measured dense-language-model pipeline. Karan should
be able to evaluate your checkpoint and run one training update before adding MoE.
Both of you write the milestone report after the implementation handoff.

## What the code does

1. prepare_data.py streams the **original** TinyStories text files at a pinned
   Hugging Face revision. It normalizes whitespace and removes duplicate stories,
   chooses whole stories from the training source, and fits the final byte-level
   BPE on exactly the retained training stories. The vocabulary contains 4096
   entries including four special tokens and all byte values.
2. It selects separate development and test prefixes from the official validation
   stream, excluding retained training hashes and duplicates. Each story receives
   EOS before packing. Stories stay in one split; windows can cross EOS boundaries
   within that split. Actual counts and source indices are in metadata.json.
3. data.py creates 129-token slices, with the first 128 as inputs and the last
   128 as targets. Adjacent windows overlap by one token, so target positions do
   not repeat. Incomplete final windows are discarded. A shuffle seeded by
   seed + epoch defines training order. The remaining incomplete microbatch at
   an epoch boundary is dropped. Saved epoch and position reproduce the order.
4. model.py learns token and position embeddings, then applies four pre-normalized
   causal attention/FFN blocks. A final normalization precedes the output
   projection, which shares the token embedding weight.
5. train.py uses AdamW and gradient accumulation, so two microbatches of 16
   sequences make one update of 32 sequences × 128 predictions = **4096 target
   token exposures**. The cross-entropy is averaged over these predictions.
6. evaluate.py sums development negative log-likelihood across all batches and
   divides by the number of predictions. Perplexity is exp(CE). Neither metric
   includes routing penalties. The test partition is never evaluated by these
   milestone commands.

## Pilot and horizon

Run the 200-update CUDA pilot with the intended 5000-update configuration. It
produces results.json with measured full-update throughput after 20 warmup
updates. Approximate remaining training time is
(5000 - 200) × 4096 / tokens_per_second, plus evaluation/checkpoint overhead.

If that fits your available GPU time, resume the pilot with the unchanged
configuration and horizon. If you select the 2500-update fallback instead, use
configs/milestone_2500.yaml and start a **new run from scratch**. Its warmup is
100 updates; the 5000-step pilot had a different learning-rate trajectory.
Karan must inherit the same chosen configuration for the controlled comparison.

The --stop-after N option stops at an absolute completed update. It does not
shorten the configured schedule. Resume into a new directory and keep best.pt
next to last.pt from the parent run. The resumed run carries forward all
development evaluations and the selected best checkpoint.

New checkpoints embed the selected best weights inside last.pt, which permits
recovery if a disconnect leaves the separate best.pt file missing or inconsistent.
The included older CPU pilot still requires its matching best.pt beside last.pt.
New best.pt files are evaluation-only; use last.pt to resume the optimizer and
data order.

## What to understand before handing this off

- Causal attention prevents a token from using future inputs. The causality test
  changes later tokens and checks that earlier logits stay unchanged.
- Input/target shifting teaches next-token prediction. CE decreases when the
  correct next token receives more probability.
- A repeated-batch loss decrease verifies learning mechanics; it does not establish
  generalization. The full development evaluation supplies that evidence.
- Weight tying reduces stored parameters and connects input/output token
  representations. The parameter count counts each shared tensor once.
- Training CE is a batch statistic; development CE covers a fixed held-out set.
  Compare models at the same exposure budget, tokenizer and evaluation protocol.
- A theoretically similar FLOP count does not guarantee similar wall-clock speed.
  CUDA throughput and peak allocated memory should be measured for both models.

## Documented implementation decisions

- LayerNorm, exact GELU, bias in linear layers, no output-head bias.
- Dropout 0.1 on embeddings, attention probabilities, attention residual output,
  and FFN residual output; one FFN residual dropout operation per block.
- Token/position embeddings and linear weights initialize from N(0, 0.02²);
  linear biases initialize to zero. LayerNorm uses PyTorch defaults.
- Weight decay applies to matrix parameters, including embeddings; bias and
  normalization parameters are excluded.
- Float32 CE computation; CUDA FP16 autocast plus gradient scaling. CPU checks
  explicitly use FP32. FP16 overflows reduce the scale and retry the same batch
  and RNG state up to ten times. Nonfinite FP32 gradients abort for investigation.
- No padding or masked target positions occur in fixed full windows.
- Development checkpoint selection includes only trained evaluations, not step 0.
- Auxiliary losses are averaged across modules that supply them; dense mode
  supplies scalar zero losses and empty statistics.
- Learning rate uses the zero-based update index; warmup rises to the peak, then
  cosine-decays toward the minimum at the planned horizon.
- FLOPs use two operations per multiply-add and include QKV/output projections,
  full-square attention QK/AV products, FFNs and the vocabulary head. The estimate
  omits scalar activation, normalization and softmax operations. It is an
  analytical matmul estimate; Person 2 must add router computation and active
  expert costs before claiming a whole-model compute match.

## Reference material

- [Original dataset and file definitions](https://huggingface.co/datasets/roneneldan/TinyStories)
- [PyTorch mixed-precision guidance](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [Colab GPU availability and runtime limits](https://research.google.com/colaboratory/faq.html)

This implementation was prepared with AI assistance. Review the code, run the
commands, understand the measurements, and follow your course's attribution rules.
