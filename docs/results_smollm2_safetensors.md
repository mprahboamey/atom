# Result log: SmolLM2-135M-Instruct safetensors

**Checkpoint:** [HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)  
**Format:** Hugging Face `model.safetensors` + `config.json`

## Model config

| Field | Value |
|-------|--------|
| Layers | 30 |
| Hidden | 576 |
| Heads / KV | 9 / 3 |
| Head dim | 64 |
| RoPE theta | 100000 |
| Tied embeddings | yes |

## Result A — fixed token prompt

```text
prompt:  [1, 2, 3, 4]
optical: [1, 2, 3, 4, 198, 198, 198, 376, 446, 476]
digital: [1, 2, 3, 4, 198, 198, 198, 376, 446, 476]
identical: True
```

## Result B — natural language prompt

```text
prompt text: "The capital of France is"
prompt ids:  [504, 3575, 282, 4649, 314]
optical ids: [504, 3575, 282, 4649, 314, 260, 768, 282, 282]
digital ids: [504, 3575, 282, 4649, 314, 260, 768, 282, 282]
identical: True
optical text: 'The capital of France is the most of of'
digital text: 'The capital of France is the most of of'
```

(Generation quality is limited by model size and short decode; the claim is **optical ≡ digital**, not factual correctness.)

## Result C — full forward logits (8 tokens)

```text
logits MSE ~ 7.9e-4   (float16 path; small numeric drift)
logits top-1 agreement: 100%
```

## How to reproduce

```bash
python examples/16_safetensors_text_eval.py \
  --model /path/to/SmolLM2-135M-Instruct \
  --prompt "The capital of France is" \
  --max-new 4

ATOM_SAFETENSORS_MODEL=/path/to/SmolLM2-135M-Instruct pytest tests/test_hybrid_identity.py -q
```
