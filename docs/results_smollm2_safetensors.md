# Result log: SmolLM2-135M-Instruct safetensors

**Date:** 2026-08-11  
**Host:** cloud sandbox (not local user machine)  
**Checkpoint:** HuggingFaceTB/SmolLM2-135M-Instruct (`model.safetensors`, ~256 MB)

## Config (from config.json)

- architecture: LlamaForCausalLM
- num_hidden_layers: 30
- hidden_size: 576
- num_attention_heads: 9
- num_key_value_heads: 3
- head_dim: 64
- rope_theta: 100000
- tie_word_embeddings: true

## Run

```text
HybridTransformer.from_checkpoint(path, max_layers=None, stream_layers=True)
generate(prompt=[1,2,3,4], max_new_tokens=6, use_optical=True/False)
```

## Output

```text
layers=30 hidden=576 heads=9 kv=3
optical [1, 2, 3, 4, 198, 198, 198, 376, 446, 476]
digital [1, 2, 3, 4, 198, 198, 198, 376, 446, 476]
identical True
```

Decoded (tokenizer, special tokens visible):

```text
optical text: '<|im_start|><|im_end|><repo_name><reponame>\n\n\nus = "'
digital text: same
```

## Interpretation

On a real HF safetensors weight file, the hybrid optical-score path matches digital greedy decoding for this prompt and depth. Prompt quality is not the claim; **sequence identity under optical vs digital scores** is the claim.
