# VRAM Management

Hardware target: NVIDIA RTX 5070 Ti Laptop GPU (12 GB / 12227 MiB total).

## Model VRAM Footprints

All measurements taken on RTX 5070 Ti with `torch.cuda.memory_allocated()`.

| Model | Loaded (MB) | Peak during generation (MB) | Notes |
|---|---|---|---|
| Chatterbox Turbo | 4,206 | 4,410 | ~350M params, single diffusion step |
| Higgs Audio 3B (8-bit) | 6,687 | 7,102 | Includes 774 MB audio tokenizer |
| Higgs Audio 3B (4-bit) | 4,230 | 5,019 | Includes 774 MB audio tokenizer |

The audio tokenizer for Higgs accounts for ~774 MB regardless of quantization level.

### Breakdown: Higgs components

| Component | VRAM (MB) |
|---|---|
| Audio tokenizer | 774 |
| LLM weights (8-bit) | 5,913 |
| LLM weights (4-bit) | 3,456 |

## VRAM Reclamation

Tested unload sequence: `del model` -> `gc.collect()` -> `torch.cuda.empty_cache()`.

| Model | After unload (MB) | Leaked (MB) |
|---|---|---|
| Chatterbox Turbo (load only) | 0.0 | 0.0 |
| Chatterbox Turbo (after generation) | 8.1 | 8.1 |
| Higgs 8-bit (load only) | 0.0 | 0.0 |
| Higgs 8-bit (after generation) | 9.1 | 9.1 |
| Higgs 4-bit (after generation) | 9.1 | 9.1 |

Findings:
- **Model weights are fully reclaimed** on unload (0 MB leak when no generation is performed).
- **After generation**, ~8-9 MB of CUDA context remains allocated. This is negligible and does not accumulate across swap cycles.

## Swap Cycle Stability

Tested 3 consecutive load/unload cycles for each model:

**Chatterbox (3 cycles):**
- Cycle 1 load: 4,206 MB -> unload: 0.0 MB
- Cycle 2 load: 4,206 MB -> unload: 0.0 MB
- Cycle 3 load: 4,206 MB -> unload: 0.0 MB
- Cumulative leak: **0.0 MB**

**Higgs 8-bit (3 cycles):**
- Cycle 1 load: 6,687 MB -> unload: 0.0 MB
- Cycle 2 load: 6,687 MB -> unload: 0.0 MB
- Cycle 3 load: 6,687 MB -> unload: 0.0 MB
- Cumulative leak: **0.0 MB**

Both models show **zero cumulative VRAM leak** across swap cycles. The one-at-a-time model swap strategy in `ModelManager` is safe for long-running server operation.

## Co-residency Analysis (Informational)

Can two models fit simultaneously on 12 GB?

| Combination | Estimated VRAM (MB) | Fits in 12 GB? |
|---|---|---|
| Chatterbox + Higgs 4-bit | ~8,436 (loaded) / ~9,429 (peak) | Loaded: yes. Peak: borderline. |
| Chatterbox + Higgs 8-bit | ~10,893 (loaded) / ~11,512 (peak) | Loaded: borderline. Peak: no. |
| Higgs 4-bit + Higgs 8-bit | ~10,917 | No |

**Decision**: Co-residency is not implemented. The `ModelManager` uses a one-at-a-time strategy with lazy loading and 60-second idle auto-unload. This is the safe choice given:
1. Peak VRAM during generation adds 200-800 MB beyond static model size.
2. OS and display driver consume ~1.7 GB baseline.
3. Leaves no headroom for CUDA fragmentation or context overhead.

## Recommended AVAILABLE_VRAM Setting

With the RTX 5070 Ti (12,227 MiB total, ~1,769 MiB used by OS/driver at idle):

| Setting | Value | Enables |
|---|---|---|
| Conservative | 8000 | Higgs 8-bit (peak ~7,100 MB) + headroom |
| Default | 10000 | All models, comfortable headroom |
| Aggressive | 10500 | All models, minimal headroom |

Recommended `.env` setting: `AVAILABLE_VRAM=10000`

This allows any single model to load and generate with ~3-5 GB headroom beyond peak usage.

## Model Load Times

Measured on the same hardware:

| Model | Load time |
|---|---|
| Chatterbox Turbo | ~3 seconds |
| Higgs 8-bit | ~12 seconds |
| Higgs 4-bit | ~21 seconds |

4-bit quantization is slower to load because the quantization conversion happens at load time. 8-bit uses `bitsandbytes` LLM.int8() which is faster to initialize.
