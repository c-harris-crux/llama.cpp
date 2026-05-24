# MI50 / gfx906 performance experiments

This file records the local MI50/gfx906 performance work done in this tree so future sessions do not need to rediscover the same constraints.

## Environment

- Hardware: AMD MI50 / gfx906, 32 GiB VRAM per card, wave64.
- Target model: `/AI/Models/Qwen3.6-27B/gguf/output-mtp.gguf`.
- Typical benchmark command shape:

```sh
HIP_VISIBLE_DEVICES=0 build/bin/llama-bench \
  -m /AI/Models/Qwen3.6-27B/gguf/output-mtp.gguf \
  -ngl 99 -t 8 --main-gpu 0 --progress \
  -b 2048 -ub 2048
```

## Added environment variables

### `GGML_GFX906_Q8_0_WARP_COOP_MAX_NCOLS`

Controls the experimental gfx906 Q8_0 warp-cooperative MMVQ path.

- Unset: default conservative behavior.
- Numeric value: enable the path up to that column count.
- `all`: enable for all tested column counts.

Current result: correctness passed for targeted Q8_0/F32 single-column MUL_MAT cases, but enabling broadly was slower on the target model.

### `GGML_GFX906_FA_Q8_TILE`

Controls the experimental gfx906 Flash Attention Q8 tile dispatch.

- Unset or `0`: keep existing HIP quantized-KV behavior, which prefers the VEC FA path.
- Any non-empty non-zero value: allow Q8_0 K/V FA to use the gfx906 Q8 tile path for supported head sizes when `Q->ne[1] <= 512`.

The `Q->ne[1] <= 512` cap is intentional. Larger prompt batches crashed in the `DKQ=256, DV=256` Q8 tile specialization during testing.

## Experiments and results

### TOP_K on ROCm

Implemented HIP/ROCm support for `GGML_OP_TOP_K`, based on the AITER top-k approach and adapted to the local ggml CUDA/HIP backend.

Observed result:

```text
HIP_VISIBLE_DEVICES=0,1 build/bin/test-backend-ops test -b HIP -o TOP_K
3/3 backends passed
OK
```

The backend-op test skipped the devices because the test selector did not match the registered backend names as expected, but a server prompt path exercised successfully afterward.

### Q8_0 warp-coop MMVQ

Implemented an opt-in dispatch for gfx906 Q8_0 warp-cooperative MMVQ.

Representative command:

```sh
HIP_VISIBLE_DEVICES=0 GGML_GFX906_Q8_0_WARP_COOP_MAX_NCOLS=all \
  build/bin/llama-bench \
  -m /AI/Models/Qwen3.6-27B/gguf/output-mtp.gguf \
  -ngl 99 -t 8 -fa 1 -ctk q8_0 -ctv f16 \
  --main-gpu 0 --progress -r 3 \
  -b 2048 -ub 2048 -p 512 -n 128
```

Results:

| Configuration | pp512 | tg128 |
| --- | ---: | ---: |
| Baseline FA on | 200.77 t/s | 19.68 t/s |
| Warp-coop enabled broadly | not primary win | 18.50 t/s |

Conclusion: keep this path opt-in. Broad enablement hurt decode throughput on the target model.

### Flash Attention Q8 tile path

Profiling showed the target model with `-fa 1 -ctk q8_0 -ctv f16` spent most time in:

```text
flash_attn_ext_vec<256, 2, GGML_TYPE_Q8_0, GGML_TYPE_F16, false>
```

The HIP quantized-KV dispatch forced the VEC FA path before the existing gfx906 Q8 tile branch could be selected. The VEC path avoids temporary f16 KV buffers but is weaker for prefill.

Representative baseline results:

| Configuration | pp512 | tg128 |
| --- | ---: | ---: |
| FA off, Q8_0 K / F16 V | 208.58 t/s | 19.70 t/s |
| FA on, Q8_0 K / F16 V, VEC path | 200.77 t/s | 19.68 t/s |

Implemented `GGML_GFX906_FA_Q8_TILE=1` as a guarded dispatch before the HIP quantized-KV VEC force path.

Guarded candidate results:

| Configuration | pp512 | tg128 |
| --- | ---: | ---: |
| `GGML_GFX906_FA_Q8_TILE=1` | 208.96 t/s | 19.79 t/s |

Safety test:

```text
HIP_VISIBLE_DEVICES=0 HIP_LAUNCH_BLOCKING=1 GGML_GFX906_FA_Q8_TILE=1 ... -p 768 -n 0
pp768 205.63 t/s
```

This passed after the dispatch cap, because `pp768` falls back instead of using the unstable Q8 tile path.

Failed tests before adding the cap:

- `pp768` failed.
- `pp1024` failed.
- `pp2048` failed.
- The failure path went through `launch_fattn<256,4,8>` and `ggml_cuda_flash_attn_ext_tile_q8_case<256,256>`.
- Disabling KV max scan for the Q8 tile launches did not fix the large-prompt crash.

Conclusion: the gated FA Q8 tile path recovers the `pp512` loss from FA VEC and does not hurt `tg128`, but it is not safe enough to enable by default.

## Unexplored or unfinished avenues

### Fix Q8 tile for `Q->ne[1] > 512`

The most direct next win is debugging the gfx906 Q8 tile kernel for the target model shape:

- `DKQ=256`
- `DV=256`
- Q8_0 K
- F16 V
- failing path around `launch_fattn<256,4,8>`

Likely areas to inspect:

- tile bounds for larger query batches
- mask indexing and `KV_max` interaction
- GQA/head indexing for `ncols1/ncols2`
- shared or register pressure differences on gfx906
- assumptions inherited from later CDNA-oriented kernels

### Tune Q8 tile launch shape

The current Q8 tile launch selection is inherited from the existing gfx906 kernel code. There may be a better MI50-specific launch shape than the current `cols_per_block` and `ncols2` choices for `DKQ=256, DV=256`.

### Make FA dispatch model-aware

The current gate only checks architecture, env var, head-size support, Q8_0 involvement, and `Q->ne[1] <= 512`. If the tile path is fixed, dispatch should be retuned using actual benchmark results across:

- prefill-only sizes: 512, 768, 1024, 2048, 8192
- decode-heavy runs
- mixed prompt/generation runs
- different KV cache type pairs

### Revisit warp-coop MMVQ with narrower dispatch

The broad `GGML_GFX906_Q8_0_WARP_COOP_MAX_NCOLS=all` setting was slower, but smaller and more specific decode shapes might still benefit. The existing env gate allows narrower trials without changing default behavior.

### Profile with cleaner build metrics

Current builds emit substantial HIP resource-usage output because metrics are enabled. For faster iteration, use the repo-local Codex-oriented build script and consider disabling verbose compile metrics when not actively inspecting kernel resource usage.

## Current recommendation

Leave both performance paths opt-in:

```sh
GGML_GFX906_Q8_0_WARP_COOP_MAX_NCOLS=all
GGML_GFX906_FA_Q8_TILE=1
```

Do not enable either by default yet. The FA Q8 tile gate is the more promising path for this model, but the large-prompt crash must be fixed before it is promoted from experiment to default behavior.
