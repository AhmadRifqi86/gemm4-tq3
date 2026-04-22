# Bridging llamacpp (Gemma4) → turbo3-cuda (TurboQuant)

## Background

`turbo3-cuda` is a fork of `llama.cpp` that adds TurboQuant KV cache compression
(`--cache-type-k turbo3 --cache-type-v turbo3`). The fork diverged from upstream
at commit `ded446b34` ("opencl: allow large buffer for adreno", March 26 2026).

The main `llama.cpp` repo added Gemma4 architecture support across ~20 commits
after that divergence point. This document covers how to port those changes into
`turbo3-cuda` without doing a full rebase.

---

## Why not a full rebase?

| | turbo3-cuda | llamacpp HEAD |
|---|---|---|
| Commits since fork | 107 | 292 |
| Files changed | 54 | Hundreds |
| `src/llama-model.cpp` | Not touched | Heavily modified |
| `src/models/*.cpp` | Not touched | Refactored + Gemma4 added |

Two large upstream refactors happened after the fork:
- `9db77a020` — QKV refactor touching every model file
- `4fbdabdc6` — single `llm_build` per arch, splitting model code into `src/models/`

Rebasing 107 turbo3 commits over these would cause hundreds of conflicts.
Cherry-picking only the Gemma4 commits is far safer because turbo3 never
touches `src/models/*.cpp`.

---

## Repository paths (local)

```
xpandas/llamacpp/          ← main llama.cpp with Gemma4
xpandas/turboquant-qwen/turbo3-cuda/   ← TurboQuant fork (target)
```

---

## Step 0: Add llamacpp as a remote (one-time)

```bash
cd xpandas/turboquant-qwen/turbo3-cuda

git remote add llamacpp /home/arifadh/Desktop/Skripsi-Magang-Proyek/proyek/xpandas/llamacpp
git fetch llamacpp --no-tags
```

Verify merge base:
```bash
git merge-base HEAD llamacpp/master
# Expected: ded446b34c0cd803a0122446b848619adbb458cf
```

---

## Step 1: Cherry-pick Gemma4 commits (in order, oldest first)

```bash
git cherry-pick 63f8fe0ef   # initial Gemma4: gemma4-iswa.cpp, arch enum, mmproj
git cherry-pick 5764d7c6a   # per-layer projections
git cherry-pick 057dba336   # multimodal padding fix for gemma3n/gemma4
git cherry-pick f772f6e43   # NVFP4 tensor support for Gemma4
git cherry-pick fcc750875   # Gemma4 model type detection (#22027)
```

**Do NOT manually copy `src/models/gemma4-iswa.cpp` before cherry-picking.**
The first cherry-pick creates it automatically. Copying it first will cause:
```
error: The following untracked working tree files would be overwritten by merge:
    src/models/gemma4-iswa.cpp
```
If you made this mistake: `rm src/models/gemma4-iswa.cpp` then retry.

---

## Step 2: Resolving cherry-pick conflicts

### Conflict in `tools/mtmd/mtmd.cpp`

The first cherry-pick (`63f8fe0ef`) will conflict here. turbo3-cuda's
`mtmd.cpp` is slightly behind the version the cherry-pick expects.

Resolution — accept the incoming Gemma4 version:
```bash
git checkout --theirs tools/mtmd/mtmd.cpp
git add tools/mtmd/mtmd.cpp
git cherry-pick --continue
```

### Conflict in `src/llama-graph.cpp` (if it occurs)

turbo3-cuda adds ~66 lines of KV cache hooks in `build_attn`.
Gemma4 adds ~25 lines for its architecture registration.
They are in different functions — keep **both** sides:

```bash
# Open the file, remove <<<<<<< ======= >>>>>>> markers, keep all code from both sides
git add src/llama-graph.cpp
git cherry-pick --continue
```

---

## Step 3: Fix missing mtmd files

After cherry-picking, `tools/mtmd/` references files that were added between
the merge base and the first Gemma4 commit (e.g., `mtmd-image.h` added in `#21031`).

Copy the entire mtmd directory from llamacpp:
```bash
cp -r /home/arifadh/Desktop/Skripsi-Magang-Proyek/proyek/xpandas/llamacpp/tools/mtmd/ tools/
```

---

## Step 4: Fix `llama-common` → `common` rename

llamacpp renamed `libcommon` to `libllama-common` in commit `#21936`.
The `tools/mtmd/CMakeLists.txt` we copied uses the new name, but turbo3-cuda
still uses the old name `common`. Edit `tools/mtmd/CMakeLists.txt`:

```cmake
# Change these two lines (line ~114 and ~120):
# FROM:
target_link_libraries  (${TARGET} PRIVATE llama-common mtmd Threads::Threads)
target_link_libraries(llama-mtmd-debug PRIVATE llama-common mtmd Threads::Threads)

# TO:
target_link_libraries  (${TARGET} PRIVATE common mtmd Threads::Threads)
target_link_libraries(llama-mtmd-debug PRIVATE common mtmd Threads::Threads)
```

---

## Step 5: Build

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DGGML_CUDA_FORCE_CUBLAS=OFF \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

> For RTX 5090 / SM 12.0: use `-DCMAKE_CUDA_ARCHITECTURES=120`
> For RTX 3090/4090 / SM 86/89: use `-DCMAKE_CUDA_ARCHITECTURES=89`
> For Raspberry Pi 5 (no CUDA): remove all `-DGGML_CUDA*` flags, add `-DGGML_NEON=ON -DGGML_NATIVE=ON`

---

## Step 6: Verify

```bash
# Confirm Gemma4 loads
./build/bin/llama-cli \
  -m /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  -n 1 --log-disable -p "test" 2>&1 | head -10
# Should show: arch = gemma4
# Error before fix was: unknown model architecture: 'gemma4'

# Run with TurboQuant KV cache
./build/bin/llama-server \
  -m /path/to/gemma-4-E4B-it-Q4_K_M.gguf \
  --cache-type-k turbo3 --cache-type-v turbo3 \
  -c 32768 -fa -ngl 99 \
  --jinja --host 0.0.0.0 --port 8080
```

---

## Potential errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `unknown model architecture: 'gemma4'` | Gemma4 not ported yet | Follow this guide |
| `untracked working tree files would be overwritten` | Manually copied `gemma4-iswa.cpp` before cherry-pick | `rm src/models/gemma4-iswa.cpp` then retry |
| `fatal error: mtmd-image.h: No such file or directory` | File added in `#21031`, missing from turbo3 | `cp -r llamacpp/tools/mtmd/ tools/` |
| `'struct clip_hparams' has no member named 'image_resize_algo'` | clip files outdated | `cp -r llamacpp/tools/mtmd/ tools/` |
| `fatal error: arg.h: No such file or directory` | mtmd CMakeLists links `llama-common` (renamed), so include path missing | Change `llama-common` → `common` in `tools/mtmd/CMakeLists.txt` |
| MMQ segfault during inference | CUDA 13.x incompatibility | Use CUDA 12.8 |
| Question marks in output (`????`) | Using `q4_0` or `q8_0` KV cache type | Use `turbo3` or `bf16` only |

---

## Files changed by this integration

| File | Action | Notes |
|---|---|---|
| `src/models/gemma4-iswa.cpp` | Created by cherry-pick | Gemma4 model graph |
| `tools/mtmd/models/gemma4v.cpp` | Created by cherry-pick | Gemma4 vision encoder |
| `src/llama-arch.cpp` / `.h` | Modified by cherry-pick | Gemma4 arch enum |
| `src/llama-model.cpp` / `.h` | Modified by cherry-pick | Gemma4 weight loading |
| `tools/mtmd/*` | Replaced wholesale | Copied from llamacpp HEAD |
| `tools/mtmd/CMakeLists.txt` | Manually edited | `llama-common` → `common` |

---

## What was NOT changed (turbo3 stays intact)

- `ggml/src/ggml-cuda/turbo-*.cu/cuh` — CUDA turbo3 kernels
- `ggml/src/ggml-metal/ggml-metal.metal` — Metal turbo3 shaders
- `ggml/src/ggml-turbo-quant.c` — CPU turbo3 quantization
- `src/llama-kv-cache.cpp` / `.h` — TurboQuant KV cache integration
- `src/llama-graph.cpp` — turbo3 `build_attn` hooks (kept alongside Gemma4 additions)
