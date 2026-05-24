# Codex Agent Notes

This repository has a `codex_scripts/` directory for helper scripts intended for Codex or other LLM coding agents. These scripts are separate from the user's normal build scripts so agents can run repeatable checks without overwriting the user's preferred workflow, build directory, or main `compilation_log.txt`.

Use `codex_scripts/SCRIPT_compile_MI50_with_tests.sh` when validating MI50/gfx906 ROCm changes. It keeps output in `build-codex-mi50/` and `codex_compilation_log.txt`, enables ccache with a writable default cache under `/tmp/codex-ccache`, disables HIP metrics by default, and avoids compiling all flash-attention quant variants unless explicitly requested.

Default usage:

```bash
codex_scripts/SCRIPT_compile_MI50_with_tests.sh
```

Useful overrides:

```bash
BUILD_JOBS=8 codex_scripts/SCRIPT_compile_MI50_with_tests.sh
HIP_EXPORT_METRICS=ON codex_scripts/SCRIPT_compile_MI50_with_tests.sh
FA_ALL_QUANTS=ON codex_scripts/SCRIPT_compile_MI50_with_tests.sh
```

The script builds `ggml-hip`, `test-backend-ops`, and `test-backend-sampler`, which is usually enough for backend operation work without paying the full cost of the stock all-target MI50 build.
