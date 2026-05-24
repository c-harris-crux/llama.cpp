#!/usr/bin/env bash
set -euo pipefail

# Codex-oriented MI50/gfx906 build:
# - keeps logs separate from the user's normal compilation_log.txt
# - avoids HIP kernel metrics noise unless explicitly requested
# - avoids all-quant flash-attention template expansion by default
# - builds enough test binaries for backend/op validation

if [[ ! -f "CMakeLists.txt" ]]; then
    echo "Error: run this from the llama.cpp repository root" >&2
    exit 1
fi

export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export HIP_PATH=${HIP_PATH:-$ROCM_PATH}
export HIP_PLATFORM=${HIP_PLATFORM:-amd}
export HIP_CLANG_PATH=${HIP_CLANG_PATH:-$ROCM_PATH/llvm/bin}
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/llvm/lib:${LD_LIBRARY_PATH:-}"

if [[ -z "${AMDGPU_ARCH:-}" ]]; then
    if command -v amdgpu-arch >/dev/null 2>&1; then
        AMDGPU_ARCH=$(amdgpu-arch | head -n 1)
    else
        AMDGPU_ARCH=gfx906
    fi
fi

BUILD_DIR=${BUILD_DIR:-build-codex-mi50}
LOG_FILE=${LOG_FILE:-codex_compilation_log.txt}
BUILD_JOBS=${BUILD_JOBS:-$(nproc)}
CCACHE_DIR=${CCACHE_DIR:-/tmp/codex-ccache}
HIP_EXPORT_METRICS=${HIP_EXPORT_METRICS:-OFF}
FA_ALL_QUANTS=${FA_ALL_QUANTS:-OFF}

export CCACHE_DIR
mkdir -p "$BUILD_DIR" "$CCACHE_DIR"

{
    echo "Codex MI50 build"
    echo "  build dir:       $BUILD_DIR"
    echo "  log file:        $LOG_FILE"
    echo "  arch:            $AMDGPU_ARCH"
    echo "  jobs:            $BUILD_JOBS"
    echo "  ccache dir:      $CCACHE_DIR"
    echo "  HIP metrics:     $HIP_EXPORT_METRICS"
    echo "  FA all quants:   $FA_ALL_QUANTS"
    echo

    cmake -S . -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_COMPILER="$ROCM_PATH/llvm/bin/clang" \
        -DCMAKE_CXX_COMPILER="$ROCM_PATH/llvm/bin/clang++" \
        -DCMAKE_HIP_ARCHITECTURES="$AMDGPU_ARCH" \
        -DCMAKE_HIP_COMPILER_FORCED=1 \
        -DCMAKE_C_FLAGS="-O3 -march=native -mtune=native -DNDEBUG -ffast-math -fno-finite-math-only -ffp-contract=fast" \
        -DCMAKE_CXX_FLAGS="-O3 -march=native -mtune=native -DNDEBUG" \
        -DCMAKE_HIP_FLAGS="-Wno-ignored-attributes -Wno-cuda-compat -Wno-unused-result" \
        -DGGML_CCACHE=ON \
        -DGGML_HIP=ON \
        -DGGML_HIP_GRAPHS=ON \
        -DGGML_HIP_NO_VMM=ON \
        -DGGML_HIP_EXPORT_METRICS="$HIP_EXPORT_METRICS" \
        -DGGML_NATIVE=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS="$FA_ALL_QUANTS" \
        -DGGML_CUDA_FORCE_MMQ=OFF \
        -DGGML_CUDA_FORCE_CUBLAS=OFF \
        -DGGML_CUDA_NO_PEER_COPY=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=ON \
        -DLLAMA_CURL=ON \
        -DLLAMA_STATIC=OFF

    cmake --build "$BUILD_DIR" --target ggml-hip test-backend-ops test-backend-sampler -j "$BUILD_JOBS"

    echo
    echo "Build complete:"
    echo "  $BUILD_DIR/bin/libggml-hip.so"
    echo "  $BUILD_DIR/bin/test-backend-ops"
    echo "  $BUILD_DIR/bin/test-backend-sampler"
    echo
    echo "Useful follow-up checks:"
    echo "  $BUILD_DIR/bin/test-backend-ops support -b HIP -o TOP_K"
    echo "  $BUILD_DIR/bin/test-backend-ops test -b HIP -o TOP_K"
} 2>&1 | tee "$LOG_FILE"
