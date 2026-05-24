#!/bin/bash
# shellcheck disable=SC1143,SC2215
cat << 'EOF'

   ██╗     ██╗      █████╗ ███╗   ███╗ █████╗    ██████╗██████╗ ██████╗
   ██║     ██║     ██╔══██╗████╗ ████║██╔══██╗  ██╔════╝██╔══██╗██╔══██╗
   ██║     ██║     ███████║██╔████╔██║███████║  ██║     ██████╔╝██████╔╝
   ██║     ██║     ██╔══██║██║╚██╔╝██║██╔══██║  ██║     ██╔═══╝ ██╔═══╝
   ███████╗███████╗██║  ██║██║ ╚═╝ ██║██║  ██║  ╚██████╗██║     ██║
   ╚══════╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═════╝╚═╝     ╚═╝
            ██████╗ ███████╗██╗  ██╗ █████╗  ██████╗  ██████╗
           ██╔════╝ ██╔════╝╚██╗██╔╝██╔══██╗██╔═████╗██╔════╝
           ██║  ███╗█████╗   ╚███╔╝ ╚██████║██║██╔██║███████╗
           ██║   ██║██╔══╝   ██╔██╗  ╚═══██║████╔╝██║██╔═══██╗
           ╚██████╔╝██║     ██╔╝ ██╗ █████╔╝╚██████╔╝╚██████╔╝
            ╚═════╝ ╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚═════╝  ╚═════╝            


EOF

export HSA_OVERRIDE_GFX_VERSION=9.0.6
export HIP_VISIBLE_DEVICES=0,1
export CUDA_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1
export GGML_BACKEND_HIP=1
export HCC_AMDGPU_TARGET=gfx906

export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export KMP_SETTINGS=1


# Model path 
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/cerebras_Qwen3-Coder-REAP-25B-A3B-Q8_0.gguf"
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/Qwen3-VL-30B-A3B-Thinking-Q4_1.gguf"
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/Qwen3-4B-Instruct-2507-Q4_1.gguf"
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/openai_gpt-oss-20b-MXFP4.gguf"
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/GLM-4.7-Flash-Q4_1.gguf"
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/Qwen3-4B-Instruct-2507-Q8_0.gguf"
MODEL_PATH="/media/iacoppbk/80F42C9BF42C96061/llms/Ministral-3-14B-Reasoning-2512-Q8_0.gguf"
MODEL_PATH="/path/..."
MODEL_PATH="/AI/Models/Qwen3.6-27B/gguf/output-mtp.gguf"




# Vision projector path (uncomment for multimodal models)
#MMPROJ_PATH="/path/..."

# Model path .................. -m
# Vision projector ............ --mmproj
# GPU layers (99 = all) ....... -ngl
# Flash attention ............. -fa
# KV cache key type ........... -ctk
# KV cache value type ......... -ctv
# Listen interface ............ --host
# Server port ................. --port
# Context size (tokens) ....... -c
# Jinja templating ............ --jinja

#ngram settings:
#        --spec-type ngram-mod \
#        --spec-ngram-size-n 24 \
#        --draft-min 48 \
#        --draft-max 64 \


./build/bin/llama-server \
    --spec-type draft-mtp \
    --spec-draft-n-max 2 \
    -m "$MODEL_PATH" \
    -ngl 99 \
    -fa on \
    -ctk q8_0\
    -ctv f16 \
    --host 0.0.0.0 \
    --port 8080 \
    -c 262144 \
    -b 2048 \
    -ub 2048 \
  --chat-template-file "/AI/Models/Qwen3.6-27B/gguf/template.jinja" \
  --metrics \
    --jinja
    # --mmproj "$MMPROJ_PATH"
    
    
    
