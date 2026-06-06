#!/bin/bash

# Create pretrained_models directory
mkdir -p pretrained_models

# Model list (using array instead of associative array to avoid ordering issues)
models=(
    "black-forest-labs/FLUX.1-dev:FLUX1-dev"
    "stabilityai/stable-diffusion-3-medium-diffusers:SD3-medium"
    "stabilityai/stable-diffusion-xl-base-1.0:SDXL-base"
    "Kwai-Kolors/Kolors-diffusers:Kolors"
    "THUDM/CogView4-6B:CogView4"
    "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS:PixArt"
    "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers:HunyuanDiT"
    "FoundationVision/Infinity"
    "google/flan-t5-xl"
    "Qwen/Qwen2-VL-7B-Instruct: Qwen2-VL-7B-Instruct"
    "Qwen/Qwen2-VL-2B-Instruct: Qwen2-VL-2B-Instruct"
)

# Create tmux session and set up the first window
model_info="${models[0]}"
model_path="${model_info%:*}"
window_name="${model_info#*:}"
local_dir="${model_path##*/}"

# Set the first window name directly when creating session
tmux new-session -d -s download_pretrained_model -n "$window_name"

# Give tmux some time to initialize
sleep 0.5

# Set commands for the first window
tmux send-keys -t download_pretrained_model:"$window_name" "conda activate hpsv3" Enter
tmux send-keys -t download_pretrained_model:"$window_name" "export HF_ENDPOINT=https://alpha.hf-mirror.com" Enter
tmux send-keys -t download_pretrained_model:"$window_name" "cd pretrained_models" Enter
tmux send-keys -t download_pretrained_model:"$window_name" "while true; do huggingface-cli download $model_path --local-dir $local_dir && break || sleep 60; done" Enter

# Create new windows for remaining models
for i in $(seq 1 $((${#models[@]} - 1))); do
    model_info="${models[$i]}"
    model_path="${model_info%:*}"
    window_name="${model_info#*:}"
    local_dir="${model_path##*/}"
    
    # Create new window
    tmux new-window -t download_pretrained_model -n "$window_name"
    # Add small delay to ensure window creation is complete
    sleep 0.2
    tmux send-keys -t download_pretrained_model:"$window_name" "conda activate hpsv3" Enter
    tmux send-keys -t download_pretrained_model:"$window_name" "export HF_ENDPOINT=https://alpha.hf-mirror.com" Enter
    tmux send-keys -t download_pretrained_model:"$window_name" "cd pretrained_models" Enter
    tmux send-keys -t download_pretrained_model:"$window_name" "while true; do huggingface-cli download $model_path --local-dir $local_dir && break || sleep 60; done" Enter
done
# Switch to the first window (using the first model's window name)
first_window_name="${models[0]#*:}"
tmux select-window -t download_pretrained_model:"$first_window_name"

echo "Created tmux session 'download_pretrained_model' and started downloading all models"
echo "Use 'tmux attach -t download_pretrained_model' to view download progress"
echo "Use Ctrl+B then press number keys to switch between different download windows"
echo "Use 'tmux list-windows -t download_pretrained_model' to view all windows"
echo "Use 'tmux kill-session -t download_pretrained_model' to end the session"