#!/bin/bash
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "ColorizedNarrative Setup (V100)"
echo ""

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found${NC}"
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
echo -e "${GREEN}GPU: $GPU_NAME ($GPU_MEMORY MB)${NC}"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}Python: $(python3 --version | awk '{print $2}')${NC}"

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    source $HOME/.cargo/env 2>/dev/null || true
fi

if command -v uv &> /dev/null; then
    echo -e "${GREEN}uv: $(uv --version)${NC}"
    USE_UV=true
else
    echo -e "${YELLOW}uv not available, using pip${NC}"
    USE_UV=false
fi

mkdir -p {logs,models,data,experiments}

echo "Creating venv..."
if [ "$USE_UV" = true ]; then
    uv venv venv --python python3
else
    python3 -m venv venv
fi
source venv/bin/activate

echo "Installing numpy<2..."
if [ "$USE_UV" = true ]; then
    uv pip install "numpy<2.0"
else
    pip install "numpy<2.0"
fi

echo "Installing PyTorch (CUDA 11.8)..."
if [ "$USE_UV" = true ]; then
    uv pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
else
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
fi

python3 << 'EOF'
import torch
print(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

echo "Installing dependencies..."
DEPS=(
    "transformers>=4.36.0"
    "accelerate>=0.25.0"
    "diffusers>=0.25.0"
    "peft>=0.7.0"
    "flask>=3.0.0"
    "requests"
    "pillow"
    "gradio>=4.0"
    "pyannote.audio==4.0.3"
    "huggingface-hub"
    "safetensors"
    "sentencepiece"
    "protobuf"
    "onnxruntime-gpu>=1.16.0"
)

if [ "$USE_UV" = true ]; then
    for dep in "${DEPS[@]}"; do
        uv pip install "$dep" 2>/dev/null || true
    done
else
    pip install "${DEPS[@]}"
fi

echo -e "${GREEN}Dependencies installed${NC}"

if [ -z "$HF_TOKEN" ]; then
    echo -e "${YELLOW}HF_TOKEN not set${NC}"
    echo "Enter HuggingFace token (for pyannote):"
    read -r TOKEN
    if [ -n "$TOKEN" ]; then
        export HF_TOKEN="$TOKEN"
        echo "export HF_TOKEN='$TOKEN'" >> ~/.bashrc
    fi
fi

echo ""
echo "Downloading models..."

echo "1/3 Whisper..."
if [ -d "models/whisper/whisper_encoder" ]; then
    echo -e "${GREEN}Whisper exists${NC}"
else
    mkdir -p models/whisper
    huggingface-cli download aamocualg-hse/whisper-onnx --local-dir models/whisper 2>/dev/null || \
    huggingface-cli download openai/whisper-large-v3 --local-dir models/whisper
fi

echo "2/3 Qwen..."
if [ -f "models/qwen/model.safetensors" ]; then
    echo -e "${GREEN}Qwen exists${NC}"
else
    mkdir -p models/qwen
    huggingface-cli download Qwen/Qwen2.5-3B-Instruct --local-dir models/qwen
fi

echo "3/3 SDXL (from HuggingFace at runtime)"

echo "Caching pyannote..."
if [ -n "$HF_TOKEN" ]; then
    python3 << EOF
try:
    from pyannote.audio import Pipeline
    Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", token="$HF_TOKEN")
    print("pyannote cached")
except Exception as e:
    print(f"pyannote cache failed: {e}")
EOF
fi

cat > config_v100.env << 'EOF'
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export TORCH_CUDA_ARCH_LIST="7.0"
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export NUM_STEPS=50
export GUIDANCE_SCALE=7.5
export HEIGHT=1024
export WIDTH=1024
export MODEL_PATH_WHISPER="./models/whisper"
export MODEL_PATH_QWEN="./models/qwen"
export CONNECTOR_URL=http://localhost:5001/connector
export T2I_URL=http://localhost:5002/t2i
export LOG_LEVEL=INFO
EOF

cat > run_server.sh << 'EOF'
#!/bin/bash
set -e

if [ -f "config_v100.env" ]; then
    source config_v100.env
fi

source venv/bin/activate

if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN not set"
fi

tmux kill-session -t colorized 2>/dev/null || true

SESSION="colorized"
tmux new-session -d -s $SESSION -n t2i

tmux send-keys -t $SESSION:0 "cd $(pwd) && source venv/bin/activate && source config_v100.env" C-m
tmux send-keys -t $SESSION:0 "cd t2i && python app_local.py" C-m

sleep 3

tmux new-window -t $SESSION:1 -n connector
tmux send-keys -t $SESSION:1 "cd $(pwd) && source venv/bin/activate && source config_v100.env" C-m
tmux send-keys -t $SESSION:1 "cd connector && export MODEL_PATH='$(pwd)/models/qwen' && python app_local.py" C-m

sleep 2

tmux new-window -t $SESSION:2 -n asr
tmux send-keys -t $SESSION:2 "cd $(pwd) && source venv/bin/activate && source config_v100.env" C-m
tmux send-keys -t $SESSION:2 "cd asr && export HF_TOKEN='$HF_TOKEN' && export MODEL_PATH='$(pwd)/models/whisper' && python app_local.py" C-m

sleep 2

tmux new-window -t $SESSION:3 -n gradio
tmux send-keys -t $SESSION:3 "cd $(pwd) && source venv/bin/activate" C-m
tmux send-keys -t $SESSION:3 "cd gradio && python app.py" C-m

tmux select-window -t $SESSION:0

echo ""
echo "Started. tmux attach -t $SESSION"
echo "Windows: 0=T2I 1=Connector 2=ASR 3=Gradio"
echo "UI: http://localhost:7860"
echo ""

tmux attach -t $SESSION
EOF

chmod +x run_server.sh

cat > stop_server.sh << 'EOF'
#!/bin/bash
tmux kill-session -t colorized 2>/dev/null || echo "Not running"
EOF

chmod +x stop_server.sh

cat > test_server.sh << 'EOF'
#!/bin/bash
test_service() {
    if curl -s -f -m 5 "$2" > /dev/null 2>&1; then
        echo "$1: OK"
    else
        echo "$1: DOWN"
    fi
}

test_service "T2I" "http://localhost:5002/health"
test_service "Connector" "http://localhost:5001/health"
test_service "ASR" "http://localhost:5000/health"
test_service "Gradio" "http://localhost:7860"
EOF

chmod +x test_server.sh

echo ""
echo -e "${GREEN}Setup complete${NC}"
echo ""
echo "Start: ./run_server.sh"
echo "Stop:  ./stop_server.sh"
echo "Test:  ./test_server.sh"
echo ""
