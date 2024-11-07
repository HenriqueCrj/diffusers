#!/bin/bash

CONTROLNETDEPTH_DOWNLOADED="model/ControlNets/controlnet_depth/diffusion_pytorch_model.safetensors?download=true"
CONTROLNETDEPTH_RENAMED="model/ControlNets/controlnet_depth/diffusion_pytorch_model.safetensors"
VITONHD_DOWNLOADED="VITON-HD/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=1&dl=1"
VITONHD_RENAMED="VITON-HD/zalando-hd-resized.zip"

echo "Starting setup"

if [ ! -f "$CONTROLNETDEPTH_DOWNLOADED" ] && [ ! -f "$CONTROLNETDEPTH_RENAMED" ]; then
    echo "Downloading ControlNet Depth model"
    wget -nc "https://huggingface.co/lllyasviel/sd-controlnet-depth/resolve/main/diffusion_pytorch_model.safetensors?download=true" -P model/ControlNets/controlnet_depth/
fi
if [ -f "$CONTROLNETDEPTH_DOWNLOADED" ]; then
    mv "$CONTROLNETDEPTH_DOWNLOADED" "$CONTROLNETDEPTH_RENAMED"
fi

if [ ! -f "$VITONHD_DOWNLOADED" ] && [ ! -f "$VITONHD_RENAMED" ]; then
    echo "Downloading VITON-HD dataset"
    wget -nc "https://www.dropbox.com/scl/fi/xu08cx3fxmiwpg32yotd7/zalando-hd-resized.zip?rlkey=ks83mdv2pvmrdl2oo2bmmn69w&e=1&dl=1" -P VITON-HD
fi
if [ -f "$VITONHD_DOWNLOADED" ]; then
    mv "$VITONHD_DOWNLOADED" "$VITONHD_RENAMED"
fi

echo "Unziping VITON-HD dataset"
unzip "$VITONHD_RENAMED" -d VITON-HD

echo "Renaming test_pairs file from VITON-HD to use CatVTON inference"
mv VITON-HD/test_pairs.txt VITON-HD/test_pairs_unpaired.txt

echo "Setup finished"
