#!/usr/bin/env bash

# 1. Enforce gfx1030 compatibility override
export HSA_OVERRIDE_GFX_VERSION=10.3.0

# 2. Target discrete GPU directly
export HIP_VISIBLE_DEVICES=0

# 3. Force ROCm/HIP to use PyTorch's internal bitcode libraries (LLVM 18 matched)
# VENV_LIB="$(pwd)/.venv/lib/python3.11/site-packages"
export DEVICE_LIB_PATH="/nix/store/sbzc6rk4319vq8jzn02c95pv3b5zbndg-rocm-device-libs-6.0.2/amdgcn/bitcode"
export HIP_DEVICE_LIB_PATH="/nix/store/sbzc6rk4319vq8jzn02c95pv3b5zbndg-rocm-device-libs-6.0.2/amdgcn/bitcode"

# 5. Run test
python3 -c '
# Test GPU tensor allocation and arithmetic execution
import torch
print("Device:", torch.cuda.get_device_name(0))
x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
print("Tensor allocated successfully:", x)
y = x * 2
print("Tensor computation result   :", y)
'
