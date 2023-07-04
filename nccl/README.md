# MSCCL scheduler for NVIDIA NCCL

This is the MSCCL scheduler for the MSCCL executor in NVIDIA NCCL. It implements a static algorithm selection policy. Given a folder containing MSCCL algorithm files and collective operation requirements, this scheduler picks proper algorithms by matching different applicable conditions, including collective operation type, message size range, in-place or out-of-place, scale, etc.

## Build

    $ CXX=/path/to/nvcc NCCL_BIN_HOME=/path/to/nccl/binary NCCL_SRC_HOME=/path/to/nccl/source make

## Usage

When running applications using MSCCL, set the following environmental variables accordingly:
1. Add path to the built binary of this scheduler to `LD_PRELOAD`.
2. Set `NCCL_MSCCL_ENABLE` to 1.
3. Set `MSCCL_ALGO_DIR` to the directory containing all MSCCL algorithm candidates.
