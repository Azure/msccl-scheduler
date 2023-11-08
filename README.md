# MSCCL scheduler

MSCCL scheduler selects optimal MSCCL algorithms for MSCCL executors. It implements a static algorithm selection policy. Given a folder containing MSCCL algorithm files and collective operation requirements, this scheduler picks proper algorithms by matching different applicable conditions, including collective operation type, message size range, in-place or out-of-place, scale, etc.

## PreRequest
##### - libcurl:
    $ sudo apt-get update
    $ sudo apt-get install libcurl4-openssl-dev
##### - nlohmann json library:
    $ sudo apt-get update
    $ sudo apt-get install nlohmann-json3-dev

## Build
##### - for nccl:
    $ CXX=/path/to/nvcc BIN_HOME=/path/to/nccl/binary SRC_HOME=/path/to/nccl/source make

##### - for rccl:
    $ CXX=/path/to/hipcc BIN_HOME=/path/to/rccl/binary SRC_HOME=/path/to/rccl/source make PLATFORM=RCCL

## Install

To install MSCCL scheduler on the system, create a package then install it as root.

Debian/Ubuntu :
```shell
$ # Install tools to create debian packages
$ sudo apt install build-essential devscripts debhelper fakeroot
$ # Build NCCL deb package for nccl
$ CXX=/path/to/nvcc BIN_HOME=/path/to/nccl/binary SRC_HOME=/path/to/nccl/source make pkg.debian.build 
$ # Build NCCL deb package for rccl
$ CXX=/path/to/hipcc BIN_HOME=/path/to/rccl/binary SRC_HOME=/path/to/rccl/source make pkg.debian.build PLATFORM=RCCL
$ ls build/pkg/deb/
$ apt install build/pkg/deb/libmsccl-scheduler1_1.0.0-1+cuda._amd64.deb
```

## Usage

When running applications using MSCCL, set the following environmental variables accordingly:
1. Add path to the built binary of this scheduler to `LD_PRELOAD`.
2. Set environment variable 
   **- Nccl:** `NCCL_MSCCL_ENABLE` to 1.   
   **- Rccl:** `RCCL_MSCCL_ENABLE` to 1.
1. Set `MSCCL_ALGO_DIR` to the directory containing all MSCCL algorithm candidates.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
