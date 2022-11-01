#!/bin/sh

device=false
current_dir="$(pwd -P)"

check_requirements() {
    case $(uname -s) in
        Darwin)
            if [ "$(uname -m)" = "arm64" ]; then
                printf "macOS (Apple Silicon) system detected.\n"
                device="osx-arm64"
            else
                printf "macOS (Intel) system detected.\n"
                export CFLAGS='-stdlib=libc++'
                device="osx-64"
            fi
            ;;
        Linux)
            printf "Linux system detected.\n"
            device="linux-64"
            ;;
        *)
            printf "Only Linux and macOS are currently supported.\n"
            exit 1
            ;;
    esac
}

install_tensorflow() {
    printf "\nInstalling tensorflow...\n"
    case $device in
        osx-arm64)
            conda install -c apple tensorflow-deps -y
            pip install tensorflow-macos==2.10.0 tensorflow-metal
            ;;
        osx-64)
            pip install tensorflow==2.10.0
            ;;
        linux-64)
            conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1 nccl -y
            pip install tensorflow==2.10.0
            ;;
    esac
}

install_packages() {
    printf "\nInstalling Python packages...\n"
    pip install -e .
}

set_python_path() {
    if [ $device = "linux-64" ]; then
        conda env config vars set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"$CONDA_PREFIX/lib"
        printf "\nPlease restart conda environment"
    fi
}

check_requirements
install_tensorflow
install_packages
set_python_path

printf '\n\nSetup completed.\n'
