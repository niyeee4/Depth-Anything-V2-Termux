#!/data/data/com.termux/files/usr/bin/bash

termux-setup-storage

cd ~

export DEBIAN_FRONTEND=noninteractive

apt update -y
apt upgrade -y -o Dpkg::Options::="--force-confold"

apt install -y x11-repo

apt install -y git wget termux-api file ffmpeg python-numpy opencv-python python-opencv-python python-pillow onnxruntime python-onnxruntime

git clone https://github.com/niyeee4/Depth-Anything-V2-Termux

cd Depth-Anything-V2-Termux

mkdir -p $PREFIX/bin
cp depthmap $PREFIX/bin/

chmod +x $PREFIX/bin/depthmap

echo -e "type '\e[32mdepthmap\e[0m' for depth maps"