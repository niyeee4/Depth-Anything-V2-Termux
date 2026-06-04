#!/data/data/com.termux/files/usr/bin/bash

termux-setup-storage

cd ~

export DEBIAN_FRONTEND=noninteractive

apt update -y
apt upgrade -y -o Dpkg::Options::="--force-confold"

apt install -y x11-repo

apt install -y dbus git wget termux-api file ffmpeg python-numpy opencv-python python-opencv-python python-pillow onnxruntime python-onnxruntime

git clone https://github.com/niyeee4/Depth-Anything-V2-Termux

cd "$HOME/Depth-Anything-V2-Termux" || exit 1

mkdir -p "$PREFIX/bin"
cp depthmap "$PREFIX/bin/"

chmod +x "$PREFIX/bin/depthmap"

echo
while true; do
printf "Install custom font? (y/n): "
read -r choice < /dev/tty

case "$choice" in
    [Yy])
        mkdir -p "$HOME/.termux"
        cp "$HOME/Depth-Anything-V2-Termux/font.ttf" "$HOME/.termux/font.ttf"
        termux-reload-settings >/dev/null 2>&1
        echo "Font installed."
        break
        ;;
    [Nn])
        break
        ;;
esac

done
yes | termux-setup-storage
clear
echo -e "type '\e[32mdepthmap\e[0m' for depth maps"
