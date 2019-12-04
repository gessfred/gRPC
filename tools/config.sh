#!/bin/bash
#
apt-get -y upgrade
apt-get -y update

sudo apt-get install -y wget git vim libgtk-3-0 libasound2 libxss1 libnss3 xorg openbox python3-pip
sudo apt-get install -y "linux-tools-$(uname -r)" "linux-cloud-tools-$(uname -r)"

pip3 install virtualenv
cd /home && virtualenv sgdq
source /home/sgdq/bin/activate
pip3 install torch torchvision

git clone https://github.com/gessfred/Peerster.git

#wget -O vtune.tar.gz http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15828/vtune_amplifier_2019_update6.tar.gz
#tar -xvf vtune.tar.gz
#rm vtune.tar.gz
apt-get install -y libgtk-3-0 libasound2 libxss1 libnss3 xserver-xorg
apt-get install linux-source linux-headers-generic -y
#perf record python run.py
#perf report