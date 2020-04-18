FROM nvidia/cuda:10.2-cudnn7-devel
ENV LIB /parsa
RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim graphviz devscripts debhelper fakeroot curl libboost-all-dev sudo valgrind openssh-client openssh-server ssh
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision gprof2dot
WORKDIR /root/.vtune 
#install VTune
RUN apt-get update -y && apt-get install -y wget
RUN wget -O vtune.tar.gz http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15828/vtune_amplifier_2019_update6.tar.gz
RUN tar -xvf vtune.tar.gz
RUN rm vtune.tar.gz
ADD silent.cfg .
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get install -y libgtk-3-0 libasound2 libxss1 libnss3 xserver-xorg linux-source linux-headers-generic openssh-client cpio
RUN vtune_amplifier_2019_update6/install.sh -s silent.cfg
RUN touch /root/.bashrc && echo "source /opt/intel/vtune_amplifier/amplxe-vars.sh" > /root/.bashrc
ENV AMPLXE_RUNTOOL_OPTIONS="--profiling-signal 32"
# install NCCL
ENV BRANCH p2p
ADD https://api.github.com/repos/nvidia/nccl/git/refs/heads/${BRANCH} versiongrpc.json
RUN git clone https://github.com/nvidia/nccl.git  /root/.nccl
WORKDIR /root/.nccl
RUN git checkout ${BRANCH}
RUN make -j src.build 
ADD nccl/ .
RUN service ssh start
