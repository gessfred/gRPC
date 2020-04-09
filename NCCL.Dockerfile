FROM nvidia/cuda:10.2-cudnn7-devel
ENV LIB /parsa
RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim graphviz devscripts debhelper fakeroot curl libboost-all-dev sudo
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision gprof2dot
WORKDIR /root/.vtune 
#install VTune
RUN wget -O vtune.tar.gz http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15828/vtune_amplifier_2019_update6.tar.gz
RUN tar -xvf vtune_amplifier_2019_update6.tar.gz
RUN rm vtune_amplifier_2019_update6.tar.gz
ADD silent.cfg /
RUN apt-get install -y libgtk-3-0 libasound2 libxss1 libnss3 xserver-xorg linux-source linux-headers-generic
RUN vtune_amplifier_2019_update6/install.sh -s silent.cfg
RUN touch /root/.bashrc && echo "source /opt/intel/vtune_amplifier/amplxe-vars.sh" > /root/.bashrc
ENV AMPLXE_RUNTOOL_OPTIONS="--profiling-signal 32"
# install NCCL
ENV BRANCH eval/prof
ADD https://api.github.com/repos/gessfred/nccl/git/refs/heads/${BRANCH} versiongrpc.json
RUN git clone https://github.com/gessfred/nccl.git  /root/.nccl
WORKDIR /root/.nccl
RUN git checkout ${BRANCH}
RUN make -j src.build 
RUN make pkg.debian.build
ADD nccl/ experiments
WORKDIR /root/.nccl/experiments
RUN make

