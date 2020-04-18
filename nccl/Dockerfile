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
ENV BRANCH eval/prof
ADD https://api.github.com/repos/gessfred/nccl/git/refs/heads/${BRANCH} versiongrpc.json
RUN git clone https://github.com/gessfred/nccl.git  /root/.nccl
WORKDIR /root/.nccl
RUN git checkout ${BRANCH}
RUN make -j src.build 
RUN make pkg.debian.build
ADD nccl/ .
RUN nvcc -g -std=c++11 -I /usr/local/cuda/include -O0 -o exp5 \
 /root/.nccl/build/obj/init.o /root/.nccl/build/obj/channel.o /root/.nccl/build/obj/bootstrap.o \
 /root/.nccl/build/obj/transport.o /root/.nccl/build/obj/enqueue.o /root/.nccl/build/obj/group.o\
 /root/.nccl/build/obj/debug.o /root/.nccl/build/obj/misc/nvmlwrap.o /root/.nccl/build/obj/misc/ibvwrap.o \
 /root/.nccl/build/obj/misc/utils.o /root/.nccl/build/obj/misc/argcheck.o /root/.nccl/build/obj/transport/p2p.o \
 /root/.nccl/build/obj/transport/shm.o /root/.nccl/build/obj/transport/net.o /root/.nccl/build/obj/transport/net_socket.o \
 /root/.nccl/build/obj/transport/net_ib.o /root/.nccl/build/obj/collectives/all_reduce.o /root/.nccl/build/obj/collectives/all_gather.o \
 /root/.nccl/build/obj/collectives/broadcast.o /root/.nccl/build/obj/collectives/reduce.o \
 /root/.nccl/build/obj/collectives/reduce_scatter.o /root/.nccl/build/obj/graph/topo.o /root/.nccl/build/obj/graph/paths.o \
 /root/.nccl/build/obj/graph/search.o /root/.nccl/build/obj/graph/connect.o /root/.nccl/build/obj/graph/rings.o \
 /root/.nccl/build/obj/graph/trees.o /root/.nccl/build/obj/graph/tuning.o \
 /root/.nccl/build/obj/collectives/device/colldevice.a   -L/usr/local/cuda/lib64 -lcudart_static -lpthread -lrt -ldl exp5.cc
RUN service ssh start
