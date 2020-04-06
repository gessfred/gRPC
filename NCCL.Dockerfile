FROM nvidia/cuda:10.2-cudnn7-devel
ENV LIB /parsa
RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim graphviz devscripts debhelper fakeroot curl libboost-all-dev sudo
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision gprof2dot
# install NCCL
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
RUN git clone https://github.com/NVIDIA/nccl.git /root/.nccl
WORKDIR /root/.nccl
RUN make -j src.build CXXFLAGS=-pg
RUN make pkg.debian.build
RUN sudo apt install /root/.nccl/build/pkg/deb/libnccl2_2.6.4-1+cuda9.1_amd64.deb
RUN sudo apt install /root/.nccl/build/pkg/deb/libnccl-dev_2.6.4-1+cuda9.1_amd64.deb
ADD nccl/ experiments
WORKDIR /root/.nccl/experiments
RUN make