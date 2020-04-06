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
RUN git clone https://github.com/NVIDIA/nccl.git /root/.nccl
WORKDIR /root/.nccl
RUN make -j src.build 
RUN make pkg.debian.build
ADD nccl/ experiments
WORKDIR /root/.nccl/experiments
RUN make