FROM nvidia/cuda:10.2-cudnn7-devel
ENV LIB /parsa
RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim graphviz devscripts debhelper fakeroot curl libboost-all-dev sudo valgrind openssh-client openssh-server ssh
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision gprof2dot
# install NCCL
ENV BRANCH p2p
ADD https://api.github.com/repos/nvidia/nccl/git/refs/heads/${BRANCH} versiongrpc.json
RUN git clone https://github.com/nvidia/nccl.git  /root/.nccl
WORKDIR /root/.nccl
RUN git checkout ${BRANCH}
RUN make -j src.build 
ADD native/ /.native/
WORKDIR /.native
RUN python setup.py install
