FROM nvidia/cuda:10.1-base-ubuntu18.04
ENV LIB /pyparsa
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget nvidia-cuda-toolkit openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev
# install openMPI: from epfml/IamTao
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision pymongo mpi4py
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
ADD /lib/nccl/src ${LIB}/lib/nccl/src
ADD /lib/nccl/ext-net ${LIB}/lib/nccl/ext-net
ADD /lib/nccl/makefiles ${LIB}/lib/nccl/makefiles
ADD /lib/nccl/pkg ${LIB}/lib/nccl/pkg
ADD /lib/nccl/src ${LIB}/lib/nccl/src
ADD /lib/nccl/Makefile ${LIB}/lib/nccl/Makefile
RUN make -j -C ${LIB}/lib/nccl src.build
ADD /lib/nccl/collectives.cc ${LIB}/lib/nccl/collectives.cc
ADD /lib/nccl/setup.py ${LIB}/lib/nccl/setup.py
RUN cd ${LIB}/lib/nccl && python setup.py install
ADD /lib/test_allgather.py ${LIB}/lib/test_allgather.py
ENV LD_LIBRARY_PATH "${LIB}/lib/nccl/build/lib:${LD_LIBRARY_PATH}"
ENV LD_PRELOAD "${LIB}/lib/nccl/build/lib/libnccl.so:$LD_PRELOAD"
RUN git clone https://github.com/gessfred/LocalSGD-Code.git
EXPOSE 29500
EXPOSE 60000
