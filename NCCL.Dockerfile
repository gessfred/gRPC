FROM nvidia/cuda:10.1-base-ubuntu18.04
ENV LIB /parsa
RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim nvidia-cuda-toolkit
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision
# install NCCL
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
RUN mkdir $HOME/.nccl
RUN git clone https://github.com/NVIDIA/nccl.git ${HOME}/.nccl
RUN make -C ${HOME}/.nccl -j src.build
RUN apt install -y devscripts debhelper fakeroot
RUN make -C ${HOME}/.nccl pkg.debian.build
RUN apt-get install -y curl

WORKDIR /coltrain/nccl
ADD nccl/ /coltrain/nccl