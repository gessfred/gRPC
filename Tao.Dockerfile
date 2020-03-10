
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

# install some necessary tools.
RUN echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-get update \
        && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        pkg-config \
        software-properties-common
RUN apt-get install -y \
        inkscape \
        jed \
        libsm6 \
        libxext-dev \
        libxrender1 \
        lmodern \
        libcurl3-dev \
        libfreetype6-dev \
        libzmq3-dev \
        libcupti-dev \
        pkg-config \
        libav-tools \
        libjpeg-dev \
        libpng-dev \
        zlib1g-dev \
        locales
RUN apt-get install -y \
        sudo \
        rsync \
        cmake \
        g++ \
        swig \
        vim \
        git \
        curl \
        wget \
        unzip \
        zsh \
        git \
        screen \
        tmux \
        openssh-server
RUN apt-get update && \
        apt-get install -y pciutils net-tools iputils-ping && \
        apt-get install -y htop
RUN add-apt-repository ppa:openjdk-r/ppa \
        && apt-get update \
        && apt-get install -y \
        openjdk-7-jdk \
        openjdk-7-jre-headless

USER $NB_USER
WORKDIR $HOME

# install openMPI
RUN mkdir $HOME/.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH

# install conda
ENV PYTHON_VERSION=3.6
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    sh miniconda.sh -b -p $HOME/conda && \
    rm ~/miniconda.sh
RUN $HOME/conda/bin/conda update -n base conda
RUN $HOME/conda/bin/conda create -y --name pytorch-py$PYTHON_VERSION python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -c soumith magma-cuda100
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION scikit-learn
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install pytelegraf pymongo influxdb kubernetes jinja2
ENV PATH $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin:$PATH

# install pytorch, torchvision, torchtext.
RUN git clone --recursive  https://github.com/pytorch/pytorch
RUN cd pytorch && \
    git submodule update --init && \
    TORCH_CUDA_ARCH_LIST="3.5 3.7 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which $HOME/conda/bin/conda))/../" \
    pip install -v .
RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout v0.4.0 && python setup.py install
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install --upgrade git+https://github.com/pytorch/text
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install spacy
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download en
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download de


# install bit2byte.
RUN git clone https://github.com/tvogels/signSGD-with-Majority-Vote.git && \
    cd signSGD-with-Majority-Vote/main/bit2byte-extension/ && \
    $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python setup.py develop --user

# install other python related softwares.
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y opencv protobuf
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y networkx
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c anaconda pandas
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c conda-forge tabulate
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -c conda-forge python-blosc
RUN $HOME/conda/bin/conda clean -ya