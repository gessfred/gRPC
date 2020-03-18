<<<<<<< HEAD:Dockerfile
FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget nvidia-cuda-toolkit libopenmpi-dev openmpi-bin libhdf5-openmpi-dev
ENV HOME=/home
ENV LIB /pyparsa
# install openMPI
RUN mkdir $HOME/.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH
# install NCCL
RUN mkdir $HOME/.nccl
RUN git clone https://github.com/NVIDIA/nccl.git ${HOME}/.nccl
RUN make -C ${HOME}/.nccl -j src.build
RUN apt install -y devscripts debhelper fakeroot
RUN make -C ${HOME}/.nccl pkg.debian.build
RUN apt-get install -y curl
WORKDIR $HOME
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
RUN $HOME/conda/bin/conda install numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing

# install pytorch, torchvision, torchtext.
RUN git clone --recursive  https://github.com/pytorch/pytorch
RUN cd pytorch && \
    git checkout tags/v1.3.0 && \
    git submodule sync && \
    git submodule update --init --recursive && \
    TORCH_CUDA_ARCH_LIST="3.5 3.7 5.2 6.0 6.1 7.0+PTX" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
    CMAKE_PREFIX_PATH="$(dirname $(which $HOME/conda/bin/conda))/../" \
    python setup.py install 
#instead of pip install . -v
RUN git clone https://github.com/pytorch/vision.git && cd vision && git checkout v0.4.0 && python setup.py install
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install --upgrade git+https://github.com/pytorch/text
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install spacy
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download en
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/python -m spacy download de
# install other python related softwares.
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y opencv protobuf
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y networkx
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c anaconda pandas
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -y -c conda-forge tabulate
RUN $HOME/conda/envs/pytorch-py$PYTHON_VERSION/bin/pip install lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py
RUN $HOME/conda/bin/conda install --name pytorch-py$PYTHON_VERSION -c conda-forge python-blosc
RUN $HOME/conda/bin/conda clean -ya

ADD /pyparsa ${LIB}/pyparsa
RUN make -j -C ${LIB}/pyparsa/nccl src.build
RUN cd ${LIB}/pyparsa/nccl && python setup.py install
EXPOSE 29500
EXPOSE 60000
=======
FROM nvidia/cuda:10.1-base-ubuntu18.04
ENV LIB /jet
RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision

ADD /lib/q_cpp_extension ${LIB}/lib/q_cpp_extension
ADD /lib/q_par_cpp_extension ${LIB}/lib/q_par_cpp_extension
ADD /lib/q_general_cpp_extension ${LIB}/lib/q_general_cpp_extension
RUN cd ${LIB}/lib/q_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_par_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_general_cpp_extension/ && python setup.py install
ADD /lib/all_reduce.py ${LIB}/lib/all_reduce.py
ADD /lib/distributed_sgd.py ${LIB}/lib/distributed_sgd.py
ADD /lib/mnist.py ${LIB}/lib/mnist.py
ADD /lib/quantizy.py ${LIB}/lib/quantizy.py
ADD /lib/benchmark.py ${LIB}/lib/benchmark.py
ADD /lib/tests ${LIB}/lib/tests
>>>>>>> eric/gpu_reduction:gpuTest.Dockerfile
