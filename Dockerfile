FROM nvidia/cuda:10.1-base-ubuntu18.04
ENV LIB /pyparsa
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget nvidia-cuda-toolkit openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev
# install openMPI: from epfml/IamTao
RUN mkdir $HOME/.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install
ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision pymongo mpi4py
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
ADD /lib ${LIB}/lib
RUN make -j -C ${LIB}/lib/nccl src.build
RUN cd ${LIB}/lib/nccl && python setup.py install
RUN cd ${LIB}/lib/q_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_par_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_general_cpp_extension/ && python setup.py install
RUN git clone https://github.com/gessfred/LocalSGD-Code.git
EXPOSE 29500
EXPOSE 60000