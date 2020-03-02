FROM nvidia/cuda:10.1-base-ubuntu18.04 
ENV LIB /pyparsa
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget nvidia-cuda-toolkit 
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.2.tar.gz
RUN gunzip -c openmpi-4.0.2.tar.gz | tar xf -
RUN mkdir /usr/local/include
RUN ln -s /usr/include/cuda.h /usr/local/cuda/include/cuda.h
RUN cd openmpi-4.0.2 && ./configure --prefix=/home/$USER/.openmpi --with-cuda && make -j all install
ENV PATH="$PATH:/home/$USER/.openmpi/bin"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USER/.openmpi/lib/"
RUN pip install numpy torch torchvision pymongo mpi4py
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
ADD /lib ${LIB}/lib
RUN make -j -C ${LIB}/lib/nccl src.build
RUN cd ${LIB}/lib/nccl && python setup.py install
RUN cd ${LIB}/lib/q_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_par_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_general_cpp_extension/ && python setup.py install
EXPOSE 29500
EXPOSE 60000
#RUN apt-get install -y openmpi-bin libopenmpi-dev