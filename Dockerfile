FROM nvidia/cuda:10.1-base-ubuntu18.04 AS pytorch
ENV LIB /pyparsa
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget 
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision pymongo

FROM pytorch AS pyflame
RUN git clone https://github.com/uber-archive/pyflame.git
RUN cd /pyflame && ./autogen.sh
RUN cd /pyflame && ./configure
RUN cd /pyflame && make
RUN mv /pyflame/src/pyflame /usr/bin

FROM pyflame AS nccl
RUN apt-get install -y nvidia-cuda-toolkit
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
ADD /lib/nccl ${LIB}/lib/nccl
RUN make -C ${LIB}/lib/nccl -j src.build
RUN cd ${LIB}/lib/nccl && python setup.py install

FROM nccl 
ADD /lib ${LIB}/lib
RUN cd ${LIB}/lib/q_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_par_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_general_cpp_extension/ && python setup.py install
EXPOSE 29500
EXPOSE 60000
