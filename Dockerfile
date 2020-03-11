FROM nvidia/cuda:10.1-base-ubuntu18.04
ENV LIB /pyparsa

RUN apt-get update 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget nvidia-cuda-toolkit openmpi-bin openmpi-common openssh-client openssh-server libopenmpi-dev
# install openMPI: from epfml/IamTao
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision pymongo mpi4py yaml
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
ADD lib/microbenchmarkgloo.py ${LIB}/lib/microbenchmarkgloo.py
ADD lib/microbenchmarknccl.py ${LIB}/lib/microbenchmarknccl.py
ADD lib/microbenchmarkcomm.py ${LIB}/lib/microbenchmarkcomm.py
ADD .git ${LIB}/.git
ADD lib/timer.py ${LIB}/lib/timer.py
RUN git clone https://github.com/gessfred/LocalSGD-Code.git
RUN pip install numpy pyyaml scipy ipython mkl mkl-include soumith magma-cuda100 pytelegraf pymongo influxdb kubernetes jinja2 lmdb tensorboard_logger pyarrow msgpack msgpack_numpy mpi4py pandas
#ADD /lib/ncccl ${LIB}/lib/ncccl
#RUN cd ${LIB}/lib/ncccl && python setup.py install
EXPOSE 29500
EXPOSE 60000
