FROM nvidia/cuda:10.1-base-ubuntu18.04
# ENV PATH /jet
ENV LIB /jet
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision pymongo
RUN git clone https://github.com/uber-archive/pyflame.git
RUN cd /pyflame && ./autogen.sh
RUN cd /pyflame && ./configure
RUN cd /pyflame && make
RUN mv /pyflame/src/pyflame /usr/bin
RUN wget https://developer.nvidia.com/compute/machine-learning/nccl/secure/v2.5/prod/nccl-repo-ubuntu1804-2.5.6-ga-cuda10.2_1-1_amd64.deb
RUN dpkg -i nccl-repo-ubuntu1804-2.5.6-ga-cuda10.2_1-1_amd64.deb
RUN apt-get update -y
RUN apt install libnccl2 libnccl-dev
# RUN git clone https://github.com/facebookincubator/gloo.git
# RUN cd /gloo && mkdir build && cd build && cmake .. && make && make install
# This is to not recompile those every time
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
ADD /lib/data_partitioner.py ${LIB}/lib/data_partitioner.py
ADD /lib/parser.py ${LIB}/lib/parser.py
ADD /lib/reporter.py ${LIB}/lib/reporter.py
ADD /lib/gpu.py ${LIB}/lib/gpu.py
ADD .git ${LIB}/.git
# ENTRYPOINT [ "python", "/jet/lib/mnist.py", "--lr", "0.01" ]
EXPOSE 29500
EXPOSE 60000