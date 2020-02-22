FROM ubuntu
#ENV PATH /jet
RUN apt-get update && apt-get install -y --no-install-recommends \
gnupg2 curl ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get purge --autoremove -y curl && \
rm -rf /var/lib/apt/lists/*
ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2=$CUDA_VERSION-1
# For libraries in the cuda-compat-* package: https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDA_PKG_VERSION \
cuda-compat-10-2 && \
ln -s cuda-10.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*
# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
ENV LIB /jet
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision
#RUN git clone https://github.com/facebookincubator/gloo.git
#RUN cd /gloo && mkdir build && cd build && cmake .. && make && make install
# This is to not recompile those every time
ADD /lib/q_cpp_extension ${LIB}/lib/q_cpp_extension
ADD /lib/q_par_cpp_extension ${LIB}/lib/q_par_cpp_extension
RUN cd ${LIB}/lib/q_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_par_cpp_extension/ && python setup.py install
ADD /lib/all_reduce.py ${LIB}/lib/all_reduce.py
ADD /lib/distributed_sgd.py ${LIB}/lib/distributed_sgd.py
ADD /lib/mnist.py ${LIB}/lib/mnist.py
ADD /lib/quantizy.py ${LIB}/lib/quantizy.py
ADD /lib/benchmark.py ${LIB}/lib/benchmark.py
ENTRYPOINT [ "python", "/jet/lib/mnist.py", "--lr", "0.01" ]
EXPOSE 29500
EXPOSE 60000