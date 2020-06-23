FROM nvidia/cuda:10.1-cudnn7-devel
# install openMPI
RUN apt-get update -y && apt-get install -y wget
RUN mkdir $HOME/.openmpi/
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH $HOME/.openmpi/bin:$PATH
ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH

ENV LIB /parsa
RUN apt-get upgrade -y
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim graphviz devscripts debhelper fakeroot curl libboost-all-dev sudo valgrind openssh-client openssh-server ssh
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy gprof2dot
RUN pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# install NCCL
ENV BRANCH p2p
ADD https://api.github.com/repos/nvidia/nccl/git/refs/heads/${BRANCH} versiongrpc.json
RUN git clone https://github.com/nvidia/nccl.git  /root/.nccl
WORKDIR /root/.nccl
RUN git checkout ${BRANCH}
RUN make -j src.build
RUN apt-get update -y && apt-get install -y sudo
RUN apt-get update -y && apt-get install -y iperf3 iputils-ping nload
RUN sudo apt install build-essential devscripts debhelper fakeroot
RUN make pkg.debian.build
RUN dpkg -i build/pkg/deb/libnccl2_2.7.0p4-1+cuda10.1_amd64.deb
RUN dpkg -i build/pkg/deb/libnccl-dev_2.7.0p4-1+cuda10.1_amd64.deb
ADD . /.mpitoaster/
WORKDIR /.mpitoaster
RUN python setup.py install
RUN apt-get update -y && apt-get install -y iperf3 net-tools openmpi-bin libopenmpi-dev sudo
EXPOSE 29500
EXPOSE 29501
EXPOSE 60000
EXPOSE 5201
ENV MASTER_ADD=10.90.40.14
ENV MASTER_PORT=29500
ENV WORLD_SIZE=4

RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:THEPASSWORDYOUCREATED' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

ENV DEBIAN_FRONTEND=noninteractive

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22