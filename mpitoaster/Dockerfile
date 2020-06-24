FROM nvidia/cuda:10.1-cudnn7-devel
# install openMPI
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt-get install --no-install-recommends -y wget python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake vim graphviz devscripts debhelper fakeroot curl libboost-all-dev sudo valgrind openssh-client openssh-server ssh iperf3 iputils-ping nload iperf3 net-tools openmpi-bin libopenmpi-dev sudo openssh-server build-essential devscripts debhelper fakeroot
#RUN mkdir $HOME/.openmpi/
#RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
#RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
#    && cd openmpi-3.0.0 \
#    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
#    && make all install

#ENV PATH $HOME/.openmpi/bin:$PATH
#ENV LD_LIBRARY_PATH $HOME/.openmpi/lib:$LD_LIBRARY_PATH

ENV LIB /parsa
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
RUN make pkg.debian.build
RUN dpkg -i build/pkg/deb/libnccl2_2.7.0p4-1+cuda10.1_amd64.deb
RUN dpkg -i build/pkg/deb/libnccl-dev_2.7.0p4-1+cuda10.1_amd64.deb
RUN apt-get update -y && apt-get install -y 
ADD . /.mpitoaster/
WORKDIR /.mpitoaster
EXPOSE 29500
EXPOSE 29501
EXPOSE 60000
EXPOSE 5201
ENV MASTER_ADD=10.90.40.14
ENV MASTER_PORT=29500
ENV WORLD_SIZE=4

RUN mkdir /var/run/sshd
RUN echo 'root:THEPASSWORDYOUCREATED' | chpasswd
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd


ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

EXPOSE 22
RUN nvcc -I/root/.nccl/build/include/ -L/root/.nccl/build/lib -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -o toast toast.cc /root/.nccl/build/lib/libnccl.so