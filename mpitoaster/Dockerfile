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

#RUN mkdir /var/run/sshd
#RUN echo 'root:THEPASSWORDYOUCREATED' | chpasswd
#RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd


ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

# configure user.
ENV SHELL=/bin/bash \
        NB_USER=user \
        NB_UID=1000 \
        NB_GROUP=1000 \
        NB_GID=1000
ENV HOME=/home/$NB_USER

ADD base/fix-permissions /usr/local/bin/fix-permissions
RUN chmod +x /usr/local/bin/fix-permissions
ADD base/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
RUN groupadd $NB_GROUP -g $NB_GID
RUN useradd -m -s /bin/bash -N -u $NB_UID -g $NB_GID $NB_USER && \
        echo "${NB_USER}:${NB_USER}" | chpasswd && \
        usermod -aG sudo,adm,root ${NB_USER} && \
        fix-permissions $HOME
RUN echo 'user ALL=(ALL) NOPASSWD: ALL' | sudo EDITOR='tee -a' visudo

# Default ssh config file that skips (yes/no) question when first login to the host
RUN mkdir /var/run/sshd
RUN sed -i "s/#PasswordAuthentication.*/PasswordAuthentication no/g" /etc/ssh/sshd_config \
        && sed -i "s/#PermitRootLogin.*/PermitRootLogin yes/g" /etc/ssh/sshd_config \
        && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config \
        && sed -i "s/#AuthorizedKeysFile/AuthorizedKeysFile/g" /etc/ssh/sshd_config
RUN /usr/bin/ssh-keygen -A

ENV SSHDIR $HOME/.ssh
RUN mkdir -p $SSHDIR \
        && chmod go-w $HOME/ \
        && chmod 700 $SSHDIR \
        && touch $SSHDIR/authorized_keys \
        && chmod 600 $SSHDIR/authorized_keys \
        && chown -R ${NB_USER}:${NB_GROUP} ${SSHDIR} \
        && chown -R ${NB_USER}:${NB_GROUP} /etc/ssh/*

###### switch to user and compile test example.
USER ${NB_USER}
RUN ssh-keygen -b 2048 -t rsa -f $SSHDIR/id_rsa -q -N ""
RUN cat ${SSHDIR}/*.pub >> ${SSHDIR}/authorized_keys
RUN echo "StrictHostKeyChecking no" > ${SSHDIR}/config

EXPOSE 22
USER root
RUN nvcc -I/root/.nccl/build/include/ -L/root/.nccl/build/lib -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent -I/usr/lib/x86_64-linux-gnu/openmpi/include/openmpi/opal/mca/event/libevent2022/libevent/include -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi_cxx -lmpi -o toast toast.cc /root/.nccl/build/lib/libnccl.so
USER ${NB_USER}
RUN sudo service ssh restart
