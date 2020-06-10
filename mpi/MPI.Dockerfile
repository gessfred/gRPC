FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel
WORKDIR /.mpi
RUN apt-get update -y && apt-get install -y wget
# install openMPI
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH /.mpi/bin:$PATH
ENV LD_LIBRARY_PATH /.mpi/lib:$LD_LIBRARY_PATH

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
EXPOSE 5201
#CMD ["/usr/sbin/sshd", "-D"]

RUN /usr/sbin/sshd 