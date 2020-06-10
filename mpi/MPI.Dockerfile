FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-devel
WORKDIR /.mpi
# install openMPI
RUN wget https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz
RUN gunzip -c openmpi-3.0.0.tar.gz | tar xf - \
    && cd openmpi-3.0.0 \
    && ./configure --prefix=$HOME/.openmpi/ --with-cuda \
    && make all install

ENV PATH /.mpi/bin:$PATH
ENV LD_LIBRARY_PATH /.mpi/lib:$LD_LIBRARY_PATH