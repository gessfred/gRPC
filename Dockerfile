FROM ubuntu
#ENV PATH /jet
ENV LIB /jet
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install numpy torch torchvision
RUN apt-get install -y git cmake
RUN git clone https://github.com/facebookincubator/gloo.git
RUN cd /gloo && mkdir build && cd build && cmake .. && make && make install
COPY . ${LIB}
RUN cd ${LIB}/lib/q_cpp_extension/ && python setup.py install
RUN cd ${LIB}/lib/q_par_cpp_extension/ && python setup.py install
ENTRYPOINT [ "python", "/jet/lib/mnist.py" ]
EXPOSE 29500
EXPOSE 60000