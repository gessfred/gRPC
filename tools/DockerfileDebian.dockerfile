#based on Debian 10 amd64
FROM golang
MAINTAINER Frédéric Gessler (frederic.gessler@epfl.ch)
RUN apt-get update
RUN go get -u google.golang.org/grpc
#install perf
#RUN http://ftp.ch.debian.org/debian/pool/main/l/linux/linux-perf-4.19_4.19.67-2_amd64.deb
RUN wget -O /home/perf.deb http://ftp.ch.debian.org/debian/pool/main/l/linux-latest/linux-perf_4.19+105+deb10u1_all.deb
#protoc plugin
RUN go get -u github.com/golang/protobuf/protoc-gen-go 
RUN export PATH=$PATH:$GOPATH/bin
RUN export HELLO_WORLD=$GOPATH/src/google.golang.org/grpc/examples/helloworld
RUN export MICROSRV=$HELLO_WORLD/greeter_server
RUN export MICROCLI=$HELLO_WORLD/greeter_client

