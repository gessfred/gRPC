FROM ubuntu
MAINTAINER Frédéric Gessler (frederic.gessler@epfl.ch)
#install go
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get -y install wget git
RUN wget https://dl.google.com/go/go1.12.7.linux-amd64.tar.gz
RUN tar -xvf go1.12.7.linux-amd64.tar.gz
RUN rm go1.12.7.linux-amd64.tar.gz
RUN mv go /usr/local
ENV GOROOT=/usr/local/go
RUN mkdir /go
ENV GOPATH=/go
ENV PATH=$GOPATH/bin:$GOROOT/bin:$PATH
#install perf
RUN apt -y install linux-tools-common gawk
#install grpc
RUN go get -u google.golang.org/grpc
#protoc plugin
RUN go get -u github.com/golang/protobuf/protoc-gen-go 
ENV PATH=$PATH:$GOPATH/bin
#install VTune
RUN wget -O vtune.tar.gz http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15828/vtune_amplifier_2019_update6.tar.gz
RUN tar -xvf vtune.tar.gz
RUN rm vtune.tar.gz
ADD silent.cfg /
RUN apt-get install -y libgtk-3-0 libasound2 libxss1 libnss3 xserver-xorg
RUN apt-get install -y cpio
RUN apt install build-essential
RUN apt-get install linux-source linux-headers-generic -y
RUN apt-get install 
RUN vtune_amplifier_2019_update6/install.sh -s silent.cfg
RUN source /opt/intel/vtune_amplifier/amplxe-vars.sh
RUN go build -o server /go/src/google.golang.org/grpc/examples/helloworld/greeter_server/main.go
ENV AMPLXE_RUNTOOL_OPTIONS="--profiling-signal 32"
ENV AMPLXE_RUNTOOL_OPTIONS="--no-altstack"
# RUN source /opt/intel/vtune_amplifier/amplxe-vars.sh
# RUN amplxe-cl -collect hotspots --run-pass-thru=--profiling-signal=111 -result-dir r001hs ./server
# RUN amplxe-cl --collect hotspots --run-pass-thru=--no-altstack --result-dir sl ./sleep
# docker exec hello go run /go/src/google.golang.org/grpc/examples/helloworld/greeter_client/main.go