all: helloworld

helloworld: helloworld.cc
	nvcc -I ../build/include -L ../build/lib  helloworld.cc -o helloworld ../build/lib/libnccl.so