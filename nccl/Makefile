all: exp1

exp1: exp1.cc
	nvcc -I ../build/include -L ../build/lib  exp1.cc -o exp1 ../build/lib/libnccl.so