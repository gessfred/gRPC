all: exp1

exp1: exp1.cc
	nvcc -pg -I ../build/include -L ../build/lib  exp1.cc -o exp1 ../build/lib/libnccl.so