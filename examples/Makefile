all: helloworld openmpi

helloworld: helloworld.cc
        nvcc -I ../build/include -L ../build/lib  helloworld.cc -o helloworld ../build/lib/libnccl.so

openmpi: openmpi.cc
        nvcc -I ../build/include -L ../build/lib  openmpi.cc -o openmpi ../build/lib/libnccl.so