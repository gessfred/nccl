FROM nvidia/cuda:10.1-base-ubuntu18.04
RUN apt-get update -y
RUN apt-get install -y build-essential nvidia-cuda-toolkit git 

ADD . /nccl
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc 
RUN make -C /nccl -j src.build
