FROM nvidia/cuda:10.1-base-ubuntu18.04
ADD . /nccl
RUN cd /nccl && make -j src.build