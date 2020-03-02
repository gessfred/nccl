FROM nvidia/cuda:10.1-base-ubuntu18.04 
ENV LIB /pyparsa
RUN apt-get update -y 
RUN apt-get install -y --no-install-recommends python3 python3-virtualenv build-essential python3-dev iproute2 procps git cmake autoconf automake autotools-dev g++ pkg-config libtool git wget nvidia-cuda-toolkit libopenmpi-dev openmpi-bin libhdf5-openmpi-dev
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m virtualenv --python=/usr/bin/python3 $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN curl -O https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
RUN bash Anaconda3-2019.10-Linux-x86_64.sh 
#RUN pip install numpy torch torchvision pymongo mpi4py
RUN mkdir /usr/local/cuda/bin
RUN ln -s /usr/bin/nvcc /usr/local/cuda/bin/nvcc
ADD /pyparsa ${LIB}/pyparsa
RUN make -j -C ${LIB}/pyparsa/nccl src.build
RUN cd ${LIB}/pyparsa/nccl && python setup.py install
EXPOSE 29500
EXPOSE 60000