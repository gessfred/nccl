#include "enqueue.h"
#include <iostream>

//typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyArgs*);
ncclResult_t sendStub(struct ncclProxyArgs* args) {
    return ncclSuccess;
}

ncclResult_t recvStub(struct ncclProxyArgs* args) {
    return ncclSuccess;
}


__global__ void ncclSendKernel(struct CollectiveArgs* args) {
    int dst = 1;
    const int tid = threadIdx.x;
    struct CollectiveArgs* args;
    const int nthreads = args->nThreads-WARP_SIZE;
    const int bid = args->bid;
    struct ncclDevComm* comm = args->comm;
    struct ncclChannel* channel = comm->channels+blockIdx.x;
    struct ncclRing* ring = &channel->ring;
    const ssize_t size = args->N;
    const int nranks = comm->nRanks;
    const int stepSize = channel->buffSize / (sizeof(T)*NCCL_STEPS);
    const int chunkSize = stepSize * ALLGATHER_CHUNKSTEPS;
    const T * __restrict__ thisInput = (const T*)args->ThisInput;
    /***********************IMPORTANT**************************/
    offset = chunkOffset + dst * size;
    ncclPrimitives<UNROLL, ALLGATHER_CHUNKSTEPS/ALLGATHER_SLICESTEPS, ALLGATHER_SLICESTEPS, T, 1, 1, FUNC> prims();
    prims.directSend(thisInput, 0, size);
}

NCCL_API(ncclResult_t, ncclSend, const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t* comm, cudaStream_t stream);
ncclResult_t  ncclSend(const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t* comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclSend"); 
    struct CollectiveArgs args;
    struct ncclInfo info;
    info.sendbuff = sendbuff;
    info.comm = comm;
    info.stream = stream;
    info.chunkSteps = BROADCAST_CHUNKSTEPS;
    info.sliceSteps = BROADCAST_SLICESTEPS;
    info.count = count;
    //ncclEnqueue(&info, (void*)sendStub);
    comm->myParams->func = ncclSendKernel;
    ncclExec(comm);
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecv, const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclRecv"); 
    struct CollectiveArgs args;
    struct ncclInfo info;
    //info.recvbuff = recvbuff;
    info.comm = comm;
    info.stream = stream;
    info.chunkSteps = BROADCAST_CHUNKSTEPS;
    info.sliceSteps = BROADCAST_SLICESTEPS;
    info.count = count;
    ncclEnqueue(&info, (void*)recvStub);
    return ncclSuccess;
}
