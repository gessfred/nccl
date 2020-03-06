#include "enqueue.h"
#include <iostream>

//typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyArgs*);
ncclResult_t sendStub(struct ncclProxyArgs* args) {
    return ncclSuccess;
}

ncclResult_t recvStub(struct ncclProxyArgs* args) {
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclSend, const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclSend(const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclSend"); 
    struct CollectiveArgs args;
    struct ncclInfo info;
    info.sendbuff = sendbuff;
    info.comm = comm;
    info.stream = stream;
    info.chunkSteps = BROADCAST_CHUNKSTEPS;
    info.sliceSteps = BROADCAST_SLICESTEPS;
    info.count = count;
    ncclEnqueue(&info, sendStub);
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecv, const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclRecv"); 
    struct CollectiveArgs args;
    struct ncclInfo info;
    info.recvbuff = recvbuff;
    info.comm = comm;
    info.stream = stream;
    info.chunkSteps = BROADCAST_CHUNKSTEPS;
    info.sliceSteps = BROADCAST_SLICESTEPS;
    info.count = count;
    ncclEnqueue(&info, recvStub);
    return ncclSuccess;
}
