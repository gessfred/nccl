#include "enqueue.h"
#include "send_recv.h"
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
    struct ncclInfo info = {
        sendbuff=sendbuff, recvbuff=recvbuff, count=count, datatype=datatype, ncclSum, comm=comm, stream=stream, /* Args */
        chunkSteps=BROADCAST_CHUNKSTEPS, sliceSteps=BROADCAST_SLICESTEPS };
    ncclEnqueueCheck(&info, sendStub);
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecv, const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclRecv"); 
    struct CollectiveArgs args;
    struct ncclInfo info = {
        sendbuff=sendbuff, recvbuff=recvbuff, count=count, datatype=datatype, ncclSum, comm=comm, stream=stream, /* Args */
        chunkSteps=BROADCAST_CHUNKSTEPS, sliceSteps=BROADCAST_SLICESTEPS };
    ncclEnqueueCheck(&info, recvStub);
    return ncclSuccess;
}
