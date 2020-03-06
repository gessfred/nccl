#include "enqueue.h"
#include "primitives.h"
#include "send_recv.h"
#include <iostream>

__global__ void sendKernel(int dst, float* sendbuff) {
    struct CollectiveArgs args;
    ncclSendKernel(dst, &args);
}

__global__ void recvKernel(int src, float* recvbuff) {
    struct CollectiveArgs args;
    ncclRecvKernel(src, &args);
}

NCCL_API(ncclResult_t, ncclSend, const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclSend(const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclSend"); 
    struct CollectiveArgs args;
    //ncclSendCu(dst, &args);
    sendKernel<<<1, 1>>>(dst, sendbuff);
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecv, const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclRecv"); 
    struct CollectiveArgs args;
    //ncclSendCu(src, &args);
    recvKernel<<<1, 1>>>(src, recvbuff);
    return ncclSuccess;
}
