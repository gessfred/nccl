#include "enqueue.h"
#include <iostream>

NCCL_API(ncclResult_t, ncclSend, const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclSend(const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclSend"); 
    struct CollectiveArgs args;
    ncclSendCu(dst, &args);
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecv, const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream) {
    INFO(NCCL_INIT, "ncclRecv"); 
    struct CollectiveArgs args;
    ncclSendCu(src, &args);
    return ncclSuccess;
}
