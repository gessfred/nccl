#include "enqueue.h"
#include <iostream>

NCCL_API(ncclResult_t, ncclSend, const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclSend(const int dst, const void* sendbuff, size_t count, ncclDataType_t datatype,
    ncclComm_t comm, cudaStream_t stream) {
    std::cout << "ncclSend" << std::endl;
    INFO(NCCL_INIT, "ncclSend"); 
    return ncclSuccess;
}

NCCL_API(ncclResult_t, ncclRecv, const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(const int src, const void* recvbuff, size_t count, ncclDataType_t datatype,
 ncclComm_t comm, cudaStream_t stream) {
        std::cout << "ncclRecv" << std::endl;
        INFO(NCCL_INIT, "ncclRecv"); 
        return ncclSuccess;
}
