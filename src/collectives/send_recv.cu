#include "send_recv.h"
#include "common.h"
#include "collectives.h"

__device__ void ncclSendCu(int dst, struct CollectiveArgs* args) {
    ncclSendKernel(dst, args);
}

__device__ void ncclRecvCu(int src, struct CollectiveArgs *args) {
    ncclRecvKernel(src, args);
}
