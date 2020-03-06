#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

//<int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, class FUNC>
__global__ void ncclSendKernel(int dst, struct CollectiveArgs* args) {
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

__global__ void ncclRecvKernel(int src, struct CollectiveArgs* args) {
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
    prims.directRecv(src, 0, size);
}