#include <torch/extension.h>
#include <nccl.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


void init(int nDev) {
    ncclComm_t comms[nDev];
    //int size = 32*1024*1024;
    int devs[nDev];
    for(int i = 0; i < nDev; ++i) {
        devs[i] = i;
    }
    NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
    /*ncclCommInitAll(comms, nDev, devs);*/
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "init");
}