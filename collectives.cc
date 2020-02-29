#include <torch/extension.h>
#include "src/nccl.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void init() {
    ncclComm_t comms[4];
    //managing 4 devices
    int nDev = 4;
    int size = 32*1024*1024;
    int devs[4] = { 0, 1, 2, 3 };
    /*ncclCommInitAll(comms, nDev, devs);*/
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init", &init, "LLTM forward (CUDA)");
}