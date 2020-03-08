/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_INT_NET_H_
#define NCCL_INT_NET_H_

#include "nccl.h"

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

// Translation to external API
static const char* ncclNetName(ncclNet_t* ncclNet) { return ncclNet->name; }
static ncclResult_t ncclNetDevices(ncclNet_t* ncclNet, int* ndev) { NCCLCHECK(ncclNet->devices(ndev)); return ncclSuccess; }
static ncclResult_t ncclNetPciPath(ncclNet_t* ncclNet, int dev, char** path) { NCCLCHECK(ncclNet->pciPath(dev, path)); return ncclSuccess; }
static ncclResult_t ncclNetListen(ncclNet_t* ncclNet, int dev, void* handle, void** listenComm) { NCCLCHECK(ncclNet->listen(dev, handle, listenComm)); return ncclSuccess; }
static ncclResult_t ncclNetConnect(ncclNet_t* ncclNet, int dev, void* handle, void** sendComm) { NCCLCHECK(ncclNet->connect(dev, handle, sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetAccept(ncclNet_t* ncclNet, void* listenComm, void** recvComm) { NCCLCHECK(ncclNet->accept(listenComm, recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetRegMr(ncclNet_t* ncclNet, void* comm, void* data, int size, int type, void** mhandle) { NCCLCHECK(ncclNet->regMr(comm, data, size, type, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetDeregMr(ncclNet_t* ncclNet, void* comm, void* mhandle) { NCCLCHECK(ncclNet->deregMr(comm, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetIsend(ncclNet_t* ncclNet, void* sendComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->isend(sendComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetIrecv(ncclNet_t* ncclNet, void* recvComm, void* data, int size, void* mhandle, void** request) { NCCLCHECK(ncclNet->irecv(recvComm, data, size, mhandle, request)); return ncclSuccess; }
static ncclResult_t ncclNetFlush(ncclNet_t* ncclNet, void* recvComm, void* data, int size, void* mhandle) { NCCLCHECK(ncclNet->flush(recvComm, data, size, mhandle)); return ncclSuccess; }
static ncclResult_t ncclNetTest(ncclNet_t* ncclNet, void* request, int* done, int* size) { NCCLCHECK(ncclNet->test(request, done, size)); return ncclSuccess; }
static ncclResult_t ncclNetCloseSend(ncclNet_t* ncclNet, void* sendComm) { NCCLCHECK(ncclNet->closeSend(sendComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseRecv(ncclNet_t* ncclNet, void* recvComm) { NCCLCHECK(ncclNet->closeRecv(recvComm)); return ncclSuccess; }
static ncclResult_t ncclNetCloseListen(ncclNet_t* ncclNet, void* listenComm) { NCCLCHECK(ncclNet->closeListen(listenComm)); return ncclSuccess; }

#define GPU_BUF_SIZE (2*1024*1024)
static ncclResult_t ncclNetPtrSupport(ncclNet_t* ncclNet, int dev, int* supportedTypes) {
  int support;
  NCCLCHECK(ncclNet->ptrSupport(dev, &support));
  *supportedTypes = support & ~NCCL_PTR_CUDA;
  // The network supports GPU Direct RDMA ; verify the GPU supports it as well.
  if (support & NCCL_PTR_CUDA) {
    void *lComm = NULL, *sComm = NULL, *rComm = NULL;
    ncclNetHandle_t handle;
    void* gpuPtr = NULL;
    void* mHandle = NULL;
    ncclResult_t res;
    NCCLCHECKGOTO(ncclNetListen(ncclNet, dev, &handle, &lComm), res, cleanup);
    NCCLCHECKGOTO(ncclNetConnect(ncclNet, dev, &handle, &sComm), res, cleanup);
    NCCLCHECKGOTO(ncclNetAccept(ncclNet, lComm, &rComm), res, cleanup);
    CUDACHECKGOTO(cudaMalloc(&gpuPtr, GPU_BUF_SIZE), res, cleanup);
    res = ncclNetRegMr(ncclNet, sComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle);
    if (res != ncclSuccess) goto cleanup;
    NCCLCHECKGOTO(ncclNetDeregMr(ncclNet, sComm, mHandle), res, cleanup);
    NCCLCHECKGOTO(ncclNetRegMr(ncclNet, rComm, gpuPtr, GPU_BUF_SIZE, NCCL_PTR_CUDA, &mHandle), res, cleanup);
    NCCLCHECKGOTO(ncclNetDeregMr(ncclNet, rComm, mHandle), res, cleanup);
    *supportedTypes |= NCCL_PTR_CUDA;
cleanup:
    if (gpuPtr) cudaFree(gpuPtr);
    if (rComm) ncclNetCloseRecv(ncclNet, rComm);
    if (sComm) ncclNetCloseSend(ncclNet, sComm);
    if (lComm) ncclNetCloseListen(ncclNet, lComm);
  }
  return ncclSuccess;
}

extern ncclNet_t ncclNetIb;
extern ncclNet_t ncclNetSocket;

#endif
