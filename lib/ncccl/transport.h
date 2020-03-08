/*************************************************************************
 * Copyright (c) 2016-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_TRANSPORT_H_
#define NCCL_TRANSPORT_H_

#include "nccl.h"

#define NTRANSPORTS 3
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2

extern struct ncclTransport ncclTransports[];




#define CONNECT_SIZE 128
struct ncclConnect {
  char data[CONNECT_SIZE];
};

enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };


struct ncclTransport {
  const char name[4];
  ncclResult_t (*canConnect)(int*, struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*);
  struct ncclTransportComm send;
  struct ncclTransportComm recv;
};

#include <pthread.h>

typedef ncclResult_t (*threadFunc_t)(struct ncclProxyArgs*);

enum proxyMode {
  proxyRing = 0,
  proxyFrom = 1,
  proxyTo = 2
};
/*
ncclResult_t transportAllocateProxyArgs(struct ncclComm* comm, struct ncclProxyArgs** argsptr);
ncclResult_t transportSaveProxies(struct ncclProxyArgs* args, int pattern, int root, int nranks);
ncclResult_t transportStartProxy(struct ncclComm* comm);
ncclResult_t transportCreateProxy(struct ncclComm* comm);
ncclResult_t transportDestroyProxy(struct ncclComm* comm);
*/

#include <unistd.h>

// Spin wait until func evaluates to true
template<typename FUNC>
inline void transportProxyWait(const FUNC& func) {
  while (!func()) {
    sched_yield();
  }
}

#endif
