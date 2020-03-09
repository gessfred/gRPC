#pragma once
#include <stdint.h>
#include <cuda_runtime.h>
#include <errno.h>
#include <unistd.h>

#include <stdio.h>
#include <chrono>

#include <sys/syscall.h>
#include <limits.h>
#include <string.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <pwd.h>

static const char* userHomeDir() {
  struct passwd *pwUser = getpwuid(getuid());
  return pwUser == NULL ? NULL : pwUser->pw_dir;
}

static void setEnvFile(const char* fileName) {
  FILE * file = fopen(fileName, "r");
  if (file == NULL) return;

  char *line = NULL;
  char envVar[1024];
  char envValue[1024];
  size_t n = 0;
  ssize_t read;
  while ((read = getline(&line, &n, file)) != -1) {
    if (line[read-1] == '\n') line[read-1] = '\0';
    int s=0; // Env Var Size
    while (line[s] != '\0' && line[s] != '=') s++;
    if (line[s] == '\0') continue;
    strncpy(envVar, line, std::min(1024,s));
    envVar[s] = '\0';
    s++;
    strncpy(envValue, line+s, 1024);
    setenv(envVar, envValue, 0);
  }
  if (line) free(line);
  fclose(file);
}

#define TRACE(...)
/* Error type */
typedef enum { ncclSuccess                 =  0,
               ncclUnhandledCudaError      =  1,
               ncclSystemError             =  2,
               ncclInternalError           =  3,
               ncclInvalidArgument         =  4,
               ncclInvalidUsage            =  5,
               ncclNumResults              =  6 } ncclResult_t;
typedef enum {NCCL_LOG_NONE=0, NCCL_LOG_VERSION=1, NCCL_LOG_WARN=2, NCCL_LOG_INFO=3, NCCL_LOG_ABORT=4, NCCL_LOG_TRACE=5} ncclDebugLogLevel;
typedef enum {NCCL_INIT=1, NCCL_COLL=2, NCCL_P2P=4, NCCL_SHM=8, NCCL_NET=16, NCCL_GRAPH=32, NCCL_TUNING=64, NCCL_ALL=~0} ncclDebugLogSubSys;
typedef void (*ncclDebugLogger_t)(ncclDebugLogLevel level, unsigned long flags, const char *file, int line, const char *fmt, ...);
#define NCCL_NET_HANDLE_MAXSIZE 64

#define NCCL_PTR_HOST 0x1
#define NCCL_PTR_CUDA 0x2

int ncclDebugLevel = -1;
thread_local int ncclDebugNoWarn = 0;
uint64_t ncclDebugMask = NCCL_INIT; // Default debug sub-system mask is INIT
FILE *ncclDebugFile = stdout;
pthread_mutex_t ncclDebugLock = PTHREAD_MUTEX_INITIALIZER;


#define WARN(...) printf("[WARN] stub")//ncclDebugLog(NCCL_LOG_WARN, NCCL_ALL, __FILE__, __LINE__, __VA_ARGS__)
#define INFO(FLAGS, ...) printf("[INFO] stub")//ncclDebugLog(NCCL_LOG_INFO, (FLAGS), __func__, __LINE__, __VA_ARGS__)
// Check system calls
#define SYSCHECK(call, name) do { \
  int retval; \
  SYSCHECKVAL(call, name, retval); \
} while (false)

#define SYSCHECKVAL(call, name, retval) do { \
  SYSCHECKSYNC(call, name, retval); \
  if (retval == -1) { \
    WARN("Call to " name " failed : %s", strerror(errno)); \
    return ncclSystemError; \
  } \
} while (false)

#define SYSCHECKSYNC(call, name, retval) do { \
  retval = call; \
  if (retval == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN)) { \
    INFO(NCCL_ALL,"Call to " name " returned %s, retrying", strerror(errno)); \
  } else { \
    break; \
  } \
} while(true)
// Propagate errors up
#define NCCLCHECK(call) if(call != ncclSuccess) return ncclSystemError;

#define NCCLCHECKGOTO(call, res, label) do { \
  res = call; \
  if (res != ncclSuccess) { \
    /* Print the back trace*/ \
    INFO(NCCL_ALL,"%s:%d -> %d", __FILE__, __LINE__, res);    \
    goto label; \
  } \
} while (0);
// Check CUDA calls
#define CUDACHECK(cmd) 

#define CUDACHECKGOTO(cmd, res, label) do {                 \
    cudaError_t e = cmd;                                    \
    if( e != cudaSuccess ) {                                \
        WARN("Cuda failure '%s'", cudaGetErrorString(e));   \
        res = ncclUnhandledCudaError;                       \
        goto label;                                         \
    }                                                       \
} while(false)

#define NTRANSPORTS 3
#define TRANSPORT_P2P 0
#define TRANSPORT_SHM 1
#define TRANSPORT_NET 2

static void initEnv() {
  char confFilePath[1024];
  const char * userDir = userHomeDir();
  if (userDir) {
    sprintf(confFilePath, "%s/.nccl.conf", userDir);
    setEnvFile(confFilePath);
  }
  sprintf(confFilePath, "/etc/nccl.conf");
  setEnvFile(confFilePath);
}


#define NCCL_NET_HANDLE_MAXSIZE 64
#define NCCL_UNIQUE_ID_BYTES 128
#define MAXCHANNELS 32
#define int4 int
#define NCCL_STEPS 8
#define PROXYARGS_ALLOCATE_SIZE 32

#define MAX_IFS 16
#define MAX_IF_NAME_SIZE 16
#define SLEEP_INT            1000 // connection retry sleep interval in usec
#define RETRY_REFUSED_TIMES   2e4 // connection refused retry times before reporting a timeout (20 sec)
#define RETRY_TIMEDOUT_TIMES    3 // connection timed out retry times (each one can take 20s)

#define DIVUP(x, y) \
    (((x)+(y)-1)/(y))
#define ROUNDUP(x, y) \
    (DIVUP((x), (y))*(y))

#define ALIGN_SIZE(size, align) \
  size = ((size + (align) - 1) / (align)) * (align);


typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;
typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];


struct bootstrapNetComm {
  int fd;
};


typedef struct {
  // Name of the network (mainly for logs)
  const char* name;
  // Initialize the network.
  ncclResult_t (*init)();
  // Return the number of adapters.
  ncclResult_t (*devices)(int* ndev);
  // Return the device path in /sys. NCCL will call free on this path.
  ncclResult_t (*pciPath)(int dev, char** path);
  // Return whether this device supports host pointers and/or CUDA pointers
  // as data from the current GPU. Supported types should be composed with
  // NCCL_PTR_HOST and NCCL_PTR_CUDA.
  ncclResult_t (*ptrSupport)(int dev, int* supportedTypes);
  // Create a receiving object and provide a handle to connect to it. The
  // handle can be up to NCCL_NET_HANDLE_MAXSIZE bytes and will be exchanged
  // between ranks to create a connection.
  ncclResult_t (*listen)(int dev, void* handle, void** listenComm);
  // Connect to a handle and return a sending comm object for that peer.
  ncclResult_t (*connect)(int dev, void* handle, void** sendComm);
  // Finalize connection establishment after remote peer has called connectHandle
  ncclResult_t (*accept)(void* listenComm, void** recvComm);
  // Register/Deregister memory. Comm can be either a sendComm or a recvComm.
  // Type is either NCCL_PTR_HOST or NCCL_PTR_CUDA.
  ncclResult_t (*regMr)(void* comm, void* data, int size, int type, void** mhandle);
  ncclResult_t (*deregMr)(void* comm, void* mhandle);
  // Asynchronous send to a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*isend)(void* sendComm, void* data, int size, void* mhandle, void** request);
  // Asynchronous recv from a peer.
  // May return request == NULL if the call cannot be performed (or would block)
  ncclResult_t (*irecv)(void* recvComm, void* data, int size, void* mhandle, void** request);
  // Perform a flush/fence to make sure all data received with NCCL_PTR_CUDA is
  // visible to the GPU
  ncclResult_t (*flush)(void* recvComm, void* data, int size, void* mhandle);
  // Test whether a request is complete. If size is not NULL, it returns the
  // number of bytes sent/received.
  ncclResult_t (*test)(void* request, int* done, int* size);
  // Close and free send/recv comm objects
  ncclResult_t (*closeSend)(void* sendComm);
  ncclResult_t (*closeRecv)(void* recvComm);
  ncclResult_t (*closeListen)(void* listenComm);
} ncclNet_v2_t;

typedef ncclNet_v2_t ncclNet_t;

typedef enum {
  ncclDevSuccess,
  ncclDevAssertedMismatch,
  ncclDevSuspectedMismatch
} ncclDevError_t;

struct ncclPeerInfo {
  int rank;
  int cudaDev;
  int gdrSupport;
  uint64_t hostHash;
  uint64_t pidHash;
  dev_t shmDev;
  int64_t busId;
};

union ncclLLFifoLine {
  /* Flags have to be *after* data, because otherwise, an incomplete receive
     from the network may receive the flag but not the data.
     Note this is assuming that either we receive contiguous chunks of data
     (sockets) or data is written with an atomicity of 8 bytes (IB/RDMA). */
  struct {
    uint32_t data1;
    uint32_t flag1;
    uint32_t data2;
    uint32_t flag2;
  };
  uint64_t v[2];
  int4 i4;
};

struct ncclProxyState {
  pthread_cond_t cond;
  pthread_mutex_t mutex;
  bool stop;
  struct ncclProxyArgs* ops;
  struct ncclProxyArgs* pool;
  struct ncclProxyPool* pools;
};

struct ncclProxyArgs;
typedef ncclResult_t (*proxyProgressFunc_t)(struct ncclProxyArgs*);

struct ncclProxyArgs {
  proxyProgressFunc_t progress;
  struct ncclChannel* channel;
  struct ncclConnector* connector;
  int sliceSteps;
  int chunkSteps;
  int nsteps;
  uint64_t opCount;
  int protocol;
  int state;   // add component before this line -- it is left out during initialization

  // Internal state
  uint64_t head;
  uint64_t tail;
  uint64_t end;
  void* requests[NCCL_STEPS];
  int idle;

  // Element linking
  pthread_mutex_t mutex;
  struct ncclProxyArgs* next;
  struct ncclProxyArgs* nextPeer;
};

struct ncclProxyPool {
  struct ncclProxyPool *next;
  struct ncclProxyArgs elems[PROXYARGS_ALLOCATE_SIZE];
};


struct ncclTransportComm {
  //ncclResult_t (*setup)(struct ncclTopoSystem* topo, struct ncclTopoGraph* graph, struct ncclPeerInfo*, struct ncclPeerInfo*, struct ncclConnect*, struct ncclConnector*, int buffSize, int channelId);
  ncclResult_t (*connect)(struct ncclConnect*, struct ncclConnector*);
  ncclResult_t (*free)(void*);
  ncclResult_t (*proxy)(struct ncclProxyArgs*);
};

struct ncclConnInfo {
  // Regular comm mechanism
  char *buff;         // Local for recv, remote for send
  uint64_t *tail;     // Local for recv, remote for send
  uint64_t *head;     // Local for send, remote for recv
  uint64_t *opCountLoc; // opCount of local rank
  uint64_t *opCountRem; // opCount of remote rank

  int direct;         // Direct communication
  void **ptrExchange; // Pointer exchange for direct communication

  int *fifo;          // Size fifo for proxy

  uint64_t step;      // Keep where we are

  // Low latency mechanism
  union ncclLLFifoLine *llBuff; // Local for recv, remote for send
  uint64_t llLastCleaning;

  // High bandwidth, low latency protocol
  uint64_t* ll128Buff; // Local for recv, remote for send
};

struct ncclConnector {
  int connected;
  struct ncclProxyArgs *proxyAppend;
  struct ncclTransportComm* transportComm;
  void* transportResources; // Host-side resources
  struct ncclConnInfo conn;
  struct ncclComm_t *comm;
};

struct ncclPeer {
  struct ncclConnector send;
  struct ncclConnector recv;
};

struct ncclChannel {
  union {
    struct {
      //struct ncclRing ring;
      //struct ncclTree treeUp;
      //struct ncclTree treeDn;

      int id;
      int nthreads;
      int buffSize;

      // Communication structures
      struct ncclPeer* peers;
      struct ncclPeer* devPeers;

      // Operation list for aggregation
      //struct ncclColl* collectives;
      //struct ncclColl* devCollectives;
      int collStart;
      int collCount;
      int collFifoHead; // Only used by GPU
      int collFifoTail; // Only used by CPU
    };
    int data[0x80];
  };
};

/* Reduction operation selector */
typedef enum { ncclSum        = 0,
               ncclProd       = 1,
               ncclMax        = 2,
               ncclMin        = 3,
               ncclNumOps     = 4 } ncclRedOp_t;

/* Data types */
typedef enum { ncclInt8       = 0, ncclChar       = 0,
               ncclUint8      = 1,
               ncclInt32      = 2, ncclInt        = 2,
               ncclUint32     = 3,
               ncclInt64      = 4,
               ncclUint64     = 5,
               ncclFloat16    = 6, ncclHalf       = 6,
               ncclFloat32    = 7, ncclFloat      = 7,
               ncclFloat64    = 8, ncclDouble     = 8,
               ncclNumTypes   = 9 } ncclDataType_t;

struct ncclDevComm {
  int rank;
  int nRanks;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;
  volatile ncclDevError_t *fatalDevError;

  // Channels, device side
  struct ncclChannel* channels;
};

/* CollectiveArgs + ncclColl are to be a power of two, currently 64 bytes, */
/* to make sure reads to host from the CUDA kernel are aligned. */
/* Make sure to adjust padding at the end of ncclColl. */
struct CollectiveArgs {
  struct ncclDevComm* comm;
  uint64_t opCount;

  // local and remote input, output, and buffer
  const void * ThisInput;
  void * ThisOutput;

  // general parameters
  size_t N;
  uint32_t root;
  uint8_t bid;
  uint8_t nChannels;
  uint16_t nThreads;

  int lastChunkSize;
};
struct ncclColl {
  union {
    struct {
      struct CollectiveArgs args;
      uint16_t funcIndex;
      uint16_t nextIndex;
      uint8_t  active;
    };
    int data[0x10];
  };
};

struct ncclComm_t {
  struct ncclChannel channels[MAXCHANNELS];

  struct ncclPeerInfo* peerInfo;
  struct ncclTopoSystem* topo;

  void* bootstrap;

  int rank;    // my rank in the communicator
  int nRanks;  // number of GPUs in communicator
  int cudaDev; // my cuda device index
  int64_t busId;   // my PCI bus ID in int format

  int node;
  int nNodes;
  int localRanks;

  enum { GROUP, PARALLEL } launchMode;
  cudaStream_t userStream;
  bool userStreamSet;
  cudaEvent_t doneEvent;
  bool checkPointers;

  // Counter to make sure collectives match (needed for bcast/reduce
  // where syncs are not symmetric).
  uint64_t opCount;
  uint64_t lastOpCount;

  // Channels for collectives
  int nChannels;

  // Only nvlink is used for inter-GPU communication
  int nvlink;

  // Algorithm/Protocols thresholds
  //ssize_t threadThresholds[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  //float latencies[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  //float bandwidths[NCCL_NUM_FUNCTIONS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  //int maxThreads[NCCL_NUM_PROTOCOLS];

  // An internal CUDA stream for NCCL kernel CGMD launches
  int groupCudaStream;
  cudaStream_t groupStream;

  // Whether there has been a fatal error in this communicator.
  ncclResult_t fatalError;

  // Error reported by GPU
  volatile ncclDevError_t* fatalDevError;

  // Flag to ask NCCL kernels to abort
  volatile uint32_t *abortFlag;

  // Device side of the communicator
  struct ncclDevComm *devComm;
  // Host copy of the devComm (to free CUDA allocs)
  struct ncclDevComm hostDevComm;

  // Intra-process sync
  int intraRank;
  int intraRanks;
  int* intraBarrier;
  int intraPhase;

  // Storage for deferred intra-process launch
  struct cudaLaunchParams * intraParams;
  struct cudaLaunchParams *myParams;
  int* intraCudaDevs;
  int* intraCGMode; // Whether we can use CUDA9 CGMD or not
  int* intraCC; // Only to check all have the same ComputeCap and disable CGMode if not
  struct ncclColl args;
  void* argsptr;

  // Global proxy thread
  pthread_t proxyThread;
  struct ncclProxyState proxyState;
};



static inline ncclResult_t ncclCudaHostAlloc(void** ptr, void** devPtr, size_t size) {
  CUDACHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  memset(*ptr, 0, size);
  *devPtr = *ptr;
  return ncclSuccess;
}

static inline ncclResult_t ncclCudaHostFree(void* ptr) {
  CUDACHECK(cudaFreeHost(ptr));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCalloc(T** ptr, size_t nelem) {
  void* p = malloc(nelem*sizeof(T));
  if (p == NULL) {
    WARN("Failed to malloc %ld bytes", nelem*sizeof(T));
    return ncclSystemError;
  }
  memset(p, 0, nelem*sizeof(T));
  *ptr = (T*)p;
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaCalloc(T** ptr, size_t nelem) {
  CUDACHECK(cudaMalloc(ptr, nelem*sizeof(T)));
  CUDACHECK(cudaMemset(*ptr, 0, nelem*sizeof(T)));
  return ncclSuccess;
}

template <typename T>
static ncclResult_t ncclCudaMemcpy(T* dst, T* src, size_t nelem) {
  CUDACHECK(cudaMemcpy(dst, src, nelem*sizeof(T), cudaMemcpyDefault));
  return ncclSuccess;
}

#ifdef PROFAPI
#define NCCL_API(ret, func, args...)        \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((alias(#func)))          \
    ret p##func (args);                     \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    __attribute__ ((weak))                  \
    ret func(args)
#else
#define NCCL_API(ret, func, args...)        \
    extern "C"                              \
    __attribute__ ((visibility("default"))) \
    ret func(args)
#endif // end PROFAPI

static __inline__ int ncclTypeSize(ncclDataType_t type) {
  switch (type) {
    case ncclInt8:
    case ncclUint8:
      return 1;
    case ncclFloat16:
      return 2;
    case ncclInt32:
    case ncclUint32:
    case ncclFloat32:
      return 4;
    case ncclInt64:
    case ncclUint64:
    case ncclFloat64:
      return 8;
    default:
      return -1;
  }
}

#define NCCL_NUM_FUNCTIONS 5
typedef enum { ncclCollBroadcast, ncclCollReduce, ncclCollAllGather, ncclCollReduceScatter, ncclCollAllReduce, ncclCollSend, ncclCollRecv } ncclFunc_t;

#define NCCL_NUM_ALGORITHMS 2 // Tree/Ring
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1

#define NCCL_NUM_PROTOCOLS 3 // Simple/LL/LL128
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2

typedef enum {
  ncclPatternRing,
  ncclPatternRingTwice,
  ncclPatternPipelineFrom,
  ncclPatternPipelineTo,
  ncclPatternTreeUp,
  ncclPatternTreeDown,
  ncclPatternTreeUpDown
} ncclPattern_t;

// Used to pass NCCL call information between functions
struct ncclInfo {
  ncclFunc_t coll;
  const char* opName;
  // NCCL Coll Args
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  ncclComm_t comm;
  cudaStream_t stream;
  // Algorithm details
  int chunkSteps;
  int sliceSteps;
  // Computed later
  int algorithm;
  int protocol;
  ncclPattern_t pattern;
  int nChannels;
  int nThreads;
  size_t nBytes;
  int nstepsPerLoop;
  int nchunksPerLoop;
};

ncclResult_t ncclCpuBarrierIn(struct ncclComm_t* comm, int* isLast) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  bool done = false;
  while (done == false) {
    if (val >= comm->intraRanks) {
      WARN("Trying to launch too many collectives");
      return ncclInvalidUsage;
    }
    if (val+1 == comm->intraRanks) {
      // Reset the barrier.
      comm->intraBarrier[comm->intraPhase^1] = 0;
      *isLast = 1;
      return ncclSuccess;
    }
    done = __sync_bool_compare_and_swap(ptr, val, val+1);
    val++;
  }
  *isLast = 0;
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierLast(struct ncclComm_t* comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  int val = *ptr;
  if (__sync_bool_compare_and_swap(ptr, val, val+1) != true) {
    WARN("Trying to launch too many collectives");
    return ncclInternalError;
  }
  return ncclSuccess;
}

ncclResult_t ncclCpuBarrierOut(struct ncclComm_t* comm) {
  volatile int* ptr = (volatile int*)(comm->intraBarrier+comm->intraPhase);
  while (*ptr < comm->intraRanks) pthread_yield();
  comm->intraPhase ^= 1;
  return ncclSuccess;
}

int ncclCudaCompCap() {
  int cudaDev;
  if (cudaGetDevice(&cudaDev) != cudaSuccess) return 0;
  int ccMajor, ccMinor;
  if (cudaDeviceGetAttribute(&ccMajor, cudaDevAttrComputeCapabilityMajor, cudaDev) != cudaSuccess) return 0;
  if (cudaDeviceGetAttribute(&ccMinor, cudaDevAttrComputeCapabilityMinor, cudaDev) != cudaSuccess) return 0;
  return ccMajor*10+ccMinor;
}

static int hexToInt(char c) {
  int v = c - '0';
  if (v < 0) return -1;
  if (v > 9) v = 10 + c - 'a';
  if ((v < 0) || (v > 15)) return -1;
  return v;
}

#define CPU_SET_N_U32 (sizeof(cpu_set_t)/sizeof(uint32_t))

ncclResult_t ncclStrToCpuset(char* str, cpu_set_t* mask) {
  uint32_t cpumasks[CPU_SET_N_U32];
  int m = CPU_SET_N_U32-1;
  cpumasks[m] = 0;
  for (unsigned int o=0; o<strlen(str); o++) {
    char c = str[o];
    if (c == ',') {
      m--;
      cpumasks[m] = 0;
    } else {
      int v = hexToInt(c);
      if (v == -1) break;
      cpumasks[m] <<= 4;
      cpumasks[m] += v;
    }
  }
  // Copy cpumasks to mask
  for (unsigned int a=0; m<CPU_SET_N_U32; a++,m++) {
    memcpy(((uint32_t*)mask)+a, cpumasks+m, sizeof(uint32_t));
  }
  return ncclSuccess;
}

ncclResult_t ncclCpusetToStr(cpu_set_t* mask, char* str) {
  int c = 0;
  uint8_t* m8 = (uint8_t*)mask;
  for (int o=sizeof(cpu_set_t)-1; o>=0; o--) {
    if (c == 0 && m8[o] == 0) continue;
    sprintf(str+c, "%02x", m8[o]);
    c+=2;
    if (o && o%4 == 0) {
      sprintf(str+c, ",");
      c++;
    }
  }
  str[c] = '\0';
  return ncclSuccess;
}
enum ncclProxyOpState { ncclProxyOpNone, ncclProxyOpReady, ncclProxyOpProgress };
struct ncclRing {
  // Shortcuts for userRanks[1] and userRanks[n-1]
  int prev;
  int next;

  // Maps an internal nccl index to user-specified rank order. This is necessary
  // since we need to know how the user expects data to be ordered across
  // devices. Ordered from current device.
  int* userRanks;
  int* devUserRanks;
};
/*
struct ncclTransport p2pTransport;
struct ncclTransport shmTransport;
struct ncclTransport netTransport;

struct ncclTransport ncclTransports[NTRANSPORTS] = {
  p2pTransport,
  shmTransport,
  netTransport,
};
*/
#define RECV 0
#define SEND 1

static bool NeedProxy(int type, int pattern, int root, struct ncclRing* ring, int nranks) {
  if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice) return true;

  /* In chains, one rank does not need a proxy. Let's figure out which one it is */
  // Which index in the reorganized rings should we compare root against */
  const int myrank = 0, nextrank = 1, prevrank = nranks-1;
  int index = pattern == ncclPatternPipelineFrom ?
      /*                            no recv /  no send    if root = */
      /* bcast  */ (type == RECV ?   myrank : nextrank ):
      /* reduce */ (type == RECV ? prevrank :   myrank );
  int rank = ring->userRanks[index];
  return (root != rank);
}

enum { proxyRecv=0, proxySend=1 };

#define PROXYARGS_ALLOCATE_SIZE 32


ncclResult_t transportAllocateProxyArgs(struct ncclComm_t* comm, struct ncclProxyArgs** argsptr) {
  struct ncclProxyState* state = &comm->proxyState;
  struct ncclProxyArgs* elem;
  pthread_mutex_lock(&state->mutex);
  if (state->pool == NULL) {
    // Allocate a new pool of elements
    struct ncclProxyPool* newPool;
    NCCLCHECK(ncclCalloc(&newPool, 1));
    struct ncclProxyArgs* newElems = newPool->elems;
    // Chain newly allocated elements
    for (int i=0; i<PROXYARGS_ALLOCATE_SIZE; i++) {
      if (i+1 < PROXYARGS_ALLOCATE_SIZE) newElems[i].next = newElems+i+1;
    }
    // Add them all to the pool list
    state->pool = newElems;
    // Save the pool memory block for later resource release
    newPool->next = state->pools;
    state->pools = newPool;
  }
  elem = state->pool;
  state->pool = state->pool->next;
  pthread_mutex_unlock(&state->mutex);
  elem->next = elem->nextPeer = NULL;
  *argsptr = elem;
  return ncclSuccess;
}

static void ProxyAppend(struct ncclConnector* connector, struct ncclProxyArgs* args) {
  struct ncclComm_t* comm = connector->comm;
  struct ncclProxyState* state = &comm->proxyState;
  pthread_mutex_lock(&state->mutex);
  if (connector->proxyAppend == NULL) {
    // Nothing running for that peer. Add to the circular list
    if (state->ops == NULL) {
      // Create the list
      args->next = args;
      state->ops = args;
    } else {
      // Insert element in the list
      args->next = state->ops->next;
      state->ops->next = args;
    }
    connector->proxyAppend = args;
  } else {
    // There is an active operation already for that peer.
    // Add it to the per-peer list
    connector->proxyAppend->nextPeer = args;
    connector->proxyAppend = args;
  }
  pthread_mutex_unlock(&state->mutex);
}

template <int type>
static ncclResult_t SaveProxy(int peer, struct ncclProxyArgs* args) {
  if (peer < 0) return ncclSuccess;

  struct ncclPeer* peerComm = args->channel->peers+peer;
  struct ncclConnector* connector = type == proxyRecv ? &peerComm->recv : &peerComm->send;
  if (connector->transportComm->proxy == NULL) return ncclSuccess;

  struct ncclProxyArgs* op;
  NCCLCHECK(transportAllocateProxyArgs(connector->comm, &op));
  memcpy(op, args, sizeof(struct ncclProxyArgs));
  op->connector = connector;
  /**************IMpORTANT****************/
  op->progress = connector->transportComm->proxy;
  //op->state = ncclProxyOpReady;
  ProxyAppend(connector, op);
  return ncclSuccess;
}

ncclResult_t transportSaveProxies(struct ncclProxyArgs* args, int pattern, int root, int nranks) {
  /*if (pattern == ncclPatternRing || pattern == ncclPatternRingTwice || pattern == ncclPatternPipelineFrom || pattern == ncclPatternPipelineTo) {
    struct ncclRing* ring = &args->channel->ring;
    if (NeedProxy(RECV, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxyRecv>(ring->prev, args));
    if (NeedProxy(SEND, pattern, root, ring, nranks)) NCCLCHECK(SaveProxy<proxySend>(ring->next, args));
  }
  if (pattern == ncclPatternTreeUp || pattern == ncclPatternTreeUpDown) {
    // Tree up
    struct ncclTree* tree = &args->channel->treeUp;
    for (int i=0; i<NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy<proxyRecv>(tree->down[i], args));
    NCCLCHECK(SaveProxy<proxySend>(tree->up, args));
  }
  if (pattern == ncclPatternTreeDown || pattern == ncclPatternTreeUpDown) {
    // Tree down
    struct ncclTree* tree = &args->channel->treeDn;
    for (int i=0; i< NCCL_MAX_TREE_ARITY; i++) NCCLCHECK(SaveProxy<proxySend>(tree->down[i], args));
    NCCLCHECK(SaveProxy<proxyRecv>(tree->up, args));
  }
  */
  return ncclSuccess;
}

/**
 * persistentThread The queue is comm->op
 * . The event is comm->proxyState
 * 
 */ 
void* persistentThread(void *comm_) {
  struct ncclComm_t* comm = (struct ncclComm_t*)comm_;
  struct ncclProxyState* state = &comm->proxyState;
  struct ncclProxyArgs* op = NULL;
  ncclResult_t ret = ncclSuccess;
  int idle = 1;
  int idleSpin = 0;
  while (1) {
    do {
      if (*comm->abortFlag) return NULL;
      if (op == NULL) {
        pthread_mutex_lock(&state->mutex);
        op = state->ops;
        if (op == NULL) {
          if (state->stop) {
            // No more commands to process and proxy has been requested to stop
            pthread_mutex_unlock(&state->mutex);
            return NULL;
          }
          pthread_cond_wait(&state->cond, &state->mutex);
        }
        pthread_mutex_unlock(&state->mutex);
      }
    } while (op == NULL);
    op->idle = 0;
    // opCount >= lastOpCount are part of an ongoing GroupStart/GroupEnd that hasn't started
    // yet and might be cancelled before they even start. Hold on on those.
    if (op->state != ncclProxyOpNone && op->opCount < comm->lastOpCount) ret = op->progress(op);
    if (ret != ncclSuccess) {
      comm->fatalError = ret;
      INFO(NCCL_ALL,"%s:%d -> %d [Proxy Thread]", __FILE__, __LINE__, ret);
      return NULL;
    }
    idle &= op->idle;
    pthread_mutex_lock(&state->mutex);
    if (!idle) idleSpin = 0;
    struct ncclProxyArgs *next = op->next;
    if (next->state == ncclProxyOpNone) {
      struct ncclProxyArgs *freeOp = next;
      if (next->nextPeer) {
        // Replace next by its next per-peer element.
        next = next->nextPeer;
        if (op != freeOp) {
          next->next = freeOp->next;
          op->next = next;
        } else {
          next->next = next;
        }
      } else {
        // Remove next from circular list
        next->connector->proxyAppend = NULL;
        if (op != freeOp) {
          next = next->next;
          op->next = next;
        } else {
          next = NULL;
        }
      }
      if (freeOp == state->ops) state->ops = next;
      freeOp->next = state->pool;
      state->pool = freeOp;
    }
    op = next;
    if (op == state->ops) {
      if (idle == 1) {
        if (++idleSpin == 10) {
          sched_yield();
          idleSpin = 0;
        }
      }
      idle = 1;
    }
    pthread_mutex_unlock(&state->mutex);
  }
}

ncclResult_t transportStartProxy(struct ncclComm_t* comm) {
  pthread_mutex_lock(&comm->proxyState.mutex);
  if (comm->proxyState.ops != NULL)
    pthread_cond_signal(&comm->proxyState.cond);
  pthread_mutex_unlock(&comm->proxyState.mutex);
  return ncclSuccess;
}

/***
 *  transportCreateProxy spawns a new "persistent" thread running with arg comm
 */ 
ncclResult_t transportCreateProxy(struct ncclComm_t* comm) {
  if (!comm->proxyThread) {
    comm->proxyState.cond = PTHREAD_COND_INITIALIZER;
    comm->proxyState.mutex = PTHREAD_MUTEX_INITIALIZER;
    comm->proxyState.ops = NULL;
    pthread_create(&comm->proxyThread, NULL, persistentThread, comm);
  }
  return ncclSuccess;
}

ncclResult_t transportDestroyProxy(struct ncclComm_t* comm) {
  struct ncclProxyState* state = &comm->proxyState;

  // Request the proxy to stop and then wake it
  pthread_mutex_lock(&state->mutex);
  state->stop = true;
  pthread_cond_signal(&state->cond);
  pthread_mutex_unlock(&state->mutex);
  if (comm->proxyThread) pthread_join(comm->proxyThread, NULL);

  // Free off any memory allocated for the proxy arg pools
  pthread_mutex_lock(&state->mutex);
  struct ncclProxyState* proxyState = &comm->proxyState;
  while (proxyState->pools != NULL) {
    struct ncclProxyPool *next = proxyState->pools->next;
    free(proxyState->pools);
    proxyState->pools = next;
  }
  pthread_mutex_unlock(&state->mutex);

  return ncclSuccess;
}
