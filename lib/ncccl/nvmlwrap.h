/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NVML_DIRECT
#include <dlfcn.h>
#include "nccl.h"

/* Extracted from nvml.h */
typedef struct nvmlDevice_st* nvmlDevice_t;
#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE   16

typedef enum nvmlEnableState_enum
{
    NVML_FEATURE_DISABLED    = 0,     //!< Feature disabled
    NVML_FEATURE_ENABLED     = 1      //!< Feature enabled
} nvmlEnableState_t;

typedef enum nvmlNvLinkCapability_enum
{
    NVML_NVLINK_CAP_P2P_SUPPORTED = 0,     // P2P over NVLink is supported
    NVML_NVLINK_CAP_SYSMEM_ACCESS = 1,     // Access to system memory is supported
    NVML_NVLINK_CAP_P2P_ATOMICS   = 2,     // P2P atomics are supported
    NVML_NVLINK_CAP_SYSMEM_ATOMICS= 3,     // System memory atomics are supported
    NVML_NVLINK_CAP_SLI_BRIDGE    = 4,     // SLI is supported over this link
    NVML_NVLINK_CAP_VALID         = 5,     // Link is supported on this device
    // should be last
    NVML_NVLINK_CAP_COUNT
} nvmlNvLinkCapability_t;

typedef enum nvmlReturn_enum
{
    NVML_SUCCESS = 0,                   //!< The operation was successful
    NVML_ERROR_UNINITIALIZED = 1,       //!< NVML was not first initialized with nvmlInit()
    NVML_ERROR_INVALID_ARGUMENT = 2,    //!< A supplied argument is invalid
    NVML_ERROR_NOT_SUPPORTED = 3,       //!< The requested operation is not available on target device
    NVML_ERROR_NO_PERMISSION = 4,       //!< The current user does not have permission for operation
    NVML_ERROR_ALREADY_INITIALIZED = 5, //!< Deprecated: Multiple initializations are now allowed through ref counting
    NVML_ERROR_NOT_FOUND = 6,           //!< A query to find an object was unsuccessful
    NVML_ERROR_INSUFFICIENT_SIZE = 7,   //!< An input argument is not large enough
    NVML_ERROR_INSUFFICIENT_POWER = 8,  //!< A device's external power cables are not properly attached
    NVML_ERROR_DRIVER_NOT_LOADED = 9,   //!< NVIDIA driver is not loaded
    NVML_ERROR_TIMEOUT = 10,            //!< User provided timeout passed
    NVML_ERROR_IRQ_ISSUE = 11,          //!< NVIDIA Kernel detected an interrupt issue with a GPU
    NVML_ERROR_LIBRARY_NOT_FOUND = 12,  //!< NVML Shared Library couldn't be found or loaded
    NVML_ERROR_FUNCTION_NOT_FOUND = 13, //!< Local version of NVML doesn't implement this function
    NVML_ERROR_CORRUPTED_INFOROM = 14,  //!< infoROM is corrupted
    NVML_ERROR_GPU_IS_LOST = 15,        //!< The GPU has fallen off the bus or has otherwise become inaccessible
    NVML_ERROR_RESET_REQUIRED = 16,     //!< The GPU requires a reset before it can be used again
    NVML_ERROR_OPERATING_SYSTEM = 17,   //!< The GPU control device has been blocked by the operating system/cgroups
    NVML_ERROR_LIB_RM_VERSION_MISMATCH = 18,   //!< RM detects a driver/library version mismatch
    NVML_ERROR_IN_USE = 19,             //!< An operation cannot be performed because the GPU is currently in use
    NVML_ERROR_UNKNOWN = 999            //!< An internal driver error occurred
} nvmlReturn_t;

typedef struct nvmlPciInfo_st
{
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE]; //!< The tuple domain:bus:device.function PCI identifier (&amp; NULL terminator)
    unsigned int domain;             //!< The PCI domain on which the device's bus resides, 0 to 0xffff
    unsigned int bus;                //!< The bus on which the device resides, 0 to 0xff
    unsigned int device;             //!< The device's id on the bus, 0 to 31
    unsigned int pciDeviceId;        //!< The combined 16-bit device id and 16-bit vendor id

    // Added in NVML 2.285 API
    unsigned int pciSubSystemId;     //!< The 32-bit Sub System Device ID

    // NVIDIA reserved for internal use only
    unsigned int reserved0;
    unsigned int reserved1;
    unsigned int reserved2;
    unsigned int reserved3;
} nvmlPciInfo_t;
static enum { nvmlUninitialized, nvmlInitializing, nvmlInitialized, nvmlError } nvmlState = nvmlUninitialized;

static nvmlReturn_t (*nvmlInternalInit)(void);
static nvmlReturn_t (*nvmlInternalShutdown)(void);
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByPciBusId)(const char* pciBusId, nvmlDevice_t* device);
static nvmlReturn_t (*nvmlInternalDeviceGetIndex)(nvmlDevice_t device, unsigned* index);
static nvmlReturn_t (*nvmlInternalDeviceGetHandleByIndex)(unsigned int index, nvmlDevice_t* device);
static const char* (*nvmlInternalErrorString)(nvmlReturn_t r);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkState)(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive);
static nvmlReturn_t (*nvmlInternalDeviceGetPciInfo)(nvmlDevice_t device, nvmlPciInfo_t* pci);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkRemotePciInfo)(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci);
static nvmlReturn_t (*nvmlInternalDeviceGetNvLinkCapability)(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult);
static nvmlReturn_t (*nvmlInternalDeviceGetMinorNumber)(nvmlDevice_t device, unsigned int* minorNumber);
static nvmlReturn_t (*nvmlInternalDeviceGetCudaComputeCapability)(nvmlDevice_t device, int* major, int* minor);

// Used to make the NVML library calls thread safe
pthread_mutex_t nvmlLock = PTHREAD_MUTEX_INITIALIZER;

ncclResult_t wrapNvmlSymbols(void) {
  if (nvmlState == nvmlInitialized)
    return ncclSuccess;
  if (nvmlState == nvmlError)
    return ncclSystemError;

  if (__sync_bool_compare_and_swap(&nvmlState, nvmlUninitialized, nvmlInitializing) == false) {
    // Another thread raced in front of us. Wait for it to be done.
    while (nvmlState == nvmlInitializing) pthread_yield();
    return (nvmlState == nvmlInitialized) ? ncclSuccess : ncclSystemError;
  }

  static void* nvmlhandle = NULL;
  void* tmp;
  void** cast;

  nvmlhandle=dlopen("libnvidia-ml.so.1", RTLD_NOW);
  if (!nvmlhandle) {
    WARN("Failed to open libnvidia-ml.so.1");
    goto teardown;
  }

#define LOAD_SYM(handle, symbol, funcptr) do {         \
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      WARN("dlsym failed on %s - %s", symbol, dlerror());\
      goto teardown;                                     \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

#define LOAD_SYM_OPTIONAL(handle, symbol, funcptr) do {\
    cast = (void**)&funcptr;                             \
    tmp = dlsym(handle, symbol);                         \
    if (tmp == NULL) {                                   \
      INFO(NCCL_INIT,"dlsym failed on %s, ignoring", symbol); \
    }                                                    \
    *cast = tmp;                                         \
  } while (0)

  LOAD_SYM(nvmlhandle, "nvmlInit", nvmlInternalInit);
  LOAD_SYM(nvmlhandle, "nvmlShutdown", nvmlInternalShutdown);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByPciBusId", nvmlInternalDeviceGetHandleByPciBusId);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetIndex", nvmlInternalDeviceGetIndex);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetHandleByIndex", nvmlInternalDeviceGetHandleByIndex);
  LOAD_SYM(nvmlhandle, "nvmlErrorString", nvmlInternalErrorString);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetPciInfo", nvmlInternalDeviceGetPciInfo);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetMinorNumber", nvmlInternalDeviceGetMinorNumber);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkState", nvmlInternalDeviceGetNvLinkState);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkRemotePciInfo", nvmlInternalDeviceGetNvLinkRemotePciInfo);
  LOAD_SYM_OPTIONAL(nvmlhandle, "nvmlDeviceGetNvLinkCapability", nvmlInternalDeviceGetNvLinkCapability);
  LOAD_SYM(nvmlhandle, "nvmlDeviceGetCudaComputeCapability", nvmlInternalDeviceGetCudaComputeCapability);

  nvmlState = nvmlInitialized;
  return ncclSuccess;

teardown:
  nvmlInternalInit = NULL;
  nvmlInternalShutdown = NULL;
  nvmlInternalDeviceGetHandleByPciBusId = NULL;
  nvmlInternalDeviceGetIndex = NULL;
  nvmlInternalDeviceGetHandleByIndex = NULL;
  nvmlInternalDeviceGetPciInfo = NULL;
  nvmlInternalDeviceGetMinorNumber = NULL;
  nvmlInternalDeviceGetNvLinkState = NULL;
  nvmlInternalDeviceGetNvLinkRemotePciInfo = NULL;
  nvmlInternalDeviceGetNvLinkCapability = NULL;

  if (nvmlhandle != NULL) dlclose(nvmlhandle);
  nvmlState = nvmlError;
  return ncclSystemError;
}


ncclResult_t wrapNvmlInit(void) {
  if (nvmlInternalInit == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalInit();
  if (ret != NVML_SUCCESS) {
    WARN("nvmlInit() failed: %s",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlShutdown(void) {
  if (nvmlInternalShutdown == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret = nvmlInternalShutdown();
  if (ret != NVML_SUCCESS) {
    WARN("nvmlShutdown() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetHandleByPciBusId(const char* pciBusId, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByPciBusId == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetHandleByPciBusId(pciBusId, device), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetHandleByPciBusId() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetIndex(nvmlDevice_t device, unsigned* index) {
  if (nvmlInternalDeviceGetIndex == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetIndex(device, index), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetIndex() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t* device) {
  if (nvmlInternalDeviceGetHandleByIndex == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetHandleByIndex(index, device), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetHandleByIndex() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t* pci) {
  if (nvmlInternalDeviceGetPciInfo == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetPciInfo(device, pci), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetPciInfo() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetMinorNumber(nvmlDevice_t device, unsigned int* minorNumber) {
  if (nvmlInternalDeviceGetMinorNumber == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetMinorNumber(device, minorNumber), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetMinorNumber() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link, nvmlEnableState_t *isActive) {
  if (nvmlInternalDeviceGetNvLinkState == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkState(device, link, isActive), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkState() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link, nvmlPciInfo_t *pci) {
  if (nvmlInternalDeviceGetNvLinkRemotePciInfo == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkRemotePciInfo(device, link, pci), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkRemotePciInfo() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
    nvmlNvLinkCapability_t capability, unsigned int *capResult) {
  if (nvmlInternalDeviceGetNvLinkCapability == NULL) {
    /* Do not warn, this symbol is optional. */
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetNvLinkCapability(device, link, capability, capResult), ret);
  if (ret != NVML_SUCCESS) {
    if (ret != NVML_ERROR_NOT_SUPPORTED)
      INFO(NCCL_INIT,"nvmlDeviceGetNvLinkCapability() failed: %s ",
          nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}

ncclResult_t wrapNvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int* major, int* minor) {
  if (nvmlInternalDeviceGetNvLinkCapability == NULL) {
    WARN("lib wrapper not initialized.");
    return ncclInternalError;
  }
  nvmlReturn_t ret;
  NVMLLOCKCALL(nvmlInternalDeviceGetCudaComputeCapability(device, major, minor), ret);
  if (ret != NVML_SUCCESS) {
    WARN("nvmlDeviceGetCudaComputeCapability() failed: %s ",
        nvmlInternalErrorString(ret));
    return ncclSystemError;
  }
  return ncclSuccess;
}
#endif
