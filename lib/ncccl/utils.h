#ifndef NCCL_UTILS_H_
#define NCCL_UTILS_H_

#include <stdint.h>
#include <unistd.h>

//int ncclCudaCompCap();

// PCI Bus ID <-> int64 conversion functions
//ncclResult_t int64ToBusId(int64_t id, char* busId);
ncclResult_t busIdToInt64(char* busId, int64_t* id) {
  const int size = strlen(busId);
  char* hexStr;
  ncclCalloc(&hexStr, size);
  int hexOffset = 0;
  for (int i=0; i<size; i++) {
    char c = busId[i];
    if (c == '.' || c == ':') continue;
    if ((c >= '0' && c <= '9') ||
        (c >= 'A' && c <= 'F') ||
        (c >= 'a' && c <= 'f')) {
      hexStr[hexOffset++] = busId[i];
    } else break;
  }
  hexStr[hexOffset] = '\0';
  *id = strtol(hexStr, NULL, 16);
  free(hexStr);
  return ncclSuccess;
}

// Convert a logical cudaDev index to the NVML device minor number
ncclResult_t getBusId(int cudaDev, int64_t *busId) {
  // On most systems, the PCI bus ID comes back as in the 0000:00:00.0
  // format. Still need to allocate proper space in case PCI domain goes
  // higher.
  char busIdStr[] = "00000000:00:00.0";
  cudaDeviceGetPCIBusId(busIdStr, sizeof(busIdStr), cudaDev);
  busIdToInt64(busIdStr, busId);
  return ncclSuccess;
}

ncclResult_t getHostName(char* hostname, int maxlen, const char delim) {
  if (gethostname(hostname, maxlen) != 0) {
    strncpy(hostname, "unknown", maxlen);
    return ncclSystemError;
  }
  int i = 0;
  while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen-1)) i++;
  hostname[i] = '\0';
  return ncclSuccess;
}

//uint64_t getHash(const char* string, int n);
//uint64_t getHostHash();
//uint64_t getPidHash();

/*struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

static long log2i(long n) {
 long l = 0;
 while (n>>=1) l++;
 return l;
}*/

#endif