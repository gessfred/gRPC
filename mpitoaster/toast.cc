#include <iostream>
#include <cstdlib>
#include <nccl.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <vector>
#include <tuple>
#include <fstream>
#include <string>
#include <sstream>
#include <functional>
#include <mpi.h>
#include <unistd.h>

using namespace std::chrono;
using namespace std;

static uint64_t getHostHash(const char* string) {
  // Based on DJB2, result = result * 33 + char
  uint64_t result = 5381;
  for (int c = 0; string[c] != '\0'; c++){
    result = ((result << 5) + result) + string[c];
  }
  return result;
}


static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '.') {
        hostname[i] = '\0';
        return;
    }
  }
}

class mpitoaster_t {
  int device;
  int rank;
  int world_size;
  ncclComm_t comm;
  cudaStream_t stream;
  vector<tuple<cudaEvent_t, cudaEvent_t>> events;


  public:
  mpitoaster_t(int, int, int);
  ~mpitoaster_t();
  void init( int argc, char** argv );
  void bcast_org(float*, size_t);
  void bcast(float*, size_t);
  void bcast_grouped(float*, size_t);
  void bcast_ring(float*, size_t);
  void bcast_ring_grouped(float*, size_t);
  void bcast_ring_chunked(float*, size_t, size_t);
  void bcast_ring_chunked_pipelined(float*, size_t, size_t);
  void barrier();
  void reset();
  float elapsed_time();
};

mpitoaster_t::mpitoaster_t(int d, int r, int w) {
  device = d;
  rank = r;
  world_size = w;
}

mpitoaster_t::~mpitoaster_t() {
  ncclCommDestroy(comm);  
}

void mpitoaster_t::init( int argc, char** argv ) {

  cudaSetDevice(device);
  cudaStreamCreate(&stream);

  //initializing MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  //calculating localRank based on hostname which is used in selecting a GPU
  uint64_t hostHashs[world_size];
  char hostname[1024];
  getHostName(hostname, 1024);
  hostHashs[rank] = getHostHash(hostname);
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostHashs, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD);
  for (int p=0; p<world_size; p++) {
     if (p == rank) break;
     if (hostHashs[p] == hostHashs[rank]) rank++;
  }

  ncclUniqueId id;
  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  ncclCommInitRank(&comm, world_size, id, rank);
  std::cout << "init[" << rank << "]" << std::endl;
}
/*

ncclResult_t  ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t  ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream)
*/
void mpitoaster_t::bcast_org(float* tensor, size_t count) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  ncclBroadcast(tensor, tensor, count, ncclFloat32, 0, comm, stream);
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void mpitoaster_t::bcast(float* tensor, size_t count) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  if(rank == 0) {
    for(size_t i = 1; i < world_size; ++i) {
      ncclSend(tensor, count, ncclFloat32, i, comm, stream);
    }
  } else {
    ncclRecv(tensor, count, ncclFloat32, 0, comm, stream);
  }
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void mpitoaster_t::bcast_grouped(float* tensor, size_t count) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  ncclGroupStart();
  if(rank == 0) {
    for(size_t i = 1; i < world_size; ++i) {
      ncclSend(tensor, count, ncclFloat32, i, comm, stream);
    }
  } else {
    ncclRecv(tensor, count, ncclFloat32, 0, comm, stream);
  }
  ncclGroupEnd();
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void mpitoaster_t::bcast_ring(float* tensor, size_t count) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  if(rank != 0) {
    ncclRecv(tensor, count, ncclFloat32, rank-1, comm, stream);  
  }
  if(rank < world_size - 1) {
    ncclSend(tensor, count, ncclFloat32, rank+1, comm, stream);  
  }
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void mpitoaster_t::bcast_ring_grouped(float* tensor, size_t count) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  ncclGroupStart();
  if(rank != 0) {
    ncclRecv(tensor, count, ncclFloat32, rank-1, comm, stream);
  }
  if(rank < world_size - 1) {
    ncclSend(tensor, count, ncclFloat32, rank+1, comm, stream);
  }
  ncclGroupEnd();
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void mpitoaster_t::bcast_ring_chunked(float* tensor, size_t count, size_t chunks) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  size_t chunk_size = count / chunks;
  for(size_t i = 0; i < chunks; ++i) {
    if(rank != 0) {
      ncclRecv(tensor+i*chunk_size, chunk_size, ncclFloat32, rank-1, comm, stream);  
    }
    if(rank < world_size - 1) {
      ncclSend(tensor+i*chunk_size, chunk_size, ncclFloat32, rank+1, comm, stream);  
    }
  }
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void mpitoaster_t::bcast_ring_chunked_pipelined(float* tensor, size_t count, size_t chunks) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  
  size_t chunk_size = count / chunks;
  
  //ncclGroupStart();
  if(rank == 0) {
    ncclGroupStart();
    for(size_t i = 0; i < chunks; ++i) {
      ncclSend(tensor+i*chunk_size, chunk_size, ncclFloat32, rank+1, comm, stream);
    }  
    ncclGroupEnd();
  } else {
  if(rank != 0) {
    //ncclGroupStart();
    for(size_t i = 0; i < chunks; ++i) {
      ncclRecv(tensor+i*chunk_size, chunk_size, ncclFloat32, rank-1, comm, stream);  
    }
    //ncclGroupEnd();
  }
  if(rank < world_size - 1) {
    //ncclGroupStart();
    for(size_t i = 0; i < chunks; ++i) {
      ncclSend(tensor+i*chunk_size, chunk_size, ncclFloat32, rank+1, comm, stream);  
    }
    //ncclGroupEnd();
  }
  }
  /*for(size_t i = 0; i < chunks; ++i) {
    if(rank != 0) {
      ncclRecv(tensor+i*chunk_size, chunk_size, ncclFloat32, rank-1, comm, stream);
    }
    if(rank < world_size - 1) {
      ncclSend(tensor+i*chunk_size, chunk_size, ncclFloat32, rank+1, comm, stream);
    }
  }*/
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void mpitoaster_t::reset() {
  barrier();
  //ncclCommDestroy(comm);
  events.clear();
  //init();
}

void mpitoaster_t::barrier() {
  cudaStreamSynchronize(stream);
}

float mpitoaster_t::elapsed_time() {
  barrier();
  float tmp = 0;
  float elapsed = 0;
  for(auto event: events) {
    std::tie(start, end) = event;
    cudaEventElapsedTime(&tmp, start, end);
    elapsed += tmp;
  }
  reset();
  return elapsed;
}

void print_tensor(float** tensor, size_t count) {
  float *buff = (float*)calloc(  count, sizeof(float));
  cudaDeviceSynchronize();
  cudaMemcpy(buff, *tensor, count*sizeof(float), cudaMemcpyDeviceToHost);
  std::cout << "tensor([";
  for(size_t i = 0; i < count; ++i) std::cout << buff[i] << ",";
  std::cout << "])" << std::endl;
  free(buff);
}

float run(int rank, mpitoaster_t& mpi, std::function<void(float*, size_t)> f, size_t count) {
  float* input = (float*)calloc(count, sizeof(float));
  if(rank == 0)
    for(size_t i = 0; i < count; ++i)
      input[i] = 1.0;

  float* sendbuff;
  cudaMalloc(&sendbuff, count * sizeof(float));
  cudaMemcpy(sendbuff, input, count * sizeof(float), cudaMemcpyHostToDevice);
  for(size_t i = 0; i < 500; ++i) {
    f(sendbuff, count);
    mpi.barrier();
  }
  return mpi.elapsed_time(); 
}

int main( int argc, char** argv ){
  int rank = atoi(std::getenv("RANK"));
  int device = 0;//atoi(std::getenv("LOCAL_RANK"));
  int world = atoi(std::getenv("WORLD_SIZE"));

  std::stringstream csv_path;
  csv_path << "data_chunk_" << rank << ".csv";
  std::ofstream csv;
  csv.open(csv_path.str(), std::ios_base::app);
  //csv << "version,tensor,world_size,rank,elapsed_time\n";
  mpitoaster_t mpi(device, rank, world);
  mpi.init(argc, argv);
  size_t chunks = 2;
  size_t size0 = 256000;
  float res = run(rank, mpi, [&mpi, chunks](float* tensor, size_t count) {
    mpi.bcast_org(tensor, count);  
  }, size0);
  /*for(size_t chunks = 2; chunks <= 64; chunks *= 2) {
    size_t size0 = 256000;
    for(size_t i = 0; i < 5; ++i) {
      float res = run(rank, mpi, [&mpi, chunks](float* tensor, size_t count) {
        mpi.bcast_ring_chunked(tensor, count, chunks);  
      }, size0);
      stringstream v;
      v << "ring-chunks-" << chunks;
      std::cout << v.str() << "," << size0 << "," << world << "," << rank << "," << "," << res << std::endl;
      csv << v.str() << "," << size0 << "," << world << "," << rank << "," << res << "\n";
      size0 *= 2;
    }
  }*/
  csv.close();


}
