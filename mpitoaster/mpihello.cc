#include <mpi.h>
#include <stdio.h>
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
#include <thread>
#include <sstream>
#include <functional>
#include <mpi.h>
#include <unistd.h>

using namespace std::chrono;
using namespace std;


void barrier(cudaStream_t& stream) {
  cudaStreamSynchronize(stream);
}

float elapsed_time(vector<tuple<cudaEvent_t, cudaEvent_t>>& events) {
  float tmp = 0;
  float elapsed = 0;
  for(auto event: events) {
    cudaEvent_t start;
    cudaEvent_t end;
    std::tie(start, end) = event;
    cudaEventElapsedTime(&tmp, start, end);
    elapsed += tmp;
  }
  return elapsed;
}

float run(ncclComm_t& comm, cudaStream_t& stream, int world_rank, int world_size, std::function<void(float*, size_t, int, int, ncclComm_t&, cudaStream_t&, vector<tuple<cudaEvent_t, cudaEvent_t>>&)> f, size_t count) {
  
  vector<tuple<cudaEvent_t, cudaEvent_t>> events;
  float* input = (float*)calloc(count, sizeof(float));
  if(world_rank == 0)
    for(size_t i = 0; i < count; ++i)
      input[i] = 1.0;

  float* sendbuff;
  cudaMalloc(&sendbuff, count * sizeof(float));
  cudaMemcpy(sendbuff, input, count * sizeof(float), cudaMemcpyHostToDevice);
  for(size_t i = 0; i < 500; ++i) {
    f(sendbuff, count, world_rank, world_size, comm, stream, events);
    barrier(stream);
  }
  return elapsed_time(events); 
}

void bcast_verb(float* tensor, size_t count, int world_rank, int world_size, ncclComm_t& comm, cudaStream_t& stream, vector<tuple<cudaEvent_t, cudaEvent_t>>& events) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  ncclBroadcast(tensor, tensor, count, ncclFloat32, 0, comm, stream);
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void bcast_ring_chunked(float* tensor, size_t count, int world_rank, int world_size, ncclComm_t& comm, cudaStream_t& stream, vector<tuple<cudaEvent_t, cudaEvent_t>>& events, size_t chunks) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  size_t chunk_size = count / chunks;
  for(size_t i = 0; i < chunks; ++i) {
    if(world_rank != 0) {
      ncclRecv(tensor+i*chunk_size, chunk_size, ncclFloat32, world_rank-1, comm, stream);  
    }
    if(world_rank < world_size - 1) {
      ncclSend(tensor+i*chunk_size, chunk_size, ncclFloat32, world_rank+1, comm, stream);  
    }
  }
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void bcast_ring(float* tensor, size_t count, int world_rank, int world_size, ncclComm_t& comm, cudaStream_t& stream, vector<tuple<cudaEvent_t, cudaEvent_t>>& events) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  if(world_rank != 0) {
    ncclRecv(tensor, count, ncclFloat32, world_rank-1, comm, stream);  
  }
  if(world_rank < world_size - 1) {
    ncclSend(tensor, count, ncclFloat32, world_rank+1, comm, stream);  
  }
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void bcast_grouped(float* tensor, size_t count, int world_rank, int world_size, ncclComm_t& comm, cudaStream_t& stream, vector<tuple<cudaEvent_t, cudaEvent_t>>& events) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  ncclGroupStart();
  if(world_rank == 0) {
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

void bcast_direct(float* tensor, size_t count, int world_rank, int world_size, ncclComm_t& comm, cudaStream_t& stream, vector<tuple<cudaEvent_t, cudaEvent_t>>& events) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  if(world_rank == 0) {
    for(size_t i = 1; i < world_size; ++i) {
      ncclSend(tensor, count, ncclFloat32, i, comm, stream);
    }
  } else {
    ncclRecv(tensor, count, ncclFloat32, 0, comm, stream);
  }
  cudaEventRecord(end, stream);
  events.push_back(make_tuple(start, end));
}

void bcast_grouped_stream(float* tensor, size_t count, int world_rank, int world_size, ncclComm_t& comm, cudaStream_t& stream, vector<tuple<cudaEvent_t, cudaEvent_t>>& events) {
  cudaEvent_t start;
  cudaEvent_t end;
  cudaStream_t master_streams[world_size-1];
  for(size_t i = 0; i < world_size - 1; ++i) 
    cudaStreamCreate(&master_streams[i]);
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start, stream);
  ncclGroupStart();
  if(world_rank == 0) {

    for(size_t i = 1; i < world_size; ++i) {
      ncclSend(tensor, count, ncclFloat32, i, comm, master_streams[i - 1]);
    }
  } else {
    ncclRecv(tensor, count, ncclFloat32, 0, comm, stream);
  }
  ncclGroupEnd();
  cudaEventRecord(end, stream);
  for(size_t i = 0; i < world_size - 1; ++i) 
    cudaStreamDestroy(master_streams[i]);
  events.push_back(make_tuple(start, end));
}

int main(int argc, char** argv) {
   // Initialize the MPI environment
    MPI_Init(NULL, NULL);
std::stringstream csv;
  //csv.open("data.csv");
csv << "version,tensor,world_size,world_rank,elapsed_time\n";
 
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Get the name of the processor
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
  int device = world_rank %  2;
  cudaStream_t stream;
  cudaSetDevice(device);
  cudaStreamCreate(&stream);

  ncclComm_t comm;
  ncclUniqueId id;
  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (world_rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);


  ncclCommInitRank(&comm, world_size, id, world_rank);
  std::cout << "init[" << world_rank << "]" << std::endl;

  size_t size0 = 266144; // 2**18
  for(size_t i = 0, ts = size0; i < 6; ++i, ts *= 2) {
    float res = run(comm, stream, world_rank, world_size, bcast_verb, ts);
    csv << "verb" << "," << ts << "," << world_size << "," << world_rank << "," << res << "\n";
  }

  for(size_t i = 0, ts = size0; i < 6; ++i, ts *= 2) {
    float res = run(comm, stream, world_rank, world_size, bcast_direct, ts);
    csv << "direct" << "," << ts << "," << world_size << "," << world_rank << "," << res << "\n";
  }

  for(size_t i = 0, ts = size0; i < 6; ++i, ts *= 2) {
    float res = run(comm, stream, world_rank, world_size, bcast_grouped, ts);
    csv << "direct_grouped" << "," << ts << "," << world_size << "," << world_rank << "," << res << "\n";
  }


  for(size_t i = 0, ts = size0; i < 6; ++i, ts *= 2) {
    float res = run(comm, stream, world_rank, world_size, bcast_ring, ts);
    csv << "ring" << "," << ts << "," << world_size << "," << world_rank << "," << res << "\n";
  }
  for(size_t chunks = 2; chunks <= 64; chunks *= 2) {
   stringstream v;
      v << "ring-pipelined-" << chunks; 
 for(size_t i = 0, ts = size0; i < 6; ++i, ts *= 2) {
    float res = run(comm, stream, world_rank, world_size, [chunks](float* tensor, size_t count, int world_rank, int world_size, ncclComm_t& comm, cudaStream_t& stream, vector<tuple<cudaEvent_t, cudaEvent_t>>& events){
       bcast_ring_chunked(tensor, count, world_rank, world_size, comm, stream, events, chunks);
     }, ts);
    csv << v.str() << "," << ts << "," << world_size << "," << world_rank << "," << res << "\n";
  }
}

    // Print off a hello world message
    printf("Hello world from processor %s, rank %d out of %d processors\n",
           processor_name, world_rank, world_size);
std::cout << csv.str() << std::endl;
std::this_thread::sleep_for(std::chrono::seconds(5));
    // Finalize the MPI environment.
    MPI_Finalize();

}

