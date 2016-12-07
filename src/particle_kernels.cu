#ifndef _KERNELS_CU_
#define _KERNELS_CU_

#include "../headers/particles.h"
#include "../utils/cuda_vector_math.cuh"
#include "../utils/cuda_device.h"
#include "../utils/simple_io.h"

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/reduce.h>



__constant__ Movementparams par_dev;

cudaError __host__ ParticleSystem::copyParamsToDevice(){
	return cudaMemcpyToSymbol(par_dev, &par, sizeof(Movementparams));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// KERNEL to set up RANDOM GENERATOR STATES
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
__global__ void rngStateSetup_kernel(int * rng_Seeds, curandState * rngStates, int N){
	//int tid = threadIdx.x;							// each block produces exactly the same random numbers
	int tid_u = threadIdx.x + blockIdx.x*blockDim.x;	// each block produces different random numbers
	if (tid_u >= N) return;
	
	curand_init (rng_Seeds[tid_u], 0, 0, &rngStates[tid_u]);
}

#define PP_SEED 777

void ParticleSystem::initRNG(){
	srand(PP_SEED);
	for (int i=0; i<N; ++i) seeds_h[i] = rand(); 
	cudaMemcpy( seeds_dev, seeds_h, sizeof(int)*N, cudaMemcpyHostToDevice);

	int nt = 256, nb = (N-1)/nt+1;
	rngStateSetup_kernel <<< nb, nt>>> (seeds_dev, dev_XWstates, N);
	getLastCudaError("psys_RNG_kernel_launch");
}


// =========================================================================================
//		Interacting Paticles Kernels!!
// =========================================================================================

// given the position, get the cell ID on a square grid of dimensions nxGrid x nxGrid,
// with each cell of size cellSize
// this function returns cellId considering 0 for 1st grid cell. With multiple blocks, user must add appropriate offset
//		|---|---|---|---|---|
//		|   |   |   |   |   |
//		|---|---|---|---|---|
//		|   |   | x |   |   |	<-- x = (pos.x, pos.y)
//		|---|---|---|---|---|
//		|   |   |   |   |   |
//		|---|---|---|---|---|
//      ^ 0 = (xmin, ymin)	^ nx = xmin + nx*cellSize
inline __device__ int getCellId(float2 pos){
	int ix = (pos.x - par_dev.xmin)/(par_dev.cellSize+1e-12);	// add 1e-6 to make sure that particles on edge of last cell are included in that cell
	int iy = (pos.y - par_dev.ymin)/(par_dev.cellSize+1e-12);
	return iy*par_dev.nCellsX + ix;
}


__global__ void update_grid_kernel(float2* pos_array, int* cellCount_array, int* cellId_array, int* cellParticles_array){
	unsigned int pid = blockIdx.x*blockDim.x + threadIdx.x;
	
	if (pid < par_dev.N){
		int ic = getCellId(pos_array[pid]);//, s);	// get cell Id ic of particle pid
		cellId_array[pid] = ic;						// store cell id in array
		int n = atomicAdd(&cellCount_array[ic],1);	// atomic increment particle count of cell ic
//		n = clamp(n,0,3);
		cellParticles_array[ic*4+n] = pid;		// add particle index to cell ic
	}
}


// CAUTION: pos_new and vel_new are never initialized. Never use them before initializing.
__global__ void movement_kernel(float2 *pos, float2* vel, float2* vel_new, float* stk,
								int * cellParticles, int * cellCounts, int * cellIds){
	
	unsigned int myId = blockIdx.x*blockDim.x + threadIdx.x;	// full particle ID
	if (myId >= par_dev.N) return;
	
	float2 dirR = make_float2(0,0);
	float2 dirA = make_float2(0,0);
//	float2 dirO = make_float2(0,0);
	
	float2 myPos = pos[myId];
	float2 myVel = vel[myId];
	float  mystk = stk[myId];
	
	int nCellsScan = int(par_dev.Rs*0.99999999)+1;
	int myCell     = cellIds[myId];
	int myCellx    = myCell % par_dev.nCellsX;		// convert grid cell to x and y indices
	int myCelly    = myCell / par_dev.nCellsX;
	
	for (int innx=-nCellsScan; innx<nCellsScan+1; ++innx){			//  offsets to add in x and y indices to get neighbour cells
		for (int inny=-nCellsScan; inny<nCellsScan+1; ++inny){
			int otherCellx = myCellx + innx;
			otherCellx = otherCellx + int(otherCellx < 0)*par_dev.nCellsX - int(otherCellx >= par_dev.nCellsX)*par_dev.nCellsX;
			int otherCelly = myCelly + inny;
			otherCelly = otherCelly + int(otherCelly < 0)*par_dev.nCellsX - int(otherCelly >= par_dev.nCellsX)*par_dev.nCellsX;
			
			int nnCell = ix2(otherCellx, otherCelly, par_dev.nCellsX);
			
			for (int ip=0; ip < cellCounts[nnCell]; ++ip){ // i is particle index within nnCell (max = 3)
				
				int i = cellParticles[nnCell*4+ip];	// particle ip in cell nncell
				//printf("thread %d, myCell %d,%d otherCell %d,%d otherParticle %d\n", myId, myCellx,myCelly, otherCellx,otherCelly,i ); // << "thread" << "\n"

				if (i == myId) continue;	// Exclude self
			
				// get direction and distance to other 
				float2 v2other = periodicDisplacement(	myPos, pos[i], 
														par_dev.xmax-par_dev.xmin, 
														par_dev.ymax-par_dev.ymin  );
				float d2other = length(v2other);
		
				// indicator variables 
				float Irr = float(d2other < par_dev.Rr); //? 1:0;
				float Ira = float(d2other < par_dev.Rs); //? 1:0;
		
				// keep adding to dirR and dirA so that average direction or R/A will be taken
				v2other = normalize(v2other); // normalise to consider direction only

				dirR = dirR - v2other*Irr;				// add repulsion only if other fish lies in inside Rr
				dirA = dirA + v2other*Ira*(1-Irr); 		// add attraction only if other fish lies in (Rr < r < Ra)
//				dirO = dirO + vel[i]*Ira*(1-Irr);	// add alignment only if other fish lies in (Rr < r < Ra)
			}
			
		}	
	}

	float Ir = float(length(dirR) > 1e-6);	// particles in Rr
	float Ia = float(length(dirA) > 1e-6);	// particles in Ra and hence also in Rr

	float2 dirS = dirA; //myVel*(1-mystk) + dirA*mystk;	

	float2 finalDir = /*myVel*(1-Ir)*(1-Ia) +*/ dirR*Ir + dirS*Ia*(1-Ir);	
	finalDir = normalizeSafeZero(finalDir);	// finalDir can be zero, as this is just dirR or dirS.
	
	vel_new[myId] = finalDir;
}

// seperate integration in different kernel to keep register usage < 63 
__global__ void integrate_kernel(float2* pos, float2* vel, float2 *dirs_new, float* stk, float2* velField, int nxt, int nyt,
								 curandState * RNGstates){
	unsigned int myId = blockIdx.x*blockDim.x + threadIdx.x;	// full particle ID
	if (myId >= par_dev.N) return;

	float2 myVel = vel[myId];
	float2 myPos = pos[myId];
	float2 finalDir = dirs_new[myId];

//	// apply turning rate constraint
//	float sinT = myVel.x*finalDir.y - myVel.y*finalDir.x;		// sinT = myVel x finalDir
//	float cosT = dot(finalDir, myVel);	// Desired turning angle. Both vectors are unit so dot product is cos(theta) 
//	float cosL = clamp( max(cosT, par_dev.cosphi), -1.f, 1.f);
//	float sinL = sqrtf(1-cosL*cosL);
//	sinL = sinL - 2*sinL*float(sinT < 0);	// equivalent to: if (sinT < 0) sinL = -sinL;
//	float2 a = make_float2(myVel.x*cosL - myVel.y*sinL, myVel.x*sinL + myVel.y*cosL);

	int ipx = (nxt-1)*(myPos.x-par_dev.xmin)/(par_dev.xmax-par_dev.xmin);
	int ipy = (nyt-1)*(myPos.y-par_dev.ymin)/(par_dev.ymax-par_dev.ymin);
	float2 flowVel = velField[ipy*(nxt-1)+ipx];

	// generate copy error (copy error must be zero if finalDir = 0)
	float2 eta = float(length(finalDir)>1e-6)*curand_normal2(&RNGstates[myId])*par_dev.copyErrSd;

	myVel = flowVel*1 + stk[myId]*(finalDir+eta)*par_dev.speed; //normalize(a)*par_dev.speed;

	float2 netVel = myVel; //*1 + flowVel*5*0;
	
	myPos = myPos + netVel * par_dev.dt; 
	makePeriodic(myPos.x, par_dev.xmin, par_dev.xmax);
	makePeriodic(myPos.y, par_dev.ymin, par_dev.ymax);

	pos[myId] = myPos; 
	vel[myId] = finalDir;  /// ??? Check with Vishu
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Kernels and Launchers for grouping algo 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

__global__ void distance_kernel(float2* pos, int myId, int* otherIds){
	unsigned int otherId = blockIdx.x*blockDim.x + threadIdx.x;	// full particle ID
	if (otherId >= par_dev.N) return;

	// get direction and distance to other 
	float2 v2other = periodicDisplacement(	pos[myId], pos[otherId], 
											par_dev.xmax-par_dev.xmin, 
											par_dev.ymax-par_dev.ymin  );
	float d2other = length(v2other);

	otherIds[otherId] = int(d2other < par_dev.rGrp);
}


int ParticleSystem::launch_distance_kernel(int myId){
	distance_kernel <<<gridDims, blockDims>>> (pos_dev, myId, closeIds_dev);
	return 0;
}


__device__ int sort_count;	// keeps track of the number of close particle pairs obtained until reset

void ParticleSystem::resetSortCount(){
	int atomicCount = 0;
	cudaMemcpyToSymbol(sort_count, &atomicCount, sizeof(int), 0, cudaMemcpyHostToDevice);
}

__global__ void atomic_sort_kernel(int myId, int* otherIds, int2* pairs){
	unsigned int otherId = blockIdx.x*blockDim.x + threadIdx.x;	// full particle ID
	if (otherId >= par_dev.N) return;
	if (otherIds[otherId] == 1) pairs[atomicAdd(&sort_count,1)] = make_int2(myId, otherId);
	//__syncthreads();
}


int ParticleSystem::launch_atomic_sort_kernel(int myId){
	int count;
	atomic_sort_kernel <<<gridDims, blockDims>>> (myId, closeIds_dev, closeParticlePairs_dev);
	cudaMemcpyFromSymbol(&count, sort_count, sizeof(int), 0, cudaMemcpyDeviceToHost);
	return count;
}


void print_devArray(int * vdev, int n, int ncol=-1, bool row = false){
	int * v = new int[n];
	cudaMemcpy(v, vdev, n*sizeof(int), cudaMemcpyDeviceToHost);
	if (ncol == -1) printArray(v,n);
	else{
		cout << "particles in cell:\n";
		for (int i=0; i<n/ncol; ++i){
			if (row) cout << i << " | ";
			for (int j=0; j<ncol; ++j){
				cout << v[i*ncol+j] << " ";
			}
			cout << "\n";
		}
		cout << "\n";
	}
	delete [] v;
}


void ParticleSystem::launch_movement_kernel(){

	// reset all cell counts to 0
	thrust::fill( (thrust::device_ptr <int>)cellCounts_dev, (thrust::device_ptr <int>)cellCounts_dev + par.nCellsXY, (int)0);
	
	update_grid_kernel <<<gridDims, blockDims>>>(pos_dev, cellCounts_dev, cellIds_dev, cellParticles_dev);
	getLastCudaError("update_grid_kernel");

    movement_kernel <<<gridDims, blockDims>>>(pos_dev, vel_dev, vel_new_dev, stk_dev, 
    										  cellParticles_dev, cellCounts_dev, cellIds_dev);

	integrate_kernel <<<gridDims, blockDims>>>(pos_dev, vel_dev, vel_new_dev, stk_dev, 
											   psTE->vel_field_dev, psTE->nx, psTE->ny,
											   dev_XWstates); 

//	print_devArray(cellCounts_dev, par.nCellsXY, par.nCellsX);
//	print_devArray(cellParticles_dev, par.nCellsXY*4, 4, true);
//	print_devArray(cellIds_dev, par.N);
	
}


#endif



