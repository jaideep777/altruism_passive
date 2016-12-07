#include "../headers/particles.h"
#include "../headers/graphics.h"
#include "../utils/cuda_vector_math.cuh"
#include "../utils/simple_io.h"
#include "../utils/cuda_device.h"
using namespace std;

//cudaError __host__ copyParamsToDevice();

void ParticleSystem::printParticles(int n){
	if (n > pvec.size()) n = pvec.size();
	cout << "Particles:\n";
	cout << "Sr. n" << "\t"
//		 << "ancID " << "\t" 
		 << "gID   " << "\t" 
		 << "wA   " << "\t" 
		 << "stk  " << "\t" 
		 << "ng   " << "\t" 
		 << "kg   " << "\t" 
		 << "fit  " << "\t" 
//		 << "px  " << "\t"
//		 << "py  " << "\t"
//		 << "vx  " << "\t"
//		 << "vy  " << "\t"
		 << "\n";
	for (int i=0; i<n; ++i){
		cout << i << "\t"
//			 << pvec[i].ancID << "\t" 
			 << pvec[i].gID << "\t" 
			 << pvec[i].wA << "\t" 
			 << pvec[i].stk << "\t" 
			 << pvec[i].ng << "\t" 
			 << pvec[i].kg << "\t" 
			 << pvec[i].fitness << "\t" 
//			 << pvec[i].pos.x << "\t"
//			 << pvec[i].pos.y << "\t"
//			 << pvec[i].vel.x << "\t"
//			 << pvec[i].vel.y << "\t"
			 << "\n";
	}
	cout << "\n";
}


void ParticleSystem::init(Initializer &I){

	cout << "init particle system" << endl;

	// init variables
	name = I.getString("exptName");
	N = I.getScalar("particles");
	K = K0 = int(I.getScalar("fC0")*N);
	igen = istep = 0;

	b_graphics = I.getScalar("graphicsQual") > 0;
	b_constRg = bool(I.getScalar("b_constRg"));
	b_baseline = bool(I.getScalar("b_baseline"));
	b_anim_on = (b_graphics)? bool(I.getScalar("b_anim_on")) : true;

	if (b_constRg) rGrp = par.rGrp = I.getScalar("rGroup");
	else rGrp = par.rGrp = -1;
	stk0 = I.getScalar("stk0");
	
	// altruism variables
	b = I.getScalar("b");
	rDisp = I.getScalar("rDisp");
	fitness_base = I.getScalar("Fbase");
	mu = I.getScalar("mu");
	muStk_Sd = I.getScalar("stkNoiseSd");
	nStepsLifetime = I.getScalar("nStepsLife");
	genMax = I.getScalar("genMax");
	as = I.getScalar("arenaSize");
	c = I.getArray("c")[0];
	cS = I.getArray("cS")[0];
	iEns = I.getArray("ens")[0];
	b_dataOut = (bool)I.getScalar("dataOut");
	
	// init movement params
	par.dt = I.getScalar("dt");
	par.Rr = I.getScalar("Rr");
	par.Rs = I.getScalar("Rs");
	par.speed = I.getScalar("speed");
	par.copyErrSd = I.getScalar("copyErrSd");
	par.turnRateMax = I.getScalar("turnRateMax")*pi/180;
	par.cosphi = cos(par.turnRateMax*par.dt);

	par.xmin = -I.getScalar("arenaSize")/2;
	par.xmax =  I.getScalar("arenaSize")/2;
	par.ymin = -I.getScalar("arenaSize")/2;
	par.ymax =  I.getScalar("arenaSize")/2;

	// grid properties
	par.N = N;
	par.cellSize = par.Rr;	// this MUST BE equal to Rr. Otherwise code will fail.
	par.nCellsX  = ceil((par.xmax-par.xmin)/(par.cellSize));
	par.nCellsXY = par.nCellsX*par.nCellsX;


	blockDims.x = I.getScalar("blockSize");
	gridDims.x = (N-1)/blockDims.x + 1;

	pvec.resize(N);

	cout << "blocks: " << gridDims.x << ", threads: " << blockDims.x << ", Total threads: " << blockDims.x*gridDims.x << endl; 
	
	// alloc memory for particle velocities, stk etc.  
	cudaMalloc((void**)&stk_dev,     N*sizeof(float));
	cudaMalloc((void**)&vel_new_dev, N*sizeof(float2));

	// alloc memory for pos and vel only if no graphics desired. 
	// Otherwise, these will be allocated as openGL buffers
	if (!b_graphics){
		cudaMalloc((void**)&pos_dev, N*sizeof(float2));
		cudaMalloc((void**)&vel_dev, N*sizeof(float2));
	}

	// allocate memory for grid arrays on device
	cudaMalloc((void**)&cellIds_dev, par.N*sizeof(int));
	cudaMalloc((void**)&cellParticles_dev, 4*par.nCellsXY*sizeof(int));
	cudaMalloc((void**)&cellCounts_dev, par.nCellsXY*sizeof(int));

	// allocate memory for grouping arrays on device
	closeIds = new int[N];	
	closeParticlePairs = new int2[2*N];

	cudaMalloc((void**)&closeIds_dev, N*sizeof(int));
	cudaMalloc((void**)&closeParticlePairs_dev, 2*N*sizeof(int2));

	// alloc memory for RNG
	seeds_h = new int[N];
	cudaMalloc((void**)&seeds_dev, N*sizeof(int));
	cudaMalloc((void**)&dev_XWstates, N*sizeof(curandState));


	// initialize the RNG
	initRNG();

}	


void ParticleSystem::freeMemory(){
	// free memory for particle velocities, stk etc.  
	cudaFree(stk_dev);
	cudaFree(vel_new_dev);

	// free memory for pos and vel only if no graphics desired. 
	// Otherwise, these will be allocated as openGL buffers
	if (!b_graphics){
		cudaFree(pos_dev);
		cudaFree(vel_dev);
	}

	// free memory for grid arrays on device
	cudaFree(cellIds_dev);
	cudaFree(cellParticles_dev);
	cudaFree(cellCounts_dev);

	// free memory for grouping arrays on device
	delete [] closeIds;	
	delete [] closeParticlePairs;

	cudaFree(closeIds_dev);
	cudaFree(closeParticlePairs_dev);

	// free memory for RNG
	delete [] seeds_h;
	cudaFree(seeds_dev);
	cudaFree(dev_XWstates);
	
}


void ParticleSystem::initParticles(){

	// -------- INIT particles ------------------
	for (int i=0; i<N; ++i){

		pvec[i].pos = runif2(par.xmax, par.ymax); 
//		fin >> pvec[i].pos.x >> pvec[i].pos.y >> gi;
		pvec[i].vel = runif2(1.0); // make_float2(0,1); //
		pvec[i].wA  = (i< K0)? Cooperate:Defect; 
//		pvec[i].kA  = par.kA; //	runif(); // 
//		pvec[i].kO  = par.kO; //	runif(); // 
		pvec[i].ancID = pvec[i].gID = i;	// each fish within a block gets unique ancestor ID
		pvec[i].fitness = 0;
		pvec[i].stk = stk0;
		pvec[i].ng = pvec[i].kg = 0;
//		if (b_baseline) pvec[i].stk = stk_base;
//		if (pvec[i].wA == Cooperate) pvec[i].stk = 0.9; // par.Rs; // 1.3;
//		else 						 pvec[i].stk = 0.1; // par.Rs; // 1.1;
	}

	printParticles(20);

	cudaMemcpy2D((void*)stk_dev, sizeof(float), (void*)&pvec[0].stk, sizeof(Particle), sizeof(float),  N, cudaMemcpyHostToDevice);
	if (!b_graphics){
		cudaMemcpy2D((void*)pos_dev, sizeof(float2), (void*)&pvec[0].pos, sizeof(Particle), sizeof(float2),  N, cudaMemcpyHostToDevice);
		cudaMemcpy2D((void*)vel_dev, sizeof(float2), (void*)&pvec[0].vel, sizeof(Particle), sizeof(float2),  N, cudaMemcpyHostToDevice);
	}
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Union-Find functions to calculate group Indices 
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This file contains 3 versions of union-find:
// 		1. serial version is the standard UF algorithm. Note that latest positions 
//		   must be copied to host before this function can be called.
// 		2. Parallel version with pairwise comparisons in parallel followed by  
//		   serial looping over all pairs
//		3. Parallel_sort version with parallel pairwise comparisons followed
//		   by atomic sorting of pairs, then serial unites over only the 
//		   close pairs. This is the fastest version.
//
// Serial version supports constant and variable grouping radius 
// Parallel versions take a constant radius of grouping from Movementparams. Variable
// 		grouping radius code is not implemented in the parallel version because
// 		it is not used anyway for the current simulations.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// given array of parents, find the root of q
int root(int q, int* par){
	while (q != par[q]){
		par[q] = par[par[q]];
		q = par[q];
	}
	return q;
}

// check if p and q have same root, i.e. belong to same group
bool find(int p, int q, int *par){
	return root(p, par) == root(q, par);
}

// put p and q in same group by merging the smaller tree into large one
void unite(int p, int q, int *par, int *sz){
	int i = root(p, par);
	int j = root(q, par);
	if (i==j) return;	// if both already have the same root do nothing
	if (sz[i] < sz[j]) {par[i]=j; sz[j] += sz[i];}
	else 			   {par[j]=i; sz[i] += sz[j];}
}

void ParticleSystem::updateGroupIndices_serial(){

	vector <int> par(N);
	vector <int> sz(N,1);
	for (int i=0; i<N; ++i) par[i] = i;
	
	for (int p=0; p<N; ++p){
		for (int q=0; q<= p; ++q) {

			float2 v2other = periodicDisplacement( pvec[p].pos, pvec[q].pos, dx, dy);
			float d2other = length(v2other);
			
			// set Radius of grouping from const or Rs
			float D = rGrp; //(b_constRg)? rGrp : pvec[p].Rs;	// if b_constRg, use rGrp. Else, use particle Rs
			// if distance is < R_grp, assign same group
			if (d2other < D){
				unite(p,q,&par[0],&sz[0]);
			} 
			
		}
	}

	for (int i=0; i<N; ++i){
		pvec[i].gID = root(i, &par[0]);
	}
	
	// copy the new values to opengl color buffer
	if (b_graphics) glRenderer->setColorBufferData();
}


void ParticleSystem::updateGroupIndices_parallel(){

	// initialize serially on CPU
	vector <int> par(N);
	vector <int> sz(N,1);
	for (int i=0; i<N; ++i) par[i] = i;

	// perform pairwaise distance checks in parallel
	for (int p=0; p<N; ++p){

		launch_distance_kernel(p);
		cudaMemcpy(closeIds, closeIds_dev, N*sizeof(int), cudaMemcpyDeviceToHost);
		
		for (int q=0; q<= p; ++q) {
			if (closeIds[q])  unite(p,q,&par[0],&sz[0]);
		}

	}

	// set final parents serially
	for (int i=0; i<N; ++i){
		pvec[i].gID = root(i, &par[0]);
	}
	
	// copy the new values to opengl color buffer
	if (b_graphics) glRenderer->setColorBufferData();	
}

void ParticleSystem::updateGroupIndices_parallel_sort(){

	// initialize serially on CPU
	vector <int> par(N);
	vector <int> sz(N,1);
	for (int i=0; i<N; ++i) par[i] = i;

	for (int p=0; p<N; ++p){

		launch_distance_kernel(p); // perform pairwaise distance checks in parallel for pairs (p, threadIdx)
		int sortCount = launch_atomic_sort_kernel(p);  // put close pairs (as indices in int2 pairs) in closeParticlePairs array
		//cout << "sortCount = " << sortCount << endl;
		
		if (sortCount >= N || p == N-1){	// wait till the pairs array is full, then serially perfom unions 
			cudaMemcpy(closeParticlePairs, closeParticlePairs_dev, sortCount*sizeof(int2), cudaMemcpyDeviceToHost);

			for (int q=0; q<sortCount; ++q){
				unite(closeParticlePairs[q].x, closeParticlePairs[q].y, &par[0], &sz[0]);
			}

			resetSortCount();	// sortCount measures number of pairs in array. reset to 0 when all these pairs are united
		}
	}

	// finally, serially set the group ID of each index to its root
	for (int i=0; i<N; ++i){
		pvec[i].gID = root(i, &par[0]);
	}
//	cout << "-- 512 " << pvec[512].gID << endl;
	
	// copy the new values to opengl color buffer
	if (b_graphics) glRenderer->setColorBufferData();
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// update group-sizes and cooperators/group from group Indices 
// relies on : groupIndices - must be called after updateGroups
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::updateGroupSizes(){

	// delete previous group sizes
	g2ng_map.clear(); g2kg_map.clear();
	K = 0;
	
	// calculate new group sizes and indices
	for (int i=0; i<N; ++i) {
		Particle &p = pvec[i];
	
		++g2ng_map[p.gID]; 
		if (p.wA == Cooperate ) {
			++g2kg_map[p.gID];
			++K;
		}
	}
	
	return K;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// calculate r by 2 different methods. 
// Assumes fitness  V = (k-wA)b/n - c wA 
// relies on : ng and kg maps updated by updateGroupSizes()
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
float ParticleSystem::update_r(){
	// calculate r and related quantities
	float pbar = K/float(N);
	r = 0;
	float varPg = 0, EpgNg = 0;
	for (int i=0; i<N; ++i){
		Particle *p = &pvec[i];
		p->kg = g2kg_map[p->gID];	// number of cooperators in the group
		p->ng = g2ng_map[p->gID];		// number of individuals in the group

		EpgNg += float(p->kg)/p->ng/p->ng;
		varPg += (float(p->kg)/p->ng-pbar)*(float(p->kg)/p->ng-pbar);
	}
	EpgNg /= N;
	varPg /= N;
	
	// calc r by another method (should match with r calc above)
	r2 = 0;
	float Skg2bNg = 0, SkgbNg = 0;
	for (map <int,int>::iterator it = g2kg_map.begin(); it != g2kg_map.end(); ++it){
		float kg_g = it->second;
		float ng_g = g2ng_map[it->first];
		Skg2bNg += kg_g*kg_g/ng_g;
		SkgbNg  += kg_g/ng_g;
	}

	if (K == 0 || K == N) r = r2 = -1e20;	// put nan if p is 0 or 1
	else {
		r  = varPg/pbar/(1-pbar) - EpgNg/pbar;
		r2 = float(N)/K/(N-K)*Skg2bNg - float(K)/(N-K) - SkgbNg/K;
	}

	return r;
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// call the 3 functions above to update all info about groups
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::updateGroups(){
	updateGroupIndices_parallel_sort();
	updateGroupSizes();
	update_r();
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Disperse particles to random locations within radius R of current pos 
// if R == -1, disperse in the entire space
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::disperse(int R){
	for (int i=0; i<N; ++i){
		Particle *p = &pvec[i];
		
		// new velocity in random direction
		p->vel = runif2(1.0); 
		
		if (R == -1){ // random dispersal
			p->pos  = runif2(par.xmax, par.ymax); 
		}
		else{	// dispersal within radius R
			float2 dx_new = runif2(R, R);  // disperse offspring within R radius of parent
			p->pos += dx_new;
			makePeriodic(p->pos.x, par.xmin, par.xmax); 
			makePeriodic(p->pos.y, par.ymin, par.ymax); 
		}
	}	
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// move particles for 1 movement step
// launch_movement_kernel() is defined in kernels.cu
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::step(){

	// execute single turbulence update
	psTE->update();

    // Execute single particles-kernel launch
	launch_movement_kernel();

	kernelCounter.increment();
	++istep;

	return 0;
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// execute a single step and check generation advance, sim completion etc.
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::animate(){
	// animate particles
	if (b_anim_on) step();

	if (b_graphics){		// update display if visibilty is on and update mode is Step 
		if (glRenderer->updateMode == Step  && istep % glRenderer->nSkip == 0){
			glutPostRedisplay();	
		}
	}
	
	if (istep >= nStepsLifetime){
		istep = 0; ++igen;
		advanceGen();	// Note: CudaMemcpy at start and end of advanceGen() ensure that all movement kernel calls are done
	}

//	if (istep % 1000 == 0) b_anim_on = false;

//	if (istep % 1000 == 0 && istep !=0){
//		ofstream fout("pos.txt");
//		cudaMemcpy2D( (void*)&(pvec[0].pos),  sizeof(Particle), (void*) pos_dev,  sizeof(float2), sizeof(float2), N, cudaMemcpyDeviceToHost);
//		for (int i=0; i<N; ++i){
//			fout << pvec[i].pos.x << " " 
//				 << pvec[i].pos.y << " " 
//				 << pvec[i].vel.x << " " 
//				 << pvec[i].vel.y << " " 
////				 << pvec[i].gID << '\n';
//				 << "\n";
//		}
//		fout.close();
//		return 1;
//	}

	// when genMax genenrations are done, end run
	if (igen >= genMax) {
		return 1;
	}
	else return 0;
}

#include <unistd.h>

int ParticleSystem::launchSim(){
	
	SimpleProgressBar prog(genMax, &igen, exptDesc);

	prog.start();
	while(1){	// infinite loop needed to poll anim_on signal.
		if (b_graphics) glutMainLoopEvent();
		
		int i = animate();
		usleep(1e2);	// sleep for 2 ms. This dramatically reduces CPU consumption
		prog.update();

		if (i ==1) {
			igen = 0; 
			break;
		}
	}

	return 0;
}




