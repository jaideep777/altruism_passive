#include <vector>
#include <cmath>
#include <sstream>
#include <iomanip>
using namespace std;

#include "../headers/particles.h"
#include "../headers/graphics.h"

#include "../utils/cuda_vector_math.cuh"


// -------------------------------------------------------
// Altruism code initializations
// -------------------------------------------------------

void ParticleSystem::initIO(Initializer &I){
	
	// ~~~~~~~~~~~~~~~~~ EXPT DESC ~~~~~~~~~~~~~~~~~~~~	 
	stringstream sout;
	sout << setprecision(3);
	sout << I.getString("exptName");
	if (b_baseline) sout << "_base";
	sout << "_n("   << N
		 << ")_nm(" << nStepsLifetime
		 << ")_rd(" << rDisp
	 	 << ")_mu(" << mu
		 << ")_fb(" << fitness_base
		 << ")_as(" << as
	 	 << ")_rg(";
	if (b_constRg) sout << rGrp;
	else sout << "-1";
	if (b_baseline)  sout << ")_stk(" << stk0;
	sout << ")_c(" << c
		 << ")_cS(" << cS
		 << ")_ens(" << iEns
		 << ")";
	
	exptDesc = "p_" + sout.str(); sout.clear();
	
	if (b_dataOut) {
		string outdir = I.getString("homeDir") + "/" + I.getString("outDir");
		string dirname = outdir + "/data";
		string fname = dirname + "/" + exptDesc;

		system(string("mkdir " + outdir).c_str());
		system(string("mkdir " + dirname).c_str());

		p_fout.open(fname.c_str());	
	}

}

void ParticleSystem::closeIO(){
	p_fout.close();
}


//// output all arrays to file
void ParticleSystem::writeState(){
	if (!b_dataOut) return;
	
	// calc avg fitness and avg stk of A and D
	float fitA_avg = 0, fitD_avg = 0, stkA_avg = 0, stkD_avg = 0;
//	int nSC =0, nSD=0, nNC = 0, nND=0;
	for (int i= 0; i < N; ++i) {
		if (pvec[i].wA == Cooperate) {
			fitA_avg += pvec[i].fitness;
			stkA_avg += pvec[i].stk;
//			if (pvec[i].stk > 0.5) ++nSC;
//			else 				   ++nNC;
		}
		else{
			fitD_avg += pvec[i].fitness;
			stkD_avg += pvec[i].stk;
//			if (pvec[i].stk > 0.5) ++nSD;
//			else 				   ++nND;
		}
	}	
	fitA_avg /= (K + 1e-6);
	fitD_avg /= (N - K + 1e-6);
	stkA_avg  /= (K + 1e-6);
	stkD_avg  /= (N - K + 1e-6);

	// print p and related measures
	p_fout  << 0 << "\t"
			<< float(K)/N << "\t"
			<< -999 /*EpgNg*/ << "\t"
			<< -999 /*varPg*/ << "\t"
			<< r << "\t"
			<< -999 /*Skg2bNg*/ << "\t"
			<< -999 /*SkgbNg*/ << "\t"
			<< r2 << "\t"
			<< fitA_avg << "\t"
			<< fitD_avg << "\t"
			<< stkA_avg << "\t"
			<< stkD_avg << "\t"
//			<< nSC << "\t"
//			<< nSD << "\t"
//			<< nNC << "\t"
//			<< nND << "\t"
			<< endl;

	return;
}



// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Use the group sizes and wA to calculate fitness for each individual
// Relies on : g2ng_map, g2kg_map, groupIndices
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//float pMinFitG = 0;
int ParticleSystem::calcFitness(){

//	vector <float> fitness(N);	// array of absolute fitnesses
	
	// calc fitness (group wise) and population minimum fitness
	float pop_min_fit = b+1e30;  // NOTE: init with b+100 as fitness calc below will never exceed b 
	for (int i=0; i<N; ++i) {

		Particle *p = &pvec[i];
		if (p->ng == 1){
			if ( p->wA == Cooperate){
				p->fitness = -c;
			}
			else{
				p->fitness = 0;
			}
		}
		else{
			if ( p->wA == Cooperate){	// individual is cooperator
				p->fitness = (p->kg-1)*b/(p->ng-1) - c;
			}
			else {	// individual is defector
				p->fitness = p->kg*b/(p->ng-1);
			}
		}
//		cout << "ng = " << p->ng << " kg = " << p->kg << " fit = " << p->fitness << endl;
		p->fitness -= cS*(p->stk*p->stk);	// additional flocking cost

		pop_min_fit = fmin(pop_min_fit, p->fitness);
	}
//	cout << "min fit = " << pop_min_fit << endl;

	// shift the fitnesses such that least fitness is = fitness_base
	for (int i=0; i<N; ++i){
		pvec[i].fitness = fitness_base + pvec[i].fitness - pop_min_fit;
	}
	
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// use fitness values to choose which individuals reproduce and 
// init positions/velocities/wA/wS for next generation
// relies on: fitnesses

//   0  1     2       3        4       5  6 7   8 ...  nFish-1			<-- fish number
// |--|---|--------|----|------------|---|-|-|--- ... ---------|		<-- fitnesses mapped onto 0-1
// 0            ^                                 ...          1		<-- range_selector
// selected fish = 3 (in this case)
	
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::select_reproduce(){

	// normalize fitness to get sum 1
	float sum_fit = 0;
	for (int i=0; i<N; ++i){
		sum_fit += pvec[i].fitness;
	}
	
	// create ranges for random numbers 
	vector <float> ranges(N+1);
	ranges[0] = 0;
	for (int i=0; i<N; ++i){
		ranges[i+1] = ranges[i] + pvec[i].fitness/sum_fit;
	}

	vector <Particle> offspring(N);

	// init offspring with mutation
	for (int i=0; i<N; ++i){
		// select reproducing individual (parent)
		float range_selector = runif();
		vector <float>::iterator hi = upper_bound(ranges.begin(), ranges.end(), range_selector);	// ranges vector is naturally ascending, can use upper_bound directly
		int reproducing_id = (hi-1) - ranges.begin();
		int parent_Ad      = reproducing_id;

		// Copy reproducing parent into offspring, then add mutations
		// NOTE: Ancestor index is naturally copied.
		offspring[i] = pvec[parent_Ad];
		
		// Mutate offspring's stk if coevolutionary sim (not baseline)
		if (!b_baseline){
			offspring[i].stk += rnorm(0.0f, muStk_Sd);		// add mutation to ws: normal random number with sd = wsNoisesd
			if (offspring[i].stk < 0) offspring[i].stk = 0; 
//			if (runif() < mu/100.0f) offspring[i].stk = (offspring[i].stk < 0.5)? 0.9:0.1;
		}
		//if (b_baseline) offspring[i].Rs = Rs_base;		// in baseline experiment, set Rs = Rs_base

		// Mutate offspring's wA with some probability
		if (runif() < mu/100.0f) offspring[i].wA = (offspring[i].wA == Cooperate)? Defect:Cooperate;

	}
	
	// kill parents and init new generation from offspring
	pvec = offspring;
}

#include <unistd.h>

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// This function advances the generation after movement is done.
// performs: Group formation, Fitness calc,   Selection,  Reproduction, Output
// relies on: ^ groupIndices, ^ ng & kg maps,  ^ fitness,  ^ fitness
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int ParticleSystem::advanceGen(){

	// copy latest positions to CPU
	cudaMemcpy2D( (void*)&(pvec[0].pos),  sizeof(Particle), (void*) pos_dev,  sizeof(float2), sizeof(float2), N, cudaMemcpyDeviceToHost);

	// Advance generation
	updateGroupIndices_parallel_sort();		// calc group indices 
	updateGroupSizes();	// update groups
	update_r();			// calc assortment
	calcFitness();		// calc fitnesses
	writeState();		// output data if desired
	select_reproduce();	// init next generation 
	disperse(rDisp);	// dispersal within rDisp units. Random dispersal if rDisp = -1

	// copy new arrays back to GPU
	cudaMemcpy2D( (void*) pos_dev, sizeof(float2), (void*)&(pvec[0].pos), sizeof(Particle), sizeof(float2), N, cudaMemcpyHostToDevice);
	cudaMemcpy2D( (void*) vel_dev, sizeof(float2), (void*)&(pvec[0].vel), sizeof(Particle), sizeof(float2), N, cudaMemcpyHostToDevice);
	cudaMemcpy2D( (void*) stk_dev,  sizeof(float),  (void*)&(pvec[0].stk),  sizeof(Particle), sizeof(float),  N, cudaMemcpyHostToDevice);

	// re-init turbulence potential function
	psTE->calcEquilPsi();
	psTE->transformPsi();


	return 0;
}



