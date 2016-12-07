#include <iostream>
#include <curand.h>
#include <cstdlib>
#include <unistd.h>
using namespace std;

#include "../headers/graphics.h"
#include "../headers/particles.h"
#include "../headers/turbulence.h"

#include "../utils/cuda_device.h" 
#include "../utils/simple_initializer.h" 

curandGenerator_t generator_host;

//#define SWEEP_LOOP(x) 	for (int i_##x =0; i_##x < x##_sweep.size(); ++i_##x)


int main(int argc, char **argv){

	// select device
	int cDevice;
	cDevice = initDevice(argc, argv);

	// read execution parameters
	string config_filename = "../exec_config/execution_config.r";
	if (argc >2) config_filename = argv[2];

	Initializer I(config_filename);
	I.readFile();
	I.printVars();
	
	// ~~~~~~~~~~~~~~ CPU random generator (MT) ~~~~~~~~~~~~~~~~~~~~~~~~~~~
	curandCreateGeneratorHost(&generator_host, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator_host, time(NULL));	// seed by time in every expt


	TurbulenceEngine *turb = new TurbulenceEngine;
	turb->init(I);
	turb->generateSpectrum();
//	turb->generateNoise();
	turb->calcEquilPsi();
	turb->transformPsi();

	Renderer *R;
	bool graphics = I.getScalar("graphicsQual")>0;
	if (graphics){
		R = new Renderer;
		R->init(I);
		R->glTE = turb;
		initGL(R, &argc, argv);
	}

	ParticleSystem * psys = new ParticleSystem;
	psys->init(I);
	psys->psTE = turb;

	vector <float> cvec = I.getArray("c");
	vector <float> csvec = I.getArray("cS");
	vector <float> ensvec = I.getArray("ens");

	for (int ic=0; ic<cvec.size(); ++ic){
	for (int ics=0; ics<csvec.size(); ++ics){
	for (int iens=0; iens<ensvec.size(); ++iens){
		// ----- init params here -------
		psys->c = cvec[ic];
		psys->cS = csvec[ics];
		psys->iEns = ensvec[iens];
	
		// single simulation run

		// reset particle values
		psys->copyParamsToDevice();
		psys->initParticles();

		// reset turbulence engine
		turb->calcEquilPsi();
		turb->transformPsi();

		// connect to renderer
		if (graphics) R->connect(psys);
		if (graphics) turb->updateColorMap();

		// init IO -- run simulation -- close IO 
		psys->initIO(I);
		psys->launchSim();
		psys->closeIO();
	
		// disconnect renderer
		if (graphics) R->disconnect();
		// --------------------
	}
	}
	}

	psys->freeMemory();
	turb->freeMemory();

	if (graphics) delete R;
	delete psys;
	
	return 0;
}


	// create output dirs
//	if (dataOut || plotsOut || framesOut) system(string("mkdir " + outDir).c_str());
//	if (dataOut)   system(string("mkdir " + dataDir).c_str());
//	if (plotsOut)  system(string("mkdir " + outDir + "/plots").c_str());
//	if (framesOut) system(string("mkdir " + framesDir).c_str());
	
//	// allocate memory
//	allocArrays();

//	// if graphics are on, initGL
//	if (graphicsQual > 0) initGL(&argc, argv, host_params);

//	// for all the chosen parameter sweeps, init arrays and launch simulations
//	SWEEP_LOOP(mu){ 
//	SWEEP_LOOP(nm){ 
//	SWEEP_LOOP(fb){ 
////	SWEEP_LOOP(as){ 
//	SWEEP_LOOP(rg){ 
//	SWEEP_LOOP(rsb){ 

//		// set parameters
//		mu[0] 			= mu_sweep[i_mu];
//		moveStepsPerGen = nm_sweep[i_nm];
//		fitness_base 	= fb_sweep[i_fb];
////		arenaSize 		= as_sweep[i_as];
//		Rg 				= rg_sweep[i_rg];
//		Rs_base 		= rsb_sweep[i_rsb];

//		// for each parameter set, perform as many ensembles as specified in ens_sweep[]
//		SWEEP_LOOP(ens){ 
//			iEns = ens_sweep[i_ens];
//			initStateArrays();
//			launchExpt();
//		}
//			
//	}}}}}//}

