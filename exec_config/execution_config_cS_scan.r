# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Input parameters for Simulations
# If you change the order of parameters below, you will get what you deserve
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


> STRINGS 
# # > DIR
# Directories for data output
homeDir			/home/jaideep/expts_7/output		# home dir - no spaces allowed
outDir  		turbulence_cs_scan_3x						# output dir name
exptName 	 	ens 								# expt name
 	

> SCALARS
# > GPU_CONFIG
# population
particles 		1024		# total number of particles 4096, 16384, 65536, 262144, 1048576
blockSize	 	256			# threads / block

# > GRAPHICS
# graphics
graphicsQual 	0			# 0 = no graphics, 1 = basic graphics, 2 = good graphics, 3 = fancy graphics, charts etc
dispInterval  	50 			# display interval in ms, -ve = number of mv steps to run before display
b_anim_on 		0		  	# turn animation on immediately on start? if false (0), wait for user to press "a" to start sim

# > EXPT
# experiment properties
b_baseline 		0			# Baseline  - is this a baseline experiment (Rs = 0)
b_constRg  		1			# Grouping  - use constant radius of grouping?

# > PARTICLES
# movement parameters
dt  			0.2			# time step (for movement)
speed  			1			# particle speed
Rr 				1			# radius of repulsion (body size) (NEVER CHANGE THIS)
Rs				4			# Initial/baseline value of Rs
copyErrSd 		0.05		# SD of error in following desired direction
turnRateMax 	50			# degrees per sec	

# > SELECTION
# selection and mutation
b 				100			# benefit value
stkNoiseSd 		0.01 		# SD of noise in ws at selection wS = wS(parent) + noise*rnorm(0,1)

# > INIT
# init
fC0 			0.0			# initial frequency of cooperators
stk0  			0.0			# initial kA = weight given to attraction direction while moving

# > OUTPUT
# output
dataOut  		1			# output all values in files?

# > SIM
# movement and SimParams
rGroup			2
rDisp 			-1			# Dispersal - Radius of dispersal. (-1 for random dispersal)
arenaSize 		300			# the size of the entire space (x & y), when body size = 1 --# > determines density

nStepsLife		2000 
genMax 			2000		# number of generations to simulate. **Nearest upper multiple of nPoints_plot will be taken 

# Altruism params
Fbase			1
mu				0.5

# Turbulence params
nxt				243
nyt				243

mu_t			.1		# 0.1
nu_t			.005	# 0.005
xi_t			.05		# 0.05     <- 0.05 - 0.2
dt_t			0.2		# 0.2
lambda0_t		1.0		# 1.0

> ARRAYS
# > PARAM_SWEEPS
# parameter sets to loop over. These override any of the parameters set above.
# all of these MUST BE SET AFTER particle-system initialization. 
# 	  else, particleSystem will initialize with 1st value in these vectors by default 
# c_full = 0.1 0.12 0.14 0.16 0.19 0.22 0.26 0.3 0.35 0.41 0.48 0.56 0.65 0.77 0.89 1.05 1.22 1.43 1.67 1.96 2.29 2.68 3.13 3.66 4.28 5 5.85 6.84 8 9.36 10.95 12.8  -1
# cS_full = 20 40 60 80 100 120 140 160 180 200 220 240 260 280 300 320 -1
# c = 0.1 0.14 0.19 0.26 0.35 0.48 0.65 0.89 1.22 1.67 2.29 3.13 4.28 5.85 8 10.95 -1
# c_offset = 0.12 0.16 0.22 0.3 0.41 0.56 0.77 1.05 1.43 1.96 2.68 3.66 5 6.84 9.36 12.8 -1
c			1.6 -1
cS 			22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 -1
ens			1 2 3 4 5 6 7 8 9 10 -1


