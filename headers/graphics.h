#ifndef GRAPHICS_H
#define GRAPHICS_H

#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>

#include "../utils/simple_timer.h"
#include "../utils/simple_initializer.h"
#include "../utils/simple_palettes.h"

#include "../headers/particles.h"
#include "../headers/turbulence.h"

/* =======================================================================
	Shape classes
======================================================================= */ 
class Shape{
	public:
	string objName;
	bool doubleBuffered;
	int nVertices;
	
	GLuint vbo_ids[2];
	GLuint colorBuffer_id;

	GLuint vertexShader_id;
	GLuint fragmentShader_id;
	GLuint program_id;
	
	string vertexShaderFile;
	string fragmentShaderFile;
	
	public:
	Shape(){};
	Shape(string obj_name, bool dbuff);
	void init(string obj_name, bool dbuff);

	void createVBO(void* data, int nbytes);
	void createShaders();
	void createColorBuffer();
	
	void deleteVBO();
	void deleteShaders();
	void deleteColorBuffer();
	
	void setColors(float4 *colData, int nColors);
	void setColor(float4 c);
	
	void useProgram();
};





/* =======================================================================
	NOTE ON USING A RENDERER

	Renderer should be used as follows:

	Renderer R;
	R.init();	// renderer init MUST happen before GL init.
	initGL();	// requires valid initialized renderer to be set for use by openGL 
	
	R.connect(particle_system_1);
		... render
	R.disconect();
	
	R.connect(particle_system_2);
		... render
	R.disconnect();
	
	...

=======================================================================  */ 


/* =======================================================================
	NOTE ON PARTICLE COLURING METHOD
	
	class Particle has various particle attributes starting wA.
	All attributes are of type float or int (4 bytes).
	particleColorAttribute is a void pointer to the desired attribute 
		in the first particle. i.e. &pvec[0].<attr>
	since all particles are 4 bytes, this pointer can be made to point 
		at the nth attribute as (float*)&pvec[0].<attr>+n. This is how 
		the current coloring attributes are set using the number keys.
		the value 'n' is stored in the iColorAttribute
	once the pointer is set at the first value, it can be advanced by 
		sizeof(Particle) bytes to get the same attribute value in the 
		next particle. This is used to set the color buffer.
	Of course, we need to know the type of the attribute to do this, 
		so we store the types (int/float) in an array.
	To get the final color, the value is discretized using the min/max
		bounds and the palette is looked up to get RGB.
=======================================================================  */ 


enum UpdateMode {Step, Time};

class Renderer{
	private:

	// string to store command to be executed from GUI console
	string command;

	public:
	// pointer to the particle system to render	
	ParticleSystem * psys;
	
	// pointer to turbulence engine
	TurbulenceEngine * glTE;
	
	// color options
	void * particleColorAttribute;	// pointer to the 1st value in class Particle to use as colour
	int maxColorAttributes;
	int iColorAttribute;
	vector <string> colorAttrNames;
	vector <string> colorAttrTypes;
	vector <float> colorValueMin;
	vector <float> colorValueMax;
	
	// update intervals/steps
	int nSkip;				// number of steps to skip before re-rendering
	int displayInterval;	// interval in ms between 2 display calls
	int updateMode; 		// update mode: update after fixed time or fixed steps
	int quality;			// quality of graphics
	//int b_anim_on;

	// layers - by default only layer 0 is visible
	vector <bool> layerVis;
	
	// colour palettes
	vector <Colour_rgb> palette;
	vector <Colour_rgb> palette_rand;
	vector <Colour_rgb> palette_cmap;

	unsigned int window_width;
	unsigned int window_height;

	// coordinate system
	float xmin, xmax, ymin, ymax;

	// render flags
	bool b_renderConsole;
	bool b_renderText;
	bool b_renderLabels;
	bool b_renderRs;
	bool b_renderGrid;
	bool b_renderAxes;
	bool b_renderColorMap;
		
	float tailLen;
	float tailLen_def;

	// frame counter
	SimpleCounter frameCounter;
	
	// GPU stuff
	GLuint vao_id;
	cudaGraphicsResource * pos_cgr;
	cudaGraphicsResource * vel_cgr;

	int swap;	// index of the most recently updated buffer	

	Shape pos_shape;
	Shape vel_shape;
	Shape grid_shape;
	Shape cmap_shape;
	
	public:
	// init
	void init(Initializer &I);
	void connect(ParticleSystem *P);
	void disconnect();

	// serious stuff
	void createVBO();
	void deleteVBO();
	void createShaders();
	void deleteShaders();
	void setColorBufferData();
	void setCmapColorBufferData(float* colData, int nCol, int nlevCol);

	// main stuff
	void renderParticles();

	// fancy stuff
	int getDisplayInterval();
	float getParticleColorValue(int i);
	void setColorAttr(int j);

	int renderConsole();
	int renderAxes(float lim, float trans);
	void renderGrid();
	void renderColorMap();
	
	void receiveConsoleChar(char key);
	int executeCommand();
	string makeTitle();
	
	void toggleAnim();
	void toggleConsole();
	void toggleText();
	void toggleGrid();
	void toggleAxes();
};

// pointers to the particle system to display and renderer to render display
extern Renderer * glRenderer;
//extern ParticleSystem * glPsys;

// util functions
void loadShader(string filename, GLuint &shader_id, GLenum shader_type);

// openGL callbacks
void setCoordSys(int l);
bool initGL(Renderer *R, int *argc, char **argv);
void timerEvent(int value);
void reshape(int w, int h);
void keyPress(unsigned char key, int x, int y);
void display();
void cleanup();


#endif


