#ifndef __OPENGL_DRIVER_H__
#define __OPENGL_DRIVER_H__

#define NO_MOTION			0
#define ZOOM_MOTION			1
#define ROTATE_MOTION		2
#define TRANSLATE_MOTION	3

#include <gif.h>
#include "OpenGL_GLSL.h"
#include "Timer.h"
#include "Fluid.h"

//3D display configuration
int		motion_mode = 0;
int		mouse_x = 0;
int		mouse_y = 0;
int		screen_width = 1280;
int		screen_height = 720;
float	zoom = 15;
float	swing_angle = -45;
float	elevate_angle = 15;
float	center[3] = { 0, 0, 0 };

//GLSL configuration
GLuint	depth_FBO = 0;
GLuint	depth_texture = 0;
GLuint	shadow_program = 0;
GLuint	phong_program = 0;
GLuint	vertex_handle = 0;
GLuint	normal_handle = 0;
GLuint	color_handle = 0;
GLuint	triangle_handle = 0;
float	light_position[3] = { -2, 2, 4 };
enum RENDER_MODE{VERTEX=0, EDGE, SURFACE, END};

Fluid<float> example;

//Running configuration
bool	idle_run = false;
bool	saveData = false;
bool    makeGIF = false;
bool    updated = false, rendered = false, saved = false;
int		iterations = 1;
int		select_v = -1;
int     render_mode = RENDER_MODE::SURFACE;
int     show_axis = false;
int     show_object = 0;
bool    show_streamline = true;
bool	show_text = true;

// Create shader program and VBO
void Init_GLSL()
{
	//Init GLEW
	GLenum err = glewInit();
	if (err != GLEW_OK)  printf(" Error initializing GLEW! \n");
	else printf("Initializing GLEW succeeded!\n");

	//Init depth texture and FBO
	glGenFramebuffers(1, &depth_FBO);
	glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
	// Depth texture. Slower than a depth buffer, but you can sample it later in your shader
	glGenTextures(1, &depth_texture);
	glBindTexture(GL_TEXTURE_2D, depth_texture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_FLOAT, 0);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
	glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_texture, 0);
	glDrawBuffer(GL_NONE);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) printf("Init_Shadow_Map failed.\n");

	//Load shader program
	Check_GPU_Status();
	shadow_program = Setup_GLSL("shadow");
	phong_program = Setup_GLSL("phong");

	//Create VBO
	glGenBuffers(1, &vertex_handle);
	glGenBuffers(1, &normal_handle);
	glGenBuffers(1, &color_handle);
	glGenBuffers(1, &triangle_handle);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*example.t_number * 3, example.T, GL_STATIC_DRAW);
}

void Init_OPENGL_LIGHTING()
{
	// set openGL shading
	GLfloat lightpos[] = { -2., 2., 4., 0. };
	GLfloat white[] = { 0.5f, 0.5f, 0.5f, 1.0f };
	GLfloat cyan[] = { 0.f, .8f, .8f, 1.f };
	GLfloat gray[] = { .5f, .5f, .5f, 1.f };
	GLfloat shininess[] = { 20 };
	glUseProgram(0);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_DEPTH_TEST);
	glShadeModel(GL_SMOOTH);
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, cyan);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
}

class OpenGL_Driver
{
public:

	OpenGL_Driver(int *argc, char** argv)
	{
		example.Initialize();

		glutInit(argc, argv);
		glutInitDisplayMode(GLUT_SINGLE | GLUT_RGBA | GLUT_DEPTH); // use single frame buffer to avoid read pixel confusion
		glutInitWindowPosition(100, 100);
		glutInitWindowSize(screen_width, screen_height);
		glutCreateWindow("OpenGL Demo");
		glutDisplayFunc(Handle_Display);
		glutReshapeFunc(Handle_Reshape);
		glutKeyboardFunc(Handle_Keypress);
		glutMouseFunc(Handle_Mouse_Click);
		glutMotionFunc(Handle_Mouse_Move);
		glutSpecialFunc(Handle_SpecialKeypress);
		glutIdleFunc(Handle_Idle);
		Handle_Reshape(screen_width, screen_height);

		Init_GLSL();

		glutMainLoop();
	}

	static void Handle_Idle()
	{
		// update
		if (idle_run){
			if (iterations > example.end_frame){
				example.Update(iterations);
				glutPostRedisplay();
				return;
			}
			if (!updated) {
				example.Update(iterations);
				updated = true;
				rendered = false;
				saved = false;
			}

			if (saveData && rendered) {
				Save_To_Image(example.Get_Scene_Name(), iterations);
				saved = true;
				// new update
				iterations++;
				updated = false;
			}
		}else if (makeGIF) {
			static int	i_iteration = 0; // number of examples
			static int	i_frame = 1;     // number of thresholds 
			static float  threshold_value = 0.5;
			
			//updated-->rendered-->saved
			if (!updated){ // start when saved or intial
				if (i_frame > 5) { i_iteration++; i_frame = 1; }
				if (i_iteration > 9) { glutPostRedisplay(); return; }

				printf("processing i_iteration = %d, i_frame =%d\n", i_iteration, i_frame);
				threshold_value = 0.6 - i_frame * 0.1;
				assert(threshold_value > 1e-8);
				example.Surface_Reconstruction_From_Levelset_Update(i_iteration, threshold_value);
				example.Update(i_iteration);
				updated = true;
				rendered = false;
				saved = false;
			}	
			if (saveData && rendered) { //rendered-->saved
				Save_To_GIF(example.Get_Scene_Name(), i_iteration, i_frame);
				saved = true; 
				// new update
				i_frame++;
				updated = false;
			}

			iterations = i_iteration;
			example.threshold_value = threshold_value;
		}else{
			example.Update(iterations);
		}

		glutPostRedisplay();
		// glutPostRedisplay() sets a flag
		// GLUT checks to see if the flag is set at ***THE END OF*** the event loop
		// If set then the display callback function is executed
		// cannot not save image write after parameter update(rendering may not updated yet!)
	}

	static void Handle_Display()
	{
		Create_Shadow_Map();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, screen_width, screen_height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(6, (float)screen_width / (float)screen_height, 1, 200);
		//glOrtho(-1, 1.0, -1.0, 1.0, 1, 100);
		glMatrixMode(GL_MODELVIEW);
		glShadeModel(GL_SMOOTH);

		glLoadIdentity();
		glClearColor(1, 1, 1, 0);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		gluLookAt(0+ center[0], 0+ center[1], zoom, 0+ center[0], 0+ center[1], 0, 0, 1, 0);
		glDisable(GL_LIGHTING);

		// draw background square
		glUseProgram(shadow_program);
		GLuint uniloc = glGetUniformLocation(shadow_program, "shadow_texture");
		glUniform1i(uniloc, 0);
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, depth_texture);
		uniloc = glGetUniformLocation(shadow_program, "light_position");
		glUniform3fv(uniloc, 1, light_position);
		glBegin(GL_POLYGON);
		glVertex3f(-100, -100, -1);
		glVertex3f(100, -100, -1);
		glVertex3f(100, 100, -1);
		glVertex3f(-100, 100, -1);
		glEnd();

		glPushMatrix();
		glTranslatef(-0.0, 0, 0);
		glRotated(elevate_angle, 1, 0, 0);
		glRotated(swing_angle, 0, 1, 0);
		Draw_Unit_Box();
		glPopMatrix();

		int num_streamlines = 0;
		if (show_streamline)
		{
			glPushMatrix();
			glTranslatef(-0.0, 0, 0);
			glRotated(elevate_angle, 1, 0, 0);
			glRotated(swing_angle, 0, 1, 0);
			num_streamlines = example.Draw_Streamlines();
			//example.Draw_Sketches();
			glPopMatrix();
		}
		//--visualize groundtruth and predict result
		if (show_object == 0)
		{
			glPushMatrix();
			glTranslatef(-0.0, 0, 0);
			glRotated(elevate_angle, 1, 0, 0);
			glRotated(swing_angle, 0, 1, 0);
			if (render_mode == RENDER_MODE::EDGE)
				example.Draw_Edges(example.predict_animation);
			else if (render_mode == RENDER_MODE::SURFACE) {
				example.Set_Current_Frame(example.predict_animation);
				Draw_Example(example.X);
			}
			glPopMatrix();
		}
		else if (show_object == 1)
		{
			glPushMatrix();
			glTranslatef(0.0, 0, 0);
			glRotated(elevate_angle, 1, 0, 0);
			glRotated(swing_angle, 0, 1, 0);
			if (render_mode == RENDER_MODE::EDGE)
				example.Draw_Edges(example.train_animation);
			else if (render_mode == RENDER_MODE::SURFACE) {
				example.Set_Current_Frame(example.train_animation);
				Draw_Example(example.X);
			}
			glPopMatrix();
		}

		//-- example: lvst sequence
		/*example.Set_Current_Frame(example.lvst_mesh_animation);
		glPushMatrix();
		glTranslatef(0.0, 0, 0);
		glRotated(elevate_angle, 1, 0, 0);
		glRotated(swing_angle, 0, 1, 0);
		if (render_mode == RENDER_MODE::VERTEX)
			example.Draw_Vertices();
		else if(render_mode ==RENDER_MODE::EDGE)
			example.Draw_Edges();
		else if(render_mode == RENDER_MODE::SURFACE)
			Draw_Example(example.X);
 		glPopMatrix();*/

		//--voxelize levelset grid and sketch grid
		/*Init_OPENGL_LIGHTING();
		glPushMatrix();
		glTranslatef(-0.6, 0, 0);
		glRotated(elevate_angle, 1, 0, 0);
		glRotated(swing_angle, 0, 1, 0);
		example.Draw_Voxels(0);
		glPopMatrix();
		glPushMatrix();
		glTranslatef(0.6, 0, 0);
		glRotated(elevate_angle, 1, 0, 0);
		glRotated(swing_angle, 0, 1, 0);
		example.Draw_Voxels(1);
		glPopMatrix();*/

		//--visualize velocity field
		//Init_OPENGL_LIGHTING();
		//glPushMatrix();
		//glTranslatef(-0.6, 0, 0);
		//glRotated(elevate_angle, 1, 0, 0);
		//glRotated(swing_angle, 0, 1, 0);
		//example.Draw_Arrow(0);
		//glPopMatrix();
		//glPushMatrix();
		//glTranslatef(0.6, 0, 0);
		//glRotated(elevate_angle, 1, 0, 0);
		//glRotated(swing_angle, 0, 1, 0);
		////example.Draw_Arrow(1);
		//glPopMatrix();

		//--draw axis
		if (show_axis)
		{
			// Initialize lighting here because you don't wanna change light position when change view of point!!
			Init_OPENGL_LIGHTING();
			glPushMatrix();                         // global zoom/pan motion is already applied
			glTranslatef(0, 0, 0);					// global translation here
			glRotated(elevate_angle, 1, 0, 0);		// local roation here
			glRotated(swing_angle, 0, 1, 0);
			Draw_Axes(1.0); 
			glPopMatrix();
		}

		//Draw FPS
		if (show_text)
		{
			glLoadIdentity();
			glDisable(GL_DEPTH_TEST);
			glDisable(GL_LIGHTING);
			glUseProgram(0);
			glColor3f(0, 0, 0);
			char text[1024];

			glRasterPos3f(-2.6, 1.4, -30);
			sprintf_s(text, "Iterations: %d", iterations);
			for (int i = 0; text[i] != '\0'; i++)
				glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);

			glRasterPos3f(1.8, 1.4, -30);
			sprintf_s(text, "vertices: %d", example.number);
			for (int i = 0; text[i] != '\0'; i++)
				glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);

			glRasterPos3f(1.8, 1.2, -30);
			sprintf_s(text, "streamlines: %d", num_streamlines);
			for (int i = 0; text[i] != '\0'; i++)
				glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);

			glRasterPos3f(1.8, 1.0, -30);
			sprintf_s(text, "threshold: %f", example.threshold_value);
			for (int i = 0; text[i] != '\0'; i++)
				glutBitmapCharacter(GLUT_BITMAP_TIMES_ROMAN_24, text[i]);

			glEnable(GL_DEPTH_TEST);
		}

		glutSwapBuffers();

		//updated-->rendered-->saved
		if (idle_run || makeGIF) {
			if (updated) { // updated-->rendered
				rendered = true;
				saved = false;
			}
		}
	}

	static void Handle_Reshape(int w, int h)
	{
		screen_width = w, screen_height = h;

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		{
			GLfloat LightDiffuse[] = { 1.0, 1.0, 1.0, 1 };
			GLfloat LightPosition[] = { 0, 0, -100 };
			glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse);
			glLightfv(GL_LIGHT0, GL_POSITION, LightPosition);
			glEnable(GL_LIGHT0);
		}
		{
			GLfloat LightDiffuse[] = { 1.0, 1.0, 1.0, 1 };
			GLfloat LightPosition[] = { 0, 0, 100 };
			glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
			glLightfv(GL_LIGHT1, GL_POSITION, LightPosition);
			glEnable(GL_LIGHT1);
		}
		glutPostRedisplay();
	}

	static void Handle_Keypress(unsigned char key, int mousex, int mousey)
	{
		switch (key)
		{
		case 27: exit(0);
		case'a':
		{
			show_axis = !show_axis;
			printf("show_axis = %s\n", show_axis ? "true" : "false");
			break;
		}
		case'o':
		{
			show_object = (show_object+1) % 2;
			printf("show_object = %s\n", show_object==0? "prediction": "training");
			break;
		}
		case 'p':
		{
			show_streamline = !show_streamline;
			printf("show_streamline = %s\n", show_streamline ? "true" : "false");
			break;
		}
		case 't':
		{
			show_text = !show_text;
			printf("show_text = %s\n", show_text ? "true" : "false");
			break;
		}
		case '-':
		{
			if (example.threshold_value > 0.1 + 1e-8)
			{
				example.threshold_value -= 0.1;
				printf("reconstruction threshold_value = %f\n", example.threshold_value);
				example.Surface_Reconstruction_From_Levelset_Update(example.cur_frame, example.threshold_value);
			}
			break;
		}
		case '=':
		{
			if (example.threshold_value < 1)
			{
				example.threshold_value += 0.1;
				printf("reconstruction threshold_value = %f\n", example.threshold_value);
				example.Surface_Reconstruction_From_Levelset_Update(example.cur_frame, example.threshold_value);
			}
			break;
		}
		case 'i':
		{
			idle_run = !idle_run;
			break;
		}
		case 'g': 
		{
			makeGIF = !makeGIF;
			break;
		}
		case's':
		{
			saveData = !saveData;
			printf("saveData = %s\n", saveData ? "true" : "false");
			break;
		}
		case 'q':
		{
			Save_To_Image(example.Get_Scene_Name(), iterations);
			break;
		}
		case'm':
		{
			render_mode = (render_mode + 1) % RENDER_MODE::END;
			switch (render_mode)
			{
			case RENDER_MODE::VERTEX: { printf("render_mode = VERTEX\n"); break; }
			case RENDER_MODE::EDGE: { printf("render_mode = EDGE\n"); break; }
			case RENDER_MODE::SURFACE: { printf("render_mode = SURFACE\n"); break; }
			}
			break;
		}
		case '[':
		{
			iterations--;
			example.Update(iterations);
			break;
		}
		case ']':
		{
			iterations++;
			example.Update(iterations);
			break;
		}
		}
		glutPostRedisplay();
	}

	static void Handle_Mouse_Click(int button, int state, int x, int y)
	{
		select_v = -1;
		if (state == GLUT_UP)	motion_mode = NO_MOTION;
		if (state == GLUT_DOWN)
		{
			float	p[3], q[3];

			Get_Selection_Ray(x, y, p, q, 2.0, 0.0, 0.0);
			example.Select(p, q, select_v);

			// Set up the motion target
			if (select_v != -1)
			{
				printf("select %d: (%f, %f, %f)\n", select_v, example.X[3 * select_v + 0], example.X[3 * select_v + 1], example.X[3 * select_v + 2]);
			}
			else //No selection, perform camera change
			{
				int modif = glutGetModifiers();
				if (modif & GLUT_ACTIVE_SHIFT)		motion_mode = ZOOM_MOTION;
				else if (modif & GLUT_ACTIVE_CTRL)	motion_mode = TRANSLATE_MOTION;
				else								motion_mode = ROTATE_MOTION;
				mouse_x = x;
				mouse_y = y;
			}
		}

		glutPostRedisplay();
	}

	static void Handle_Mouse_Move(int x, int y)
	{
		if (motion_mode != NO_MOTION)
		{
			if (motion_mode == ROTATE_MOTION)
			{
				swing_angle += (float)(x - mouse_x) * 360 / (float)screen_width;
				elevate_angle += (float)(y - mouse_y) * 180 / (float)screen_height;
				if (elevate_angle> 90)	elevate_angle = 90;
				else if (elevate_angle<-90)	elevate_angle = -90;
			}
			if (motion_mode == ZOOM_MOTION)	zoom += 0.05 * (y - mouse_y);
			if (motion_mode == TRANSLATE_MOTION)
			{
				center[0] += 0.005*(mouse_x - x);
				center[1] -= 0.005*(mouse_y - y);
			}
			mouse_x = x;
			mouse_y = y;
			glutPostRedisplay();
		}
		if (select_v != -1)
		{
			glutPostRedisplay();
		}
	}

	static void Handle_SpecialKeypress(int key, int x, int y)
	{
		if (key == 100)		swing_angle += 45;
		else if (key == 102)	swing_angle -= 45;
		else if (key == 103)	elevate_angle -= 3;
		else if (key == 101)	elevate_angle += 3;
		printf("swing_angle = %f, elevate_angle = %f\n", swing_angle, elevate_angle);
		Handle_Reshape(screen_width, screen_height);
		glutPostRedisplay();
	}


	//utility functions
	static void Delay(float sec)
	{
		Timer time;
		while (time.Get_Time() < sec) {}
	}

	static void Draw_Example(float *X)
	{
		glUseProgram(phong_program);
		// Send light position to vertex shader
		GLuint uniloc = glGetUniformLocation(phong_program, "light_position");
		glUniform3fv(uniloc, 1, light_position);
		// Send vertex data to VBO
		GLuint c0 = glGetAttribLocation(phong_program, "position");
		GLuint c1 = glGetAttribLocation(phong_program, "normal");
		GLuint c2 = glGetAttribLocation(phong_program, "color");
		glEnableVertexAttribArray(c0);
		glEnableVertexAttribArray(c1);
		glEnableVertexAttribArray(c2);

		glBindBuffer(GL_ARRAY_BUFFER, vertex_handle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*example.number * 3, X, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(c0, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (char*)NULL + 0);
		glBindBuffer(GL_ARRAY_BUFFER, normal_handle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*example.number * 3, example.VN, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(c1, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (char*)NULL + 0);
		glBindBuffer(GL_ARRAY_BUFFER, color_handle);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float)*example.number * 3, example.VC, GL_DYNAMIC_DRAW);
		glVertexAttribPointer(c2, 3, GL_FLOAT, GL_FALSE, sizeof(float) * 3, (char*)NULL + 0);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, triangle_handle);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(int)*example.t_number * 3, example.T, GL_STATIC_DRAW);
		glDrawElements(GL_TRIANGLES, example.t_number * 3, GL_UNSIGNED_INT, (char*)NULL + 0);

		if (select_v != -1)
		{
			Draw_Vertex(&X[3 * select_v], 1, 0, 0);
		}

	}

	static void Arrow(GLdouble x1, GLdouble y1, GLdouble z1, GLdouble x2, GLdouble y2, GLdouble z2, GLdouble D)
	{
		double x = x2 - x1;
		double y = y2 - y1;
		double z = z2 - z1;
		double L = sqrt(x*x + y*y + z*z);

		GLUquadricObj *quadObj;

		glPushMatrix();

		glTranslated(x1, y1, z1);

		if ((x != 0.) || (y != 0.)) {
			glRotated(atan2(y, x) / 0.0174533, 0., 0., 1.);
			glRotated(atan2(sqrt(x*x + y*y), z) / 0.0174533, 0., 1., 0.);
		}
		else if (z<0) {
			glRotated(180, 1., 0., 0.);
		}

		glTranslatef(0, 0, L - 4 * D);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluCylinder(quadObj, 2 * D, 0.0, 4 * D, 32, 1);
		gluDeleteQuadric(quadObj);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluDisk(quadObj, 0.0, 2 * D, 32, 1);
		gluDeleteQuadric(quadObj);

		glTranslatef(0, 0, -L + 4 * D);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluCylinder(quadObj, D, D, L - 4 * D, 32, 1);
		gluDeleteQuadric(quadObj);

		quadObj = gluNewQuadric();
		gluQuadricDrawStyle(quadObj, GLU_FILL);
		gluQuadricNormals(quadObj, GLU_SMOOTH);
		gluDisk(quadObj, 0.0, D, 32, 1);
		gluDeleteQuadric(quadObj);

		glPopMatrix();

	}

	static void Draw_Unit_Box()
	{
		glUseProgram(0);
		glDisable(GL_LIGHTING);
		glLineWidth(1);
		glEnable(GL_LINE_SMOOTH);
		glColor3f(0.0, 0.0, 0.0);

		glBegin(GL_LINE_LOOP);
		glVertex3f(-0.5, -0.5, -0.5);
		glVertex3f(-0.5, -0.5, 0.5);
		glVertex3f(0.5, -0.5, 0.5);
		glVertex3f(0.5, -0.5, -0.5);
		glEnd();

		glBegin(GL_LINE_LOOP);
		glVertex3f(-0.5, 0.5, -0.5);
		glVertex3f(-0.5, 0.5, 0.5);
		glVertex3f(0.5, 0.5, 0.5);
		glVertex3f(0.5, 0.5, -0.5);
		glEnd();

		glBegin(GL_LINE_LOOP);
		glVertex3f(-0.5, 0.5, 0.5);
		glVertex3f(-0.5, -0.5, 0.5);
		glVertex3f(0.5, -0.5, 0.5);
		glVertex3f(0.5, 0.5, 0.5);
		glEnd();

		glBegin(GL_LINE_LOOP);
		glVertex3f(-0.5, 0.5, -0.5);
		glVertex3f(-0.5, -0.5, -0.5);
		glVertex3f(0.5, -0.5, -0.5);
		glVertex3f(0.5, 0.5, -0.5);
		glEnd();
	}

	static void Draw_Axes(GLdouble length)
	{
		GLfloat red[] = { 1.f, 0.f, 0.f, 1.f };
		GLfloat green[] = { 0.f, 1.f, 0.f, 1.f };
		GLfloat blue[] = { 0.f, 0.f, 1.f, 1.f };

		glUseProgram(0);
		glEnable(GL_LIGHTING);
		glEnable(GL_LIGHT0);
		glEnable(GL_DEPTH_TEST);
		glShadeModel(GL_SMOOTH);

		// Do not use glscale function before gluCylinder, will transform the normal incorrectly!!
		float scale = 0.2;

		glPushMatrix();
		glTranslatef(0, 0, 0);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, red);
		Arrow(0, 0, 0, length*scale, 0, 0, 0.05*scale);
		glPopMatrix();

		glPushMatrix();
		glTranslatef(0, 0, 0);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, green);
		Arrow(0, 0, 0, 0, length*scale, 0, 0.05*scale);
		glPopMatrix();

		glPushMatrix();
		glTranslatef(0, 0, 0);
		glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, blue);
		Arrow(0, 0, 0, 0, 0, length*scale, 0.05*scale);
		glPopMatrix();
	}

	static void Draw_Vertex(float* p, float r = 1.0, float g = 1.0, float b = 1.0)
	{
		glUseProgram(0);
		glDisable(GL_LIGHTING);

		glColor3f(r, g, b);
		glPushMatrix();
		glTranslated(p[0], p[1], p[2]);
		glutSolidSphere(0.01, 10, 10);
		glPopMatrix();

		glEnable(GL_LIGHTING);
	}

	static void Create_Shadow_Map(char* filename = 0)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, depth_FBO);
		glViewport(0, 0, 1024, 1024); // Render on the whole framebuffer, complete from the lower left corner to the upper right

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glEnable(GL_DEPTH_TEST);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glOrtho(-2, 2, -2, 2, 0, 20);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		gluLookAt(light_position[0], light_position[1], light_position[2], 0, 0, 0, 0, 1, 0);
		//Use fixed program
		glPushMatrix();
		glTranslatef(0, 0, 0);
		glRotated(elevate_angle, 1, 0, 0);
		glRotated(swing_angle, 0, 1, 0);
		glTranslatef(-center[0], -center[1], -center[2]);
		//Draw_Example();
		glPopMatrix();

		//Also we need to set up the projection matrix for shadow texture	
		// This is matrix transform every coordinate x,y,z
		// Moving from unit cube [-1,1] to [0,1]  
		float bias[16] = { 0.5, 0.0, 0.0, 0.0,
			0.0, 0.5, 0.0, 0.0,
			0.0, 0.0, 0.5, 0.0,
			0.5, 0.5, 0.5, 1.0 };

		// Grab modelview and transformation matrices
		float	modelView[16];
		float	projection[16];
		float	biased_MVP[16];
		glGetFloatv(GL_MODELVIEW_MATRIX, modelView);
		glGetFloatv(GL_PROJECTION_MATRIX, projection);

		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glLoadMatrixf(bias);
		// concatating all matrice into one.
		glMultMatrixf(projection);
		glMultMatrixf(modelView);

		glGetFloatv(GL_MODELVIEW_MATRIX, biased_MVP);

		glUseProgram(shadow_program);
		GLuint m = glGetUniformLocation(shadow_program, "biased_MVP"); // get the location of the biased_MVP matrix
		glUniformMatrix4fv(m, 1, GL_FALSE, biased_MVP);
	}

	static void Get_Selection_Ray(int mouse_x, int mouse_y, float* p, float* q, float ox = 0, float oy = 0, float oz = 0)
	{
		// Convert (x, y) into the 2D unit space
		float new_x = (float)(2 * mouse_x) / (float)screen_width - 1;
		float new_y = 1 - (float)(2 * mouse_y) / (float)screen_height;

		// Convert (x, y) into the 3D viewing space
		float M[16];
		glGetFloatv(GL_PROJECTION_MATRIX, M);
		//M is in column-major but inv_m is in row-major
		float inv_M[16];
		memset(inv_M, 0, sizeof(float) * 16);
		inv_M[0] = 1 / M[0];
		inv_M[5] = 1 / M[5];
		inv_M[14] = 1 / M[14];
		inv_M[11] = -1;
		inv_M[15] = M[10] / M[14];
		float p0[4] = { new_x, new_y, -1, 1 }, p1[4];
		float q0[4] = { new_x, new_y,  1, 1 }, q1[4];
		Matrix_Vector_Product_4(inv_M, p0, p1);
		Matrix_Vector_Product_4(inv_M, q0, q1);

		// Convert (x ,y) into the 3D world space		
		glLoadIdentity();
		glTranslatef(center[0], center[1], center[2]);
		glRotatef(-swing_angle, 0, 1, 0);
		glRotatef(-elevate_angle, 1, 0, 0);
		glTranslatef(ox, oy, oz);
		glTranslatef(0, 0, zoom);
		glGetFloatv(GL_MODELVIEW_MATRIX, M);
		Matrix_Transpose_4(M, M);
		Matrix_Vector_Product_4(M, p1, p0);
		Matrix_Vector_Product_4(M, q1, q0);

		p[0] = p0[0] / p0[3];
		p[1] = p0[1] / p0[3];
		p[2] = p0[2] / p0[3];
		q[0] = q0[0] / q0[3];
		q[1] = q0[1] / q0[3];
		q[2] = q0[2] / q0[3];
	}

	static void Swap(float &a, float &b)
	{
		float c = a; a = b; b = c;
	}

	static void Matrix_Vector_Product_4(float *A, float *x, float *r)
	{
		r[0] = A[0] * x[0] + A[1] * x[1] + A[2] * x[2] + A[3] * x[3];
		r[1] = A[4] * x[0] + A[5] * x[1] + A[6] * x[2] + A[7] * x[3];
		r[2] = A[8] * x[0] + A[9] * x[1] + A[10] * x[2] + A[11] * x[3];
		r[3] = A[12] * x[0] + A[13] * x[1] + A[14] * x[2] + A[15] * x[3];
	}

	static void Matrix_Transpose_4(float *A, float *R)
	{
		if (R != A) memcpy(R, A, sizeof(float) * 16);
		Swap(R[1], R[4]);
		Swap(R[2], R[8]);
		Swap(R[3], R[12]);
		Swap(R[6], R[9]);
		Swap(R[7], R[13]);
		Swap(R[11], R[14]);
	}

	static void Save_To_Image(const string folder, int iterations)
	{
		string dir = "images\\" + folder;
		struct stat st = { 0 };
		if (stat(dir.c_str(), &st) == -1)
			CreateDirectory(dir.c_str(), NULL);

		char imagename[1024];
		float* pixels = new float[screen_width*screen_height * 3];
		glReadPixels(0, 0, screen_width, screen_height, GL_RGB, GL_FLOAT, pixels);
		sprintf_s(imagename, "%s\\example_%04d.bmp", dir.c_str(), iterations);
		BMP_Write(imagename, pixels, screen_width, screen_height);
		delete[] pixels;
	}

	static void Flip_Image(uint8_t *pixels, int channels, int width, int height)
	{
		for (int j = 0; j < height / 2; j++)
			for (int i = 0; i < width; i++) 
				for(int c = 0; c < channels; c++)
					swap(pixels[(j*width + i) * channels + c], pixels[((height - 1 - j)*width + i) * channels + c]);
	}

	static void Save_To_GIF(const string folder, int i_iteration, int iframe)
	{
		string dir = "images\\" + folder;
		struct stat st = { 0 };
		if (stat(dir.c_str(), &st) == -1)
			CreateDirectory(dir.c_str(), NULL);
		
		char fileName[1024];
		sprintf_s(fileName, "%s\\example_%04d.gif", dir.c_str(), i_iteration);

		const static int all_frames = 5;
		static uint8_t** all_pixels = nullptr;
		if (!all_pixels) {
			all_pixels = new uint8_t*[all_frames];
			for (int i = 0; i < all_frames; i++) {
				all_pixels[i] = new uint8_t[screen_width*screen_height * 4];
			}
		}
		
		glReadPixels(0, 0, screen_width, screen_height, GL_RGBA, GL_UNSIGNED_BYTE, all_pixels[iframe-1]);
		if (iframe == all_frames) {
			int delay = 100;
			GifWriter g;
			GifBegin(&g, fileName, screen_width, screen_height, delay);
			for (int i = 0; i < all_frames; i++) {
				Flip_Image(all_pixels[i], 4, screen_width, screen_height);
				GifWriteFrame(&g, all_pixels[i], screen_width, screen_height, delay);
			}
			GifEnd(&g);
		}
	}

	static bool BMP_Write(const char *filename, float *pixels, int width, int height)
	{
		//Preparing the BITMAP data structure from IMAGE object.
		HDC dc = GetDC(NULL);
		BITMAPINFO info;
		ZeroMemory(&info.bmiHeader, sizeof(BITMAPINFOHEADER));
		info.bmiHeader.biWidth = width;
		info.bmiHeader.biHeight = height;
		info.bmiHeader.biPlanes = 1;
		info.bmiHeader.biBitCount = 24;
		info.bmiHeader.biSizeImage = 0;
		info.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
		info.bmiHeader.biClrUsed = 0;
		info.bmiHeader.biClrImportant = 0;
		VOID *pvBits;
		HBITMAP bmp_handle = CreateDIBSection(dc, &info, DIB_RGB_COLORS, &pvBits, NULL, 0);
		BITMAP bitmap;
		GetObject(bmp_handle, sizeof(BITMAP), (LPSTR)&bitmap);

		unsigned char *ptr = (unsigned char *)(bitmap.bmBits);
		for (int j = 0; j<height; j++)
		{
			unsigned char *line_ptr = ptr;
			for (int i = 0; i<width; i++)
			{
				int r = pixels[(j*width + i) * 3 + 0] * 255;
				int g = pixels[(j*width + i) * 3 + 1] * 255;
				int b = pixels[(j*width + i) * 3 + 2] * 255;

				line_ptr[2] = (unsigned char)(r > 255 ? 255 : (r < 0 ? 0 : r));
				line_ptr[1] = (unsigned char)(g > 255 ? 255 : (g < 0 ? 0 : g));
				line_ptr[0] = (unsigned char)(b > 255 ? 255 : (b < 0 ? 0 : b));
				line_ptr += 3;
			}
			ptr += bitmap.bmWidthBytes;
		}

		//Decide the data size.
		WORD wBitCount = 24;
		DWORD dwPaletteSize = 0, dwBmBitsSize, dwDIBSize, dwWritten;
		LPBITMAPINFOHEADER lpbi;
		dwBmBitsSize = ((bitmap.bmWidth *  wBitCount + 31) / 32) * 4 * bitmap.bmHeight;

		//Preparing the palette and the pixel data.
		HANDLE hDib = GlobalAlloc(GHND, dwBmBitsSize + dwPaletteSize + sizeof(BITMAPINFOHEADER));
		lpbi = (LPBITMAPINFOHEADER)GlobalLock(hDib);
		*lpbi = info.bmiHeader;
		HANDLE hPal, hOldPal = NULL;
		hPal = GetStockObject(DEFAULT_PALETTE);
		if (hPal) { hOldPal = SelectPalette(dc, (HPALETTE)hPal, FALSE); RealizePalette(dc); }
		GetDIBits(dc, bmp_handle, 0, (UINT)bitmap.bmHeight, (LPSTR)lpbi + sizeof(BITMAPINFOHEADER) + dwPaletteSize, (BITMAPINFO *)lpbi, DIB_RGB_COLORS);
		if (hOldPal) { SelectPalette(dc, (HPALETTE)hOldPal, TRUE); RealizePalette(dc);	ReleaseDC(NULL, dc); }

		//Start writing the file.
		HANDLE fh = CreateFileA(filename, GENERIC_WRITE, NULL, NULL, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN, NULL);
		if (fh == INVALID_HANDLE_VALUE) return false;
		BITMAPFILEHEADER bmfHdr;
		bmfHdr.bfType = 0x4D42;//"BM"
		dwDIBSize = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + dwPaletteSize + dwBmBitsSize;
		bmfHdr.bfSize = dwDIBSize;
		bmfHdr.bfReserved1 = 0;
		bmfHdr.bfReserved2 = 0;
		bmfHdr.bfOffBits = (DWORD)sizeof(BITMAPFILEHEADER) + (DWORD)sizeof(BITMAPINFOHEADER) + dwPaletteSize;
		WriteFile(fh, (LPSTR)&bmfHdr, sizeof(BITMAPFILEHEADER), &dwWritten, NULL);
		WriteFile(fh, (LPSTR)lpbi, dwDIBSize, &dwWritten, NULL);
		CloseHandle(fh);

		//Deallocation.
		GlobalUnlock(hDib);
		GlobalFree(hDib);
		DeleteObject(bmp_handle);
		return true;
	}
};

#endif