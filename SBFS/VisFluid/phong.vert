attribute vec4 position; 
attribute vec4 normal; 
attribute vec4 color; 


varying vec3 fragNormal;
varying vec4 fragPosition;
varying vec4 fragColor;


void main()
{ 
	
	fragNormal = normalize(gl_NormalMatrix * vec3(normal));
	
	gl_Position = gl_ModelViewProjectionMatrix * position; 

	fragPosition = gl_Position;
	fragColor = color;
}
