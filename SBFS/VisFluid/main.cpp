#define  _CRT_SECURE_NO_DEPRECATE

#include "OpenGL_Driver.h"

#include <stdio.h> 

void Read_Input_CPP(const string& filename)
{
	vector<vector<float>> dataMatrix;
	fstream file;
	string line;
	file.open(filename);
	if (!file.is_open()) { printf("ERROR: cannot open %s\n", filename.c_str()); getchar(); }
	while (getline(file, line, '\n'))
	{
		vector<float> dataLine;
		stringstream ssline(line);
		string token;
		while (getline(ssline, token, ' '))
			dataLine.push_back(atof(token.c_str()));
		dataMatrix.push_back(dataLine);
	}
	file.close();

	std::cout << dataMatrix.size() << "," << dataMatrix[0].size() << std::endl;
	getchar();
}

void Read_Input_C(const char* filename)
{
	const int cols = 204;
	const int rows = 204;

	float *dataMatrix[rows];
	for (int i = 0; i < rows; i++)
		dataMatrix[i] = (float *)malloc(cols * sizeof(float));

	FILE *fp = 0;
	fp = fopen(filename, "r+");
	if (fp == 0) { printf("ERROR: cannot open %s\n", filename); getchar(); }

	for (int r = 0; r < rows; r++)
	{
		for (int c = 0; c < cols; c++)
		{
			fscanf(fp, "%f", &dataMatrix[r][c]);
		}
		char temp;
		//fscanf(fp, "\n", &temp);
	}

	printf("%f\n", dataMatrix[1][2]);
	getchar();
}

int main(int argc, char *argv[])
{

	//Read_Input_CPP("input.txt");
	//Read_Input_C("input.txt");

	OpenGL_Driver(&argc, argv);
	return 0;
}
