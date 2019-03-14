#ifndef __MANTA_WRAPPER_H__
#define __MANTA_WRAPPER_H__

static const int STR_LEN_GRID = 252;
//! uni file header, v4, (note - uni files always store single prec. values)
typedef struct {
	int dimX, dimY, dimZ; // grid size
	int gridType, elementType, bytesPerElement; // data type info
	char info[STR_LEN_GRID]; // mantaflow build information
	int dimT;                // optionally store forth dimension for 4d grids
	unsigned long long timestamp; // creation time
} UniHeader;


static const int STR_LEN_PDATA = 256;
//! pdata uni header, v3  (similar to grid header)
typedef struct {
	int dim; // number of partilces
	int dimX, dimY, dimZ; // underlying solver resolution (all data in local coordinates!)
	int elementType, bytesPerElement; // type id and byte size
	char info[STR_LEN_PDATA]; // mantaflow build information
	unsigned long long timestamp; // creation time
} UniPartHeader;


#endif