#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <omp.h>
#include <unistd.h>
#include <algorithm>
using std::string;
using std::getline;
using std::cout;
using std::endl;
using std::to_string;
using std::vector;

unsigned long long NNODES;
unsigned long long NEDGES;
unsigned long long NUM_THREADS;
unsigned long long TILE_WIDTH; // Width of tile
unsigned long long SIDE_LENGTH; // Number of rows of tiles
unsigned long long NUM_TILES; // Number of tiles
unsigned long long ROW_STEP; // Number of rows of tiles in a Group

unsigned long long KCORE;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";

void input_weighted(
		char filename[], 
		unsigned long long *&graph_heads, 
		unsigned long long *&graph_ends, 
		unsigned long long *&graph_weights,
		unsigned long long *&tile_offsets,
		unsigned long long *&nneibor,
		int *&is_empty_tile) 
{
	//printf("data: %s\n", filename);
	//string prefix = string(filename) + "_untiled";
	string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%llu %llu", &NNODES, &NEDGES);
	fclose(fin);
	if (NNODES % TILE_WIDTH) {
		SIDE_LENGTH = NNODES / TILE_WIDTH + 1;
	} else {
		SIDE_LENGTH = NNODES / TILE_WIDTH;
	}
	NUM_TILES = SIDE_LENGTH * SIDE_LENGTH;
	// Read tile Offsets
	fname = prefix + "-offsets";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	tile_offsets = (unsigned long long *) malloc(NUM_TILES * sizeof(unsigned long long));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		//fscanf(fin, "%llu", tile_offsets + i);
		unsigned long long offset;
		fscanf(fin, "%llu", &offset);
		tile_offsets[i] = offset;
	}
	fclose(fin);
	// Read degrees
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	nneibor = (unsigned long long *) malloc(NNODES * sizeof(unsigned long long));
	for (unsigned long long i = 0; i < NNODES; ++i) {
		fscanf(fin, "%llu", nneibor + i);
	}
	fclose(fin);

	graph_heads = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	graph_ends = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	graph_weights = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			if (tile_offsets[i] == tile_offsets[i + 1]) {
				is_empty_tile[i] = 1;
			}
		} else {
			if (tile_offsets[i] == NEDGES) {
				is_empty_tile[i] = 1;
			}
		}
	}
	NUM_THREADS = 64;
	unsigned edge_bound = NEDGES / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
{
	unsigned tid = omp_get_thread_num();
	unsigned long long offset = tid * edge_bound;
	fname = prefix + "-" + to_string(tid);
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
		exit(1);
	}
	if (0 == tid) {
		fscanf(fin, "%llu %llu\n", &NNODES, &NEDGES);
	}
	unsigned long long bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned long long index = offset; index < bound_index; ++index) {
		unsigned long long n1;
		unsigned long long n2;
		unsigned long long wt;
		fscanf(fin, "%llu %llu %llu", &n1, &n2, &wt);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_ends[index] = n2;
		graph_weights[index] = wt;
	}

	fclose(fin);
}
}

void input(
		char filename[], 
		unsigned long long *&graph_heads, 
		unsigned long long *&graph_ends, 
		unsigned long long *&tile_offsets,
		unsigned long long *&nneibor,
		int *&is_empty_tile) 
{
	//printf("data: %s\n", filename);
	//string prefix = string(filename) + "_untiled";
	string prefix = string(filename) + "_coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", fname.c_str());
		exit(1);
	}
	fscanf(fin, "%llu %llu", &NNODES, &NEDGES);
	fclose(fin);
	if (NNODES % TILE_WIDTH) {
		SIDE_LENGTH = NNODES / TILE_WIDTH + 1;
	} else {
		SIDE_LENGTH = NNODES / TILE_WIDTH;
	}
	NUM_TILES = SIDE_LENGTH * SIDE_LENGTH;
	// Read tile Offsets
	fname = prefix + "-offsets";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	tile_offsets = (unsigned long long *) malloc(NUM_TILES * sizeof(unsigned long long));
	for (unsigned long long i = 0; i < NUM_TILES; ++i) {
		//fscanf(fin, "%llu", tile_offsets + i);
		unsigned long long offset;
		fscanf(fin, "%llu", &offset);
		tile_offsets[i] = offset;
	}
	fclose(fin);
	// Read degrees
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	nneibor = (unsigned long long *) malloc(NNODES * sizeof(unsigned long long));
	for (unsigned long long i = 0; i < NNODES; ++i) {
		fscanf(fin, "%llu", nneibor + i);
	}
	fclose(fin);

	graph_heads = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	graph_ends = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	is_empty_tile = (int *) malloc(sizeof(int) * NUM_TILES);
	memset(is_empty_tile, 0, sizeof(int) * NUM_TILES);
	for (unsigned long long i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			if (tile_offsets[i] == tile_offsets[i + 1]) {
				is_empty_tile[i] = 1;
			}
		} else {
			if (tile_offsets[i] == NEDGES) {
				is_empty_tile[i] = 1;
			}
		}
	}
	NUM_THREADS = 64;
	unsigned long long edge_bound = NEDGES / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
{
	unsigned long long tid = omp_get_thread_num();
	unsigned long long offset = tid * edge_bound;
	fname = prefix + "-" + to_string(tid);
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
		exit(1);
	}
	if (0 == tid) {
		fscanf(fin, "%llu %llu\n", &NNODES, &NEDGES);
	}
	unsigned long long bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned long long index = offset; index < bound_index; ++index) {
		unsigned long long n1;
		unsigned long long n2;
		fscanf(fin, "%llu %llu", &n1, &n2);
		if(n1==0)
		  n1++;
		if(n2==0)
		  n2++;
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_ends[index] = n2;
	}

	fclose(fin);
}
}

void convert_to_col_major_weight(
						char *filename,
						unsigned long long *graph_heads, 
						unsigned long long *graph_ends, 
						unsigned long long *graph_weights,
						unsigned long long *tile_offsets,
						unsigned long long *nneibor)
{
	unsigned long long *new_heads = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	unsigned long long *new_ends = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	unsigned long long *new_weights = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	unsigned long long *new_offsets = (unsigned long long *) malloc(NUM_TILES * sizeof(unsigned long long));
	//unsigned long long step = 16;
	//unsigned long long step = 2;//test
	unsigned long long edge_index = 0;
	unsigned long long new_tile_id = 0;
	unsigned long long side_i = 0;

	printf("Converting...\n");

	for (side_i = 0; side_i + ROW_STEP <= SIDE_LENGTH; side_i += ROW_STEP) {
		for (unsigned long long col = 0; col < SIDE_LENGTH; ++col) {
			for (unsigned long long row = side_i; row < side_i + ROW_STEP; ++row) {
				unsigned long long tile_id = row * SIDE_LENGTH + col;
				unsigned long long bound_edge_i;
				if (NUM_TILES - 1 != tile_id) {
					bound_edge_i = tile_offsets[tile_id + 1];
				} else {
					bound_edge_i = NEDGES;
				}
				new_offsets[new_tile_id++] = edge_index;
				for (unsigned long long edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					new_heads[edge_index] = graph_heads[edge_i] + 1;
					new_ends[edge_index] = graph_ends[edge_i] + 1;
					new_weights[edge_index] = graph_weights[edge_i];
					++edge_index;
				}
			}
		}
	}
	if (side_i != SIDE_LENGTH) {
		for (unsigned long long col = 0; col < SIDE_LENGTH; ++col) {
			for (unsigned long long row = side_i; row < SIDE_LENGTH; ++row) {
				unsigned long long tile_id = row * SIDE_LENGTH + col;
				unsigned long long bound_edge_i;
				if (NUM_TILES - 1 != tile_id) {
					bound_edge_i = tile_offsets[tile_id + 1];
				} else {
					bound_edge_i = NEDGES;
				}
				new_offsets[new_tile_id++] = edge_index;
				for (unsigned long long edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					new_heads[edge_index] = graph_heads[edge_i] + 1;
					new_ends[edge_index] = graph_ends[edge_i] + 1;
					new_weights[edge_index] = graph_weights[edge_i];
					++edge_index;
				}
			}
		}
	}
	printf("Finally, edge_index: %llu (NEDGES: %llu), new_tile_id: %llu (NUM_TILES: %llu)\n", edge_index, NEDGES, new_tile_id, NUM_TILES);

	// Write to files
	string prefix = string(filename) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
	NUM_THREADS = 64;
	unsigned long long edge_bound = NEDGES / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS)
{
	unsigned long long tid = omp_get_thread_num();
	unsigned long long offset = tid * edge_bound;
	string fname = prefix + "-" + to_string(tid);
	FILE *fout = fopen(fname.c_str(), "w");
	if (0 == tid) {
		fprintf(fout, "%llu %llu\n", NNODES, NEDGES);
	}
	unsigned long long bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned long long index = offset; index < bound_index; ++index) {
		fprintf(fout, "%llu %llu %llu\n", new_heads[index], new_ends[index], new_weights[index]);
	}
	fclose(fout);
}
	printf("Main files done...\n");
	// Write offsets
	string fname = prefix + "-offsets";
	FILE *fout = fopen(fname.c_str(), "w");
	for (unsigned long long i = 0; i < NUM_TILES; ++i) {
		fprintf(fout, "%llu\n", new_offsets[i]);//Format: offset
	}
	fclose(fout);
	fname = prefix + "-nneibor";
	fout = fopen(fname.c_str(), "w");
	for (unsigned long long i = 0; i < NNODES; ++i) {
		fprintf(fout, "%llu\n", nneibor[i]);
	}
	printf("Done.\n");

	//// test
	//fout = fopen("output.txt", "w");
	//fprintf(fout, "%llu %llu\n", NNODES, NEDGES);
	//for (unsigned long long i = 0; i < NEDGES; ++i) {
	//	fprintf(fout, "%llu %llu\n", new_heads[i], new_ends[i]);
	//}
	//fclose(fout);

	free(new_heads);
	free(new_ends);
	free(new_weights);
}

void convert_to_col_major(
						char *filename,
						unsigned long long *graph_heads, 
						unsigned long long *graph_ends, 
						unsigned long long *tile_offsets,
						unsigned long long *nneibor)
{
	unsigned long long *new_heads = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	unsigned long long *new_ends = (unsigned long long *) malloc(NEDGES * sizeof(unsigned long long));
	unsigned long long *new_offsets = (unsigned long long *) malloc(NUM_TILES * sizeof(unsigned long long));
	//unsigned long long step = 16;
	//unsigned long long step = 2;//test
	unsigned long long edge_index = 0;
	unsigned long long new_tile_id = 0;
	unsigned long long side_i = 0;

	printf("Converting...\n");

	for (side_i = 0; side_i + ROW_STEP <= SIDE_LENGTH; side_i += ROW_STEP) {
		for (unsigned long long col = 0; col < SIDE_LENGTH; ++col) {
			for (unsigned long long row = side_i; row < side_i + ROW_STEP; ++row) {
				unsigned long long tile_id = row * SIDE_LENGTH + col;
				unsigned long long  bound_edge_i;
				if (NUM_TILES - 1 != tile_id) {
					bound_edge_i = tile_offsets[tile_id + 1];
				} else {
					bound_edge_i = NEDGES;
				}
				new_offsets[new_tile_id++] = edge_index;
				for (unsigned long long edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					new_heads[edge_index] = graph_heads[edge_i] + 1;
					new_ends[edge_index] = graph_ends[edge_i] + 1;
					++edge_index;
				}
			}
		}
	}
	if (side_i != SIDE_LENGTH) {
		for (unsigned long long col = 0; col < SIDE_LENGTH; ++col) {
			for (unsigned long long row = side_i; row < SIDE_LENGTH; ++row) {
				unsigned long long tile_id = row * SIDE_LENGTH + col;
				unsigned long long bound_edge_i;
				if (NUM_TILES - 1 != tile_id) {
					bound_edge_i = tile_offsets[tile_id + 1];
				} else {
					bound_edge_i = NEDGES;
				}
				new_offsets[new_tile_id++] = edge_index;
				for (unsigned long long edge_i = tile_offsets[tile_id]; edge_i < bound_edge_i; ++edge_i) {
					new_heads[edge_index] = graph_heads[edge_i] + 1;
					new_ends[edge_index] = graph_ends[edge_i] + 1;
					++edge_index;
				}
			}
		}
	}
	printf("Finally, edge_index: %llu (NEDGES: %llu), new_tile_id: %llu (NUM_TILES: %llu)\n", edge_index, NEDGES, new_tile_id, NUM_TILES);

	// Write to files
	string prefix = string(filename) + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
	NUM_THREADS = 64;
	unsigned long long edge_bound = NEDGES / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS)
{
	unsigned long long tid = omp_get_thread_num();
	unsigned long long offset = tid * edge_bound;
	string fname = prefix + "-" + to_string(tid);
	FILE *fout = fopen(fname.c_str(), "w");
	if (0 == tid) {
		fprintf(fout, "%llu %llu\n", NNODES, NEDGES);
	}
	unsigned long long bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned long long index = offset; index < bound_index; ++index) {
		fprintf(fout, "%llu %llu\n", new_heads[index], new_ends[index]);
	}
	fclose(fout);
}
	printf("Main files done...\n");
	// Write offsets
	string fname = prefix + "-offsets";
	FILE *fout = fopen(fname.c_str(), "w");
	for (unsigned long long i = 0; i < NUM_TILES; ++i) {
		fprintf(fout, "%llu\n", new_offsets[i]);//Format: offset
	}
	fclose(fout);
	fname = prefix + "-nneibor";
	fout = fopen(fname.c_str(), "w");
	for (unsigned long long i = 0; i < NNODES; ++i) {
		fprintf(fout, "%llu\n", nneibor[i]);
	}
	printf("Done.\n");

	//// test
	//fout = fopen("output.txt", "w");
	//fprintf(fout, "%llu %llu\n", NNODES, NEDGES);
	//for (unsigned i = 0; i < NEDGES; ++i) {
	//	fprintf(fout, "%llu %llu\n", new_heads[i], new_ends[i]);
	//}
	//fclose(fout);

	free(new_heads);
	free(new_ends);
}

int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	unsigned long long min_row_step;
	unsigned long long max_row_step;

	if (argc > 4) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
		//ROW_STEP = strtoul(argv[3], NULL, 0);
		min_row_step = strtoul(argv[3], NULL, 0);
		max_row_step = strtoul(argv[4], NULL, 0);
	} else {
		////filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		//filename = "/home/zpeng/benchmarks/data/skitter/out.skitter";
		//TILE_WIDTH = 1024;
		//ROW_STEP = 16;
		printf("Usage: ./kcore <data_file> <tile_width> <min_stripe_length> <max_stripe_length>\n");
		exit(1);
	}
	// Input
	unsigned long long *graph_heads;
	unsigned long long *graph_ends;
	unsigned long long *tile_offsets;
	unsigned long long *nneibor;
	int *is_empty_tile;

	unsigned long long *graph_weights = nullptr;

#ifdef WEIGHTED
	input_weighted(
		filename, 
		graph_heads, 
		graph_ends, 
		graph_weights,
		tile_offsets,
		nneibor,
		is_empty_tile);

	for (ROW_STEP = min_row_step; ROW_STEP <= max_row_step; ROW_STEP *= 2) {
	convert_to_col_major_weight(
						filename,
						graph_heads, 
						graph_ends, 
						graph_weights,
						tile_offsets,
						nneibor);
	}
#else
	input(
		filename, 
		graph_heads, 
		graph_ends, 
		tile_offsets,
		nneibor,
		is_empty_tile);

	for (ROW_STEP = min_row_step; ROW_STEP <= max_row_step; ROW_STEP *= 2) {
	convert_to_col_major(
						filename,
						graph_heads, 
						graph_ends, 
						tile_offsets,
						nneibor);
	}
#endif



	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(tile_offsets);
	if (nullptr != graph_weights) {
		free(graph_weights);
	}

	return 0;
}
