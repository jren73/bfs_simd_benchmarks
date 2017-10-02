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

unsigned NNODES;
unsigned NEDGES;
unsigned NUM_THREADS;
unsigned TILE_WIDTH; // Width of tile
unsigned SIDE_LENGTH; // Number of rows of tiles
unsigned NUM_TILES; // Number of tiles
unsigned ROW_STEP; // Number of rows of tiles in a Group

unsigned KCORE;

double start;
double now;
FILE *time_out;
char *time_file = "timeline.txt";


void input(
		char filename[], 
		unsigned *&graph_heads, 
		unsigned *&graph_ends, 
		unsigned *&graph_degrees,
		unsigned *&tile_offsets,
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
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
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
	tile_offsets = (unsigned *) malloc(NUM_TILES * sizeof(unsigned));
	for (unsigned i = 0; i < NUM_TILES; ++i) {
		//fscanf(fin, "%u", tile_offsets + i);
		unsigned offset;
		fscanf(fin, "%u", &offset);
		tile_offsets[i] = offset;
	}
	fclose(fin);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
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
	graph_degrees = (unsigned *) malloc(NNODES * sizeof(unsigned));
	memset(graph_degrees, 0, NNODES * sizeof(unsigned));
	NUM_THREADS = 64;
	unsigned edge_bound = NEDGES / NUM_THREADS;
#pragma omp parallel num_threads(NUM_THREADS) private(fname, fin)
{
	unsigned tid = omp_get_thread_num();
	unsigned offset = tid * edge_bound;
	fname = prefix + "-" + to_string(tid);
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s\n", fname.c_str());
		exit(1);
	}
	if (0 == tid) {
		fscanf(fin, "%u %u\n", &NNODES, &NEDGES);
	}
	unsigned bound_index;
	if (NUM_THREADS - 1 != tid) {
		bound_index = offset + edge_bound;
	} else {
		bound_index = NEDGES;
	}
	for (unsigned index = offset; index < bound_index; ++index) {
		unsigned n1;
		unsigned n2;
		fscanf(fin, "%u %u", &n1, &n2);
		n1--;
		n2--;
		graph_heads[index] = n1;
		graph_ends[index] = n2;
		graph_degrees[n1]++;
	}

	fclose(fin);
}
}

void input_serial(char filename[], unsigned *&graph_heads, unsigned *&graph_ends, unsigned *&graph_degrees)
{
	//printf("data: %s\n", filename);
	FILE *fin = fopen(filename, "r");
	if (!fin) {
		fprintf(stderr, "Error: cannot open file %s.\n", filename);
		exit(1);
	}
	fscanf(fin, "%u %u", &NNODES, &NEDGES);
	graph_heads = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_ends = (unsigned *) malloc(NEDGES * sizeof(unsigned));
	graph_degrees = (unsigned *) calloc(NNODES, sizeof(unsigned));
	for (unsigned i = 0; i < NEDGES; ++i) {
		unsigned head;
		unsigned end;
		fscanf(fin, "%u %u", &head, &end);
		--head;
		--end;
		graph_heads[i] = head;
		graph_ends[i] = end;
		graph_degrees[head]++;
		//graph_degrees[end]++;
	}
	fclose(fin);
}

void print(unsigned *graph_cores) {
	FILE *foutput = fopen("ranks.txt", "w");
	unsigned kc = 0;
	for (unsigned i = 0; i < NNODES; ++i) {
		fprintf(foutput, "%u: %u\n", i+1, graph_cores[i]);
		if (kc < graph_cores[i]) {
			kc = graph_cores[i];
		}
	}
	fprintf(foutput, "kc: %u, KCORE: %u\n", kc, KCORE);
}
//unsigned test_count = 0;//test
inline void kcore_kernel(
				unsigned *graph_heads, 
				unsigned *graph_ends,
				unsigned *graph_degrees,
				int *graph_updating_active, 
				unsigned *graph_cores,
				const unsigned &edge_i_start, 
				const unsigned &edge_i_bound)
{
	for (unsigned edge_i = edge_i_start; edge_i < edge_i_bound; ++edge_i) {
		unsigned head = graph_heads[edge_i];
		unsigned end = graph_ends[edge_i];
		if (graph_updating_active[head] && graph_degrees[end]) {
			graph_degrees[end]--;
			if (!graph_degrees[end]) {
				graph_cores[end] = KCORE - 1;
				//test_count++;//test
			}
		}
	}
}
inline void scheduler(
					unsigned *graph_heads, 
					unsigned *graph_ends, 
					unsigned *graph_degrees,
					unsigned *tile_offsets,
					int *graph_updating_active,
					int *is_empty_tile,
					unsigned *graph_cores,
					const unsigned &start_row_index,
					const unsigned &bound_row_index)
{
#pragma omp parallel for schedule(dynamic, 1)
	for (unsigned col_id = 0; col_id < SIDE_LENGTH; ++col_id) {
		for (unsigned row_id = start_row_index; row_id < bound_row_index; ++row_id) {
			unsigned tile_id = row_id * SIDE_LENGTH + col_id;
			if (is_empty_tile[tile_id]) {
				continue;
			}
			//bfs_kernel();
			unsigned bound_edge_i;
			if (NUM_TILES - 1 != tile_id) {
				bound_edge_i = tile_offsets[tile_id + 1];
			} else {
				bound_edge_i = NEDGES;
			}
			kcore_kernel(
				graph_heads, 
				graph_ends, 
				graph_degrees,
				graph_updating_active,
				graph_cores,
				tile_offsets[tile_id], 
				bound_edge_i);
		}
	}
}
void kcore(
		unsigned *graph_heads, 
		unsigned *graph_ends, 
		unsigned *graph_degrees,
		unsigned *tile_offsets,
		int *graph_updating_active,
		int *is_updating_active_side,
		int *is_empty_tile,
		unsigned *graph_cores)
{
	omp_set_num_threads(NUM_THREADS);
	double start_time = omp_get_wtime();
	int stop = 0;
	//test_count = 0;
	while (!stop) {
		stop = 1;
		int has_remove = 1;
		KCORE++;
		while (has_remove) {
			//double ts = omp_get_wtime();
			has_remove = 0;
//#pragma omp parallel for schedule(dynamic, 1)
#pragma omp parallel for
			for (unsigned i = 0; i < NNODES; ++i) {
				if (graph_degrees[i]) {
					stop = 0;
					if(graph_degrees[i] < KCORE) {
						graph_updating_active[i] = 1;
						is_updating_active_side[i/TILE_WIDTH] = 1;
						graph_degrees[i] = 0;
						graph_cores[i] = KCORE - 1;
						//test_count++;//test
						has_remove = 1;
					}
				}
			}
			//double ts2 = omp_get_wtime();
			//printf("time for vertices: %lf\n", ts2 - ts);
			unsigned side_id;
			for (side_id = 0; side_id + ROW_STEP <= SIDE_LENGTH; ) {
				if (!is_updating_active_side[side_id]) {
					++side_id;
					continue;
				}
				scheduler(
						graph_heads, 
						graph_ends, 
						graph_degrees,
						tile_offsets,
						graph_updating_active,
						is_empty_tile,
						graph_cores,
						side_id,
						side_id + ROW_STEP);
				side_id += ROW_STEP;
			}
			scheduler(
					graph_heads, 
					graph_ends, 
					graph_degrees,
					tile_offsets,
					graph_updating_active,
					is_empty_tile,
					graph_cores,
					side_id,
					SIDE_LENGTH);
			//kcore_kernel(
			//		graph_heads, 
			//		graph_ends, 
			//		graph_degrees,
			//		graph_updating_active, 
			//		0, 
			//		NEDGES,
			//		graph_cores);
			//printf("time for edges: %lf\n", omp_get_wtime() - ts2);
			memset(graph_updating_active, 0, NNODES * sizeof(int));
			memset(is_updating_active_side, 0, SIDE_LENGTH * sizeof(int));
		}
		//printf("test_count: %u, KCORE: %u\n", test_count, KCORE);//test
	}
	KCORE -= 2;

	double end_time = omp_get_wtime();
	printf("%u %lf\n", NUM_THREADS, end_time - start_time);
}


int main(int argc, char *argv[]) 
{
	start = omp_get_wtime();
	char *filename;
	if (argc > 2) {
		filename = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
	} else {
		//filename = "/home/zpeng/benchmarks/data/pokec/soc-pokec";
		filename = "/home/zpeng/benchmarks/data/skitter/out.skitter";
		TILE_WIDTH = 1024;
	}
	// Input
	unsigned *graph_heads;
	unsigned *graph_ends;
	unsigned *graph_degrees;
	unsigned *tile_offsets;
	int *is_empty_tile;
#ifdef ONESERIAL
	//input_serial("/home/zpeng/benchmarks/data/fake/data.txt", graph_heads, graph_ends, graph_degrees);
	//input_serial("/home/zpeng/benchmarks/data/fake/mun_twitter", graph_heads, graph_ends,graph_degrees);
	input_serial("/home/zpeng/benchmarks/data/zebra/out.zebra", graph_heads, graph_ends,graph_degrees);
#else
	input(
		filename, 
		graph_heads, 
		graph_ends, 
		graph_degrees,
		tile_offsets,
		is_empty_tile);
#endif

	// K-core
	int *graph_updating_active = (int *) malloc(NNODES * sizeof(int));
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned *graph_cores = (unsigned *) malloc(NNODES * sizeof(unsigned));
	unsigned *graph_degrees_bak = (unsigned *) malloc(NNODES * sizeof(unsigned));
	memcpy(graph_degrees_bak, graph_degrees, NNODES * sizeof(unsigned));
	
	now = omp_get_wtime();
	time_out = fopen(time_file, "w");
	fprintf(time_out, "input end: %lf\n", now - start);
#ifdef ONEDEBUG
	unsigned run_count = 9;
	printf("Start K-core...\n");
#else
	unsigned run_count = 9;
#endif
	//for (unsigned s = 1; s < 2048; s *= 2) {
	//ROW_STEP = s;
	printf("ROW_STEP: %u\n", ROW_STEP);
	//ROW_STEP = 16;
	for (unsigned i = 0; i < run_count; ++i) {
		NUM_THREADS = (unsigned) pow(2, i);
		memset(graph_updating_active, 0, NNODES * sizeof(int));
		memset(is_updating_active_side, 0, SIDE_LENGTH * sizeof(int));
		for (unsigned k = 0; k < NNODES; ++k) {
			graph_cores[k] = 0;
		}
		KCORE = 0;
		memcpy(graph_degrees, graph_degrees_bak, NNODES * sizeof(unsigned));
		//sleep(10);
		kcore(
			graph_heads, 
			graph_ends, 
			graph_degrees,
			tile_offsets,
			graph_updating_active,
			is_updating_active_side,
			is_empty_tile,
			graph_cores);
		now = omp_get_wtime();
		fprintf(time_out, "Thread %u end: %lf\n", NUM_THREADS, now - start);
	}
	//}
	fclose(time_out);
#ifdef ONEDEBUG
	print(graph_cores);
#endif

	// Free memory
	free(graph_heads);
	free(graph_ends);
	free(graph_degrees);
	free(tile_offsets);
	free(graph_degrees_bak);
	free(graph_updating_active);
	free(is_updating_active_side);
	free(is_empty_tile);
	free(graph_cores);

	return 0;
}