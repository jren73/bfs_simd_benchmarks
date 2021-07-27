#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <omp.h>
#include <string>
#include <unistd.h>
#include <immintrin.h>
#include "../../include/peg_util.h"

using std::string;
using std::to_string;

#define NUM_P_INT 16 // Number of packed intergers in one __m512i variable
#define ALIGNED_BYTES 64

unsigned long long NNODES;
unsigned long long NEDGES;
unsigned long long NUM_THREADS;
unsigned long long TILE_WIDTH;
unsigned long long SIDE_LENGTH;
unsigned long long NUM_TILES;
unsigned long long ROW_STEP;
unsigned long long CHUNK_SIZE;
unsigned long long SIZE_BUFFER_MAX;
unsigned long long T_RATIO;

//double start;
//double now;
//FILE *time_out;
//char *time_file = "timeline.txt";

//// PAPI test results
//static void test_fail(char *file, int line, char *call, int retval){
//	printf("%s\tFAILED\nLine # %d\n", file, line);
//	if ( retval == PAPI_ESYS ) {
//		char buf[128];
//		memset( buf, '\0', sizeof(buf) );
//		sprintf(buf, "System error in %s:", call );
//		perror(buf);
//	}
//	else if ( retval > 0 ) {
//		printf("Error calculating: %s\n", call );
//	}
//	else {
//		printf("Error in %s: %s\n", call, PAPI_strerror(retval) );
//	}
//	printf("\n");
//	exit(1);
//}
//////////////////////////////////////////////////////////////////
// Dense (bottom-up)

void to_dense(
		int *h_graph_mask,
		int *is_active_side,
		unsigned long long *h_graph_queue,
		const unsigned long long &queue_size)

{
	memset(h_graph_mask, 0, NNODES * sizeof(int));
	memset(is_active_side, 0, SIDE_LENGTH * sizeof(int));

	unsigned long long remainder = queue_size % NUM_P_INT;
	unsigned long long bound_i = queue_size - remainder;
#pragma omp parallel for
	for (unsigned long long i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i v_ids_v = _mm512_load_epi32(h_graph_queue + i);
		_mm512_i32scatter_epi32(h_graph_mask, v_ids_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i tw_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(v_ids_v, tw_v);
		_mm512_i32scatter_epi32(is_active_side, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
	if (remainder) {
		__mmask16 in_m = (__mmask16) ((unsigned long long) 0xFFFF >> (NUM_P_INT - remainder));
		__m512i v_ids_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_m, h_graph_queue + bound_i);
		_mm512_mask_i32scatter_epi32(h_graph_mask, in_m, v_ids_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i tw_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_mask_div_epi32(_mm512_undefined_epi32(), in_m, v_ids_v, tw_v);
		_mm512_mask_i32scatter_epi32(is_active_side, in_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
	}
//#pragma omp parallel for
//	for (unsigned long long i = 0; i < queue_size; ++i) {
//		unsigned long long vertex_id = h_graph_queue[i];
//		h_graph_mask[vertex_id] = 1;
//		is_active_side[vertex_id / TILE_WIDTH] = 1;
//	}
}


inline void update_dense(
					unsigned long long &_frontier_size,
					unsigned long long &_out_degree,
					int *h_graph_mask,
					int *h_updating_graph_mask,
					int *is_active_side,
					int *is_updating_active_side,
					unsigned long long *graph_degrees)
{
	unsigned long long frontier_size = 0;
	unsigned long long out_degree = 0;
#pragma omp parallel for reduction(+: frontier_size, out_degree)
	for (unsigned long long side_id = 0; side_id < SIDE_LENGTH; ++side_id) {
		if (!is_updating_active_side[side_id]) {
			is_active_side[side_id] = 0;
			unsigned long long width;
			// Think about this bug. How did you find it? And, more importantly,
			// how to avoid it in the future?
			//	memset(h_graph_mask + side_id * TILE_WIDTH, 0, TILE_WIDTH * sizeof(unsigned long long));
			if (SIDE_LENGTH - 1 != side_id) {
				width = TILE_WIDTH;
			} else {
				width = NNODES - side_id * TILE_WIDTH;
			}
			memset(h_graph_mask + side_id * TILE_WIDTH, 0, width * sizeof(unsigned long long));
			continue;
		}
		is_updating_active_side[side_id] = 0;
		is_active_side[side_id] = 1;
		unsigned long long start_vertex_id = side_id * TILE_WIDTH;
		unsigned long long bound_vertex_id;
		if (SIDE_LENGTH - 1 != side_id) {
			bound_vertex_id = start_vertex_id + TILE_WIDTH;
		} else {
			bound_vertex_id = NNODES;
		}
		unsigned long long remainder = (bound_vertex_id - start_vertex_id) % NUM_P_INT;
		bound_vertex_id -= remainder;
		unsigned long long vertex_id;
		for (vertex_id = start_vertex_id; 
				vertex_id < bound_vertex_id; 
				vertex_id += NUM_P_INT) {
			__m512i updating_flag_v = _mm512_loadu_si512(h_updating_graph_mask + vertex_id);
			__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
			if (!is_updating_m) {
				_mm512_storeu_si512(h_graph_mask + vertex_id, _mm512_set1_epi32(0));
				continue;
			}
			_mm512_mask_storeu_epi32(h_updating_graph_mask + vertex_id, is_updating_m, _mm512_set1_epi32(0));
			_mm512_storeu_si512(h_graph_mask + vertex_id, updating_flag_v);
			__m512i num_active_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), is_updating_m, 1);
			unsigned long long num_active = _mm512_reduce_add_epi32(num_active_v);
			frontier_size += num_active;
			__m512i out_degrees_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), is_updating_m, graph_degrees + vertex_id);
			out_degree += _mm512_reduce_add_epi32(out_degrees_v);
		}

		if (remainder > 0) {
			unsigned long long in_range_m_t = (unsigned long long) 0xFFFF >> (NUM_P_INT - remainder);
			__mmask16 in_range_m = (__mmask16) in_range_m_t;
			__m512i updating_flag_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), in_range_m, h_updating_graph_mask + vertex_id);
			__mmask16 is_updating_m = _mm512_test_epi32_mask(updating_flag_v, _mm512_set1_epi32(-1));
			if (!is_updating_m) {
				_mm512_mask_storeu_epi32(h_graph_mask + vertex_id, in_range_m, _mm512_set1_epi32(0));//addition
				continue;
			}
			_mm512_mask_storeu_epi32(h_updating_graph_mask + vertex_id, is_updating_m, _mm512_set1_epi32(0));
			_mm512_mask_storeu_epi32(h_graph_mask + bound_vertex_id, in_range_m, updating_flag_v);
			//_mm512_storeu_si512(h_graph_mask + vertex_id, updating_flag_v);
			__m512i num_active_v = _mm512_mask_set1_epi32(_mm512_set1_epi32(0), is_updating_m, 1);
			unsigned long long num_active = _mm512_reduce_add_epi32(num_active_v);
			frontier_size += num_active;
			__m512i out_degrees_v = _mm512_mask_loadu_epi32(_mm512_set1_epi32(0), is_updating_m, graph_degrees + vertex_id);
			out_degree += _mm512_reduce_add_epi32(out_degrees_v);
		}
		//for (unsigned long long vertex_id = side_id * TILE_WIDTH; vertex_id < bound_vertex_id; ++vertex_id) {
		//	if (1 == h_updating_graph_mask[vertex_id]) {
		//		h_updating_graph_mask[vertex_id] = 0;
		//		h_graph_mask[vertex_id] = 1;
		//		frontier_size++;
		//		out_degree += graph_degrees[vertex_id];
		//	} else {
		//		h_graph_mask[vertex_id] = 0;
		//	}
		//}
	}
	_frontier_size = frontier_size;
	_out_degree = out_degree;
}

inline void bfs_kernel_dense(
		unsigned long long *heads_buffer,
		unsigned long long *tails_buffer,
		const unsigned long long &size_buffer,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		//int *h_graph_visited,
		unsigned long long *h_graph_parents,
		int *h_cost,
		int *is_updating_active_side)
{
	unsigned long long remainder = size_buffer % NUM_P_INT;
	unsigned long long bound_edge_i = size_buffer - remainder;
	for (unsigned long long edge_i = 0; edge_i < bound_edge_i; edge_i += NUM_P_INT) {
		__m512i head_v = _mm512_load_epi32(heads_buffer + edge_i);
		__m512i active_flag_v = _mm512_i32gather_epi32(head_v, h_graph_mask, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			continue;
		}
		__m512i tail_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + edge_i);
		//__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_visited, sizeof(int));
		//__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(1));
		__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_parents, sizeof(int));
		__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
		if (!not_visited_m) {
			continue;
		}
		__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
		__m512i cost_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
		_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, tail_v, cost_tail_v, sizeof(int));
		_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(tail_v, TILE_WIDTH_v);
		_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
		_mm512_mask_i32scatter_epi32(h_graph_parents, not_visited_m, tail_v, head_v, sizeof(unsigned long long));
	}

	if (remainder) {
		unsigned long long in_range_m_t = (unsigned long long) 0xFFFF >> (NUM_P_INT - remainder);
		__mmask16 in_range_m = (__mmask16) in_range_m_t;
		__m512i head_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, heads_buffer + bound_edge_i);
		__m512i active_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, head_v, h_graph_mask, sizeof(int));
		__mmask16 is_active_m = _mm512_test_epi32_mask(active_flag_v, _mm512_set1_epi32(-1));
		if (!is_active_m) {
			return;
		}
		__m512i tail_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), is_active_m, tails_buffer + bound_edge_i);
		//__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_visited, sizeof(int));
		//__mmask16 not_visited_m = _mm512_testn_epi32_mask(visited_flag_v, _mm512_set1_epi32(1));
		__m512i visited_flag_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(1), is_active_m, tail_v, h_graph_parents, sizeof(int));
		__mmask16 not_visited_m = _mm512_cmpeq_epi32_mask(visited_flag_v, _mm512_set1_epi32(-1));
		if (!not_visited_m) {
			return;
		}
		__m512i cost_head_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), not_visited_m, head_v, h_cost, sizeof(int));
		__m512i cost_tail_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), not_visited_m, cost_head_v, _mm512_set1_epi32(1));
		_mm512_mask_i32scatter_epi32(h_cost, not_visited_m, tail_v, cost_tail_v, sizeof(int));
		_mm512_mask_i32scatter_epi32(h_updating_graph_mask, not_visited_m, tail_v, _mm512_set1_epi32(1), sizeof(int));
		__m512i TILE_WIDTH_v = _mm512_set1_epi32(TILE_WIDTH);
		__m512i side_id_v = _mm512_div_epi32(tail_v, TILE_WIDTH_v);
		_mm512_mask_i32scatter_epi32(is_updating_active_side, not_visited_m, side_id_v, _mm512_set1_epi32(1), sizeof(int));
		_mm512_mask_i32scatter_epi32(h_graph_parents, not_visited_m, tail_v, head_v, sizeof(unsigned long long));
	}
}
inline void scheduler_dense(
		unsigned long long *graph_heads,
		unsigned long long *graph_tails,
		unsigned long long *heads_buffer,
		unsigned long long *tails_buffer,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		unsigned long long *h_graph_parents,
		int *h_cost,
		unsigned long long *tile_offsets,
		unsigned long long *tile_sizes,
		int *is_active_side,
		int *is_updating_active_side,
		const unsigned long long &start_row_index,
		const unsigned long long &tile_step)
{
	unsigned long long start_tile_id = start_row_index * SIDE_LENGTH;
	unsigned long long end_tile_id = start_tile_id + tile_step * SIDE_LENGTH;
#pragma omp parallel for schedule(dynamic, 1)
//#pragma omp parallel for
	for (unsigned long long tile_index = start_tile_id; tile_index < end_tile_id; tile_index += tile_step) {
		unsigned long long bound_tile_id = tile_index + tile_step;
		unsigned long long tid = omp_get_thread_num();
		unsigned long long *heads_buffer_base = heads_buffer + tid * SIZE_BUFFER_MAX;
		unsigned long long *tails_buffer_base = tails_buffer + tid * SIZE_BUFFER_MAX;
		unsigned long long size_buffer = 0;
		unsigned long long capacity = SIZE_BUFFER_MAX;
		for (unsigned long long tile_id = tile_index; tile_id < bound_tile_id; ++tile_id) {
			unsigned long long row_id = (tile_id - start_tile_id) % tile_step + start_row_index;
			if (0 == tile_sizes[tile_id] || !is_active_side[row_id]) {
				continue;
			}
			// Load to buffer
			unsigned long long edge_i = tile_offsets[tile_id];
			unsigned long long remain = tile_sizes[tile_id];
			while (remain != 0) {
				if (capacity > 0) {
					if (capacity > remain) {
						// Put all remain into the buffer
						memcpy(heads_buffer_base + size_buffer, graph_heads + edge_i, remain * sizeof(unsigned long long));
						memcpy(tails_buffer_base + size_buffer, graph_tails + edge_i, remain * sizeof(unsigned long long));
						edge_i += remain;
						capacity -= remain;
						size_buffer += remain;
						remain = 0;
					} else {
						// Fill the buffer to full
						memcpy(heads_buffer_base + size_buffer, graph_heads + edge_i, capacity * sizeof(unsigned long long));
						memcpy(tails_buffer_base + size_buffer, graph_tails + edge_i, capacity * sizeof(unsigned long long));
						edge_i += capacity;
						remain -= capacity;
						size_buffer += capacity;
						capacity = 0;
					}
				} else {
					// Process the full buffer
					bfs_kernel_dense(
							heads_buffer_base,
							tails_buffer_base,
							size_buffer,
							h_graph_mask,
							h_updating_graph_mask,
							//h_graph_visited,
							h_graph_parents,
							h_cost,
							is_updating_active_side);
					capacity = SIZE_BUFFER_MAX;
					size_buffer = 0;
				}
			}
			
		}
		// Process the remains in buffer
		bfs_kernel_dense(
				heads_buffer_base,
				tails_buffer_base,
				size_buffer,
				h_graph_mask,
				h_updating_graph_mask,
				//h_graph_visited,
				h_graph_parents,
				h_cost,
				is_updating_active_side);
	}
}
void BFS_dense(
		unsigned long long *graph_heads,
		unsigned long long *graph_tails,
		int *h_graph_mask,
		int *h_updating_graph_mask,
		//int *h_graph_visited,
		unsigned long long *h_graph_parents,
		int *h_cost,
		unsigned long long *tile_offsets,
		unsigned long long *tile_sizes,
		//int *is_empty_tile,
		int *is_active_side,
		int *is_updating_active_side)
{
	unsigned long long *heads_buffer = (unsigned long long *) _mm_malloc(sizeof(unsigned long long) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);
	unsigned long long *tails_buffer = (unsigned long long *) _mm_malloc(sizeof(unsigned long long) * SIZE_BUFFER_MAX * NUM_THREADS, ALIGNED_BYTES);

	unsigned long long remainder = SIDE_LENGTH % ROW_STEP;
	unsigned long long bound_side_id = SIDE_LENGTH - remainder;
	unsigned long long side_id;
	for (side_id = 0; side_id < bound_side_id; side_id += ROW_STEP) {
		scheduler_dense(
				graph_heads,
				graph_tails,
				heads_buffer,
				tails_buffer,
				h_graph_mask,
				h_updating_graph_mask,
				h_graph_parents,
				h_cost,
				tile_offsets,
				tile_sizes,
				is_active_side,
				is_updating_active_side,
				side_id,
				ROW_STEP);
	}
	if (remainder > 0) {
		scheduler_dense(
				graph_heads,
				graph_tails,
				heads_buffer,
				tails_buffer,
				h_graph_mask,
				h_updating_graph_mask,
				h_graph_parents,
				h_cost,
				tile_offsets,
				tile_sizes,
				is_active_side,
				is_updating_active_side,
				side_id,
				remainder);

	}
	_mm_free(heads_buffer);
	_mm_free(tails_buffer);
}
// End Dense (bottom-up)
///////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////
// Sparse (top-down)
//double offset_time1 = 0;
//double offset_time2 = 0;
//double degree_time = 0;
//double frontier_tmp_time = 0;
//double refine_time = 0;
//double arrange_time = 0;
unsigned long long *to_sparse(
		unsigned long long *frontier,
		const unsigned long long &frontier_size,
		int *h_graph_mask)
{
	unsigned long long *new_frontier = (unsigned long long *) _mm_malloc(frontier_size * sizeof(unsigned long long), ALIGNED_BYTES);

	const unsigned long long block_size = 1 << 12;
	unsigned long long num_blocks = (NNODES - 1)/block_size + 1;
	unsigned long long *nums_in_blocks = nullptr;
	
	if (num_blocks > 1) {
		nums_in_blocks = (unsigned long long *) malloc(num_blocks * sizeof(unsigned long long));
		memset(nums_in_blocks, 0, num_blocks * sizeof(unsigned long long));
		// The start locations where the vertices are put in the frontier.
#pragma omp parallel for
		for (unsigned long long block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned long long offset = block_i * block_size;
			unsigned long long bound;
			if (num_blocks - 1 != block_i) {
				bound = offset + block_size;
			} else {
				bound = NNODES;
			}
			for (unsigned long long vertex_i = offset; vertex_i < bound; ++vertex_i) {
				if (h_graph_mask[vertex_i]) {
					nums_in_blocks[block_i]++;
				}
			}
		}
		//TODO: blocked parallel for
		// Scan to get the offsets as start locations.
		unsigned long long offset_sum = 0;
		for (unsigned long long block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned long long tmp = nums_in_blocks[block_i];
			nums_in_blocks[block_i] = offset_sum;
			offset_sum += tmp;
		}
		// Put vertices into the frontier
#pragma omp parallel for
		for (unsigned long long block_i = 0; block_i < num_blocks; ++block_i) {
			unsigned long long base = nums_in_blocks[block_i];
			unsigned long long offset = block_i * block_size;
			unsigned long long bound;
			if (num_blocks - 1 != block_i) {
				bound = offset + block_size;
			} else {
				bound = NNODES;
			}
			for (unsigned long long vertex_i = offset; vertex_i < bound; ++vertex_i) {
				if (h_graph_mask[vertex_i]) {
					new_frontier[base++] = vertex_i;
				}
			}
		}
		free(nums_in_blocks);
	} else {
		unsigned long long k = 0;
		for (unsigned long long i = 0; i < NNODES; ++i) {
			if (h_graph_mask[i]) {
				new_frontier[k++] = i;
			}
		}
	}
	return new_frontier;
}

inline unsigned long long update_sparse(
				unsigned long long *h_graph_queue,
				const unsigned long long &queue_size,
				unsigned long long *graph_degrees,
				unsigned long long *h_graph_parents,
				int *h_cost)
{
//	unsigned long long out_degree = 0;
//#pragma omp parallel for reduction(+: out_degree)
//	for (unsigned long long i = 0; i < queue_size; ++i) {
//		unsigned long long end = h_graph_queue[i];
//		unsigned long long start = h_graph_parents[end];
//		h_cost[end] = h_cost[start] + 1;
//		out_degree += graph_degrees[end];
//	}
	unsigned long long out_degree = 0;
	unsigned long long remainder = queue_size % NUM_P_INT;
	unsigned long long bound_i = queue_size - remainder;
#pragma omp parallel for reduction(+: out_degree)
	for (unsigned long long i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i vertex_id_v = _mm512_load_epi32(h_graph_queue + i);
		__m512i start_v = _mm512_i32gather_epi32(vertex_id_v, h_graph_parents, sizeof(unsigned long long));
		__m512i cost_start_v = _mm512_i32gather_epi32(start_v, h_cost, sizeof(int));
		cost_start_v = _mm512_add_epi32(cost_start_v, _mm512_set1_epi32(1));
		_mm512_i32scatter_epi32(h_cost, vertex_id_v, cost_start_v, sizeof(int));
		__m512i degrees_v = _mm512_i32gather_epi32(vertex_id_v, graph_degrees, sizeof(unsigned long long));
		unsigned long long sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		out_degree += sum_degrees;
	}
	if (remainder) {
		__mmask16 in_range_m = (__mmask16) ((unsigned long long) 0xffff >> (NUM_P_INT - remainder));
		__m512i vertex_id_v = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, h_graph_queue + bound_i);
		__m512i start_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), in_range_m, vertex_id_v, h_graph_parents, sizeof(unsigned long long));
		__m512i cost_start_v = _mm512_mask_i32gather_epi32(_mm512_undefined_epi32(), in_range_m, start_v, h_cost, sizeof(int));
		cost_start_v = _mm512_mask_add_epi32(_mm512_undefined_epi32(), in_range_m, cost_start_v, _mm512_set1_epi32(1));
		_mm512_mask_i32scatter_epi32(h_cost, in_range_m, vertex_id_v, cost_start_v, sizeof(int));
		__m512i degrees_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_range_m, vertex_id_v, graph_degrees, sizeof(unsigned long long));
		unsigned long long sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		out_degree += sum_degrees;
	}

	return out_degree;
}
inline unsigned long long *BFS_kernel_sparse(
				unsigned long long *graph_vertices,
				unsigned long long *graph_edges,
				unsigned long long *graph_degrees,
				unsigned long long *h_graph_parents,
				unsigned long long *h_graph_queue,
				unsigned long long &queue_size)
{
	// From h_graph_queue, get the degrees (para_for)
	//double time_now = omp_get_wtime(); 
	unsigned long long *degrees = (unsigned long long *) _mm_malloc(sizeof(unsigned long long) *  queue_size, ALIGNED_BYTES);
	unsigned long long new_queue_size = 0;
	unsigned long long remainder = queue_size % NUM_P_INT;
	unsigned long long bound_i = queue_size - remainder;
#pragma omp parallel for reduction(+: new_queue_size)
	for (unsigned long long i = 0; i < bound_i; i += NUM_P_INT) {
		__m512i v_ids = _mm512_load_epi32(h_graph_queue + i);
		__m512i degrees_v = _mm512_i32gather_epi32(v_ids, graph_degrees, sizeof(unsigned long long));
		_mm512_store_epi32(degrees + i, degrees_v);
		unsigned long long sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		new_queue_size += sum_degrees;
	}
	if (remainder) {
		__mmask16 in_m = (__mmask16) ((unsigned long long) 0xffff >> (NUM_P_INT - remainder));
		__m512i v_ids = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_m, h_graph_queue + bound_i);
		__m512i degrees_v = _mm512_mask_i32gather_epi32(_mm512_set1_epi32(0), in_m, v_ids, graph_degrees, sizeof(unsigned long long));
		_mm512_mask_store_epi32(degrees + bound_i, in_m, degrees_v);
		unsigned long long sum_degrees = _mm512_reduce_add_epi32(degrees_v);
		new_queue_size += sum_degrees;
	}
//#pragma omp parallel for reduction(+: new_queue_size)
//	for (unsigned long long i = 0; i < queue_size; ++i) {
//		degrees[i] = graph_degrees[h_graph_queue[i]];
//		new_queue_size += degrees[i];
//	}
	if (0 == new_queue_size) {
		_mm_free(degrees);
		queue_size = 0;
		return nullptr;
	}
	//degree_time += omp_get_wtime() - time_now;

	// From degrees, get the offset (stored in degrees) (block_para_for)
	// TODO: blocked parallel for
	//time_now = omp_get_wtime();
	unsigned long long offset_sum = 0;
	for (unsigned long long i = 0; i < queue_size; ++i) {
		unsigned long long tmp = degrees[i];
		degrees[i] = offset_sum;
		offset_sum += tmp;
	}
	//offset_time1 += omp_get_wtime() - time_now;

	// From offset, get active vertices (para_for)
	//time_now = omp_get_wtime();
	unsigned long long *new_frontier_tmp = (unsigned long long *) _mm_malloc(sizeof(unsigned long long) * new_queue_size, ALIGNED_BYTES);
#pragma omp parallel for schedule(dynamic, CHUNK_SIZE)
	for (unsigned long long i = 0; i < queue_size; ++i) {
		unsigned long long start = h_graph_queue[i];
		unsigned long long offset = degrees[i];
		unsigned long long out_degree = graph_degrees[start];
		unsigned long long base = graph_vertices[start];
		for (unsigned long long k = 0; k < out_degree; ++k) {
			unsigned long long end = graph_edges[base + k];
			if ((unsigned long long) -1 == h_graph_parents[end]) {
				bool unvisited = __sync_bool_compare_and_swap(h_graph_parents + end, (unsigned long long) -1, start); //update h_graph_parents
				if (unvisited) {
					new_frontier_tmp[offset + k] = end;
				} else {
					new_frontier_tmp[offset + k] = (unsigned long long) -1;
				}
			} else {
				new_frontier_tmp[offset + k] = (unsigned long long) -1;
			}
		}
	}
	//frontier_tmp_time += omp_get_wtime() - time_now;


	// Refine active vertices, removing visited and redundant (block_para_for)
	//unsigned long long block_size = new_queue_size / NUM_THREADS;
	//time_now = omp_get_wtime();
	unsigned long long block_size = 1024 * 2;
	//unsigned long long num_blocks = new_queue_size % block_size == 0 ? new_queue_size/block_size : new_queue_size/block_size + 1;
	unsigned long long num_blocks = (new_queue_size - 1)/block_size + 1;

	unsigned long long *nums_in_blocks = NULL;
	if (num_blocks > 1) {
	//unsigned long long *nums_in_blocks = (unsigned long long *) malloc(sizeof(unsigned long long) * NUM_THREADS);
	nums_in_blocks = (unsigned long long *) malloc(sizeof(unsigned long long) * num_blocks);
	unsigned long long new_frontier_size_tmp = 0;
//#pragma omp parallel for schedule(dynamic) reduction(+: new_frontier_size_tmp)
#pragma omp parallel for reduction(+: new_frontier_size_tmp)
	for (unsigned long long block_i = 0; block_i < num_blocks; ++block_i) {
		unsigned long long offset = block_i * block_size;
		unsigned long long bound;
		if (num_blocks - 1 != block_i) {
			bound = offset + block_size;
		} else {
			bound = new_queue_size;
		}
		//unsigned long long size = 0;
		unsigned long long base = offset;
		for (unsigned long long end_i = offset; end_i < bound; ++end_i) {
			if ((unsigned long long) - 1 != new_frontier_tmp[end_i]) {
				new_frontier_tmp[base++] = new_frontier_tmp[end_i];
			}
		}
		nums_in_blocks[block_i] = base - offset;
		new_frontier_size_tmp += nums_in_blocks[block_i];
	}
	new_queue_size = new_frontier_size_tmp;
	} else {
		unsigned long long base = 0;
		for (unsigned long long i = 0; i < new_queue_size; ++i) {
			if ((unsigned long long) -1 != new_frontier_tmp[i]) {
				new_frontier_tmp[base++] = new_frontier_tmp[i];
			}
		}
		new_queue_size = base;
	}
	//refine_time += omp_get_wtime() - time_now;
	
	if (0 == new_queue_size) {
		//free(offsets);
		//free(frontier_vertices);
		_mm_free(degrees);
		_mm_free(new_frontier_tmp);
		free(nums_in_blocks);
		queue_size = 0;
		return nullptr;
	}

	// Get the final new h_graph_queue
	//time_now = omp_get_wtime();
	unsigned long long *new_frontier = (unsigned long long *) _mm_malloc(sizeof(unsigned long long) * new_queue_size, ALIGNED_BYTES);
	if (num_blocks > 1) {
	//TODO: blocked parallel for
	//double time_now = omp_get_wtime();
	offset_sum = 0;
	for (unsigned long long i = 0; i < num_blocks; ++i) {
		unsigned long long tmp = nums_in_blocks[i];
		nums_in_blocks[i] = offset_sum;
		offset_sum += tmp;
		//offsets_b[i] = offsets_b[i - 1] + nums_in_blocks[i - 1];
	}
	//offset_time2 += omp_get_wtime() - time_now;
//#pragma omp parallel for schedule(dynamic)
#pragma omp parallel for
	for (unsigned long long block_i = 0; block_i < num_blocks; ++block_i) {
		//unsigned long long offset = offsets_b[block_i];
		unsigned long long offset = nums_in_blocks[block_i];
		unsigned long long bound;
		if (num_blocks - 1 != block_i) {
			bound = nums_in_blocks[block_i + 1];
		} else {
			bound = new_queue_size;
		}
		unsigned long long base = block_i * block_size;
		unsigned long long remainder = (bound - offset) % NUM_P_INT;
		unsigned long long bound_i = bound - remainder;
		for (unsigned long long i = offset; i < bound_i; i += NUM_P_INT) {
			__m512i tmp = _mm512_load_epi32(new_frontier_tmp + base);
			_mm512_storeu_si512(new_frontier + i, tmp);
			base += NUM_P_INT;
		}
		if (remainder) {
			__mmask16 in_range_m = (__mmask16) ((unsigned long long) 0xffff >> (NUM_P_INT - remainder));
			__m512i tmp = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, new_frontier_tmp + base);
			_mm512_mask_storeu_epi32(new_frontier + bound_i, in_range_m, tmp);
		}
		//for (unsigned long long i = offset; i < bound; ++i) {
		//	new_frontier[i] = new_frontier_tmp[base++];
		//}
	}
	} else {
		unsigned long long remainder = new_queue_size % NUM_P_INT;
		unsigned long long bound_i = new_queue_size - remainder;
		for (unsigned long long i = 0; i < bound_i; i += NUM_P_INT) {
			__m512i tmp = _mm512_load_epi32(new_frontier_tmp + i);
			_mm512_store_epi32(new_frontier + i, tmp);
		}
		if (remainder) {
			__mmask16 in_range_m = (__mmask16) ((unsigned long long) 0xffff >> (NUM_P_INT - remainder));
			__m512i tmp = _mm512_mask_load_epi32(_mm512_undefined_epi32(), in_range_m, new_frontier_tmp + bound_i);
			_mm512_mask_store_epi32(new_frontier + bound_i, in_range_m, tmp);
		}
		//unsigned long long base = 0;
		//for (unsigned long long i = 0; i < new_queue_size; ++i) {
		//	new_frontier[i] = new_frontier_tmp[base++];
		//}
	}
	//arrange_time += omp_get_wtime() - time_now;

	// Return the results
	//free(frontier_vertices);
	_mm_free(degrees);
	_mm_free(new_frontier_tmp);
	free(nums_in_blocks);
	queue_size = new_queue_size;
	return new_frontier;
}
inline unsigned long long *BFS_sparse(
		unsigned long long *frontier,
		unsigned long long *graph_vertices,
		unsigned long long *graph_edges,
		unsigned long long *graph_degrees,
		unsigned long long *h_graph_parents,
		unsigned long long &frontier_size)
{

	return BFS_kernel_sparse(
				graph_vertices,
				graph_edges,
				graph_degrees,
				h_graph_parents,
				frontier,
				frontier_size);
}
// End Sparse (top-down)
///////////////////////////////////////////////////////////////////////////////


double dense_time;
double to_dense_time;
double sparse_time;
double to_sparse_time;
double update_time;
double run_time;

void print_time()
{
	auto percent = [=] (double t) {
		return t/run_time * 100.0;
	};
	printf("dense_time: %f (%.2f%%)\n", dense_time, percent(dense_time));
	printf("to_dense_time: %f (%.2f%%)\n", to_dense_time, percent(to_dense_time));
	printf("sparse_time: %f (%.2f%%)\n", sparse_time, percent(sparse_time));
	printf("to_sparse_time: %f (%.2f%%)\n", to_sparse_time, percent(to_sparse_time));
	printf("update_time: %f (%.2f%%)\n", update_time, percent(update_time));
	printf("==========================\n");
}

void graph_prepare(
		unsigned long long *graph_vertices,
		unsigned long long *graph_edges,
		unsigned long long *graph_heads,
		unsigned long long *graph_tails,
		unsigned long long *graph_degrees,
		unsigned long long *tile_offsets,
		unsigned long long *tile_sizes,
		const unsigned long long &source,
		int max_hop)
{
	int hop = 0;
	dense_time = 0;
	to_dense_time = 0;
	sparse_time = 0;
	to_sparse_time = 0;
	update_time = 0;
	run_time = 0;

	// Set up
	omp_set_num_threads(NUM_THREADS);
	int *h_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_updating_graph_mask = (int*) malloc(sizeof(int)*NNODES);
	int *h_cost = (int*) malloc(sizeof(int)*NNODES);
	int *is_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	int *is_updating_active_side = (int *) malloc(sizeof(int) * SIDE_LENGTH);
	unsigned long long *h_graph_parents = (unsigned long long *) malloc(sizeof(unsigned long long) * NNODES);
	printf("memory alocation finished!\n");
	memset(h_graph_mask, 0, sizeof(int)*NNODES);
	memset(h_updating_graph_mask, 0, sizeof(int)*NNODES);
#pragma omp parallel for num_threads(64)
	for (unsigned long long j = 0; j < NNODES; ++j) {
		h_cost[j] = -1;
	}
	h_cost[source] = 0;
	memset(is_active_side, 0, sizeof(int) * SIDE_LENGTH);
	memset(is_updating_active_side, 0, sizeof(int) * SIDE_LENGTH);
#pragma omp parallel for num_threads(64)
	for (unsigned long long j = 0; j < NNODES; ++j) {
		h_graph_parents[j] = (unsigned long long) -1; // means unvisited yet
	}
	h_graph_parents[source] = source;

	// The first time, running the Sparse
	unsigned long long frontier_size = 1;
	unsigned long long *frontier = (unsigned long long *) _mm_malloc(sizeof(unsigned long long) * frontier_size, ALIGNED_BYTES);
	frontier[0] = source;
	//// PAPI
	//int events[2] = { PAPI_L2_TCA, PAPI_L2_TCM};
	//int retval;
	//if ((retval = PAPI_start_counters(events, 2)) < PAPI_OK) {
	//	test_fail(__FILE__, __LINE__, "PAPI_start_counters", retval);
	//}
	//// End PAPI
	printf("finish init mem\n");
	double last_time = omp_get_wtime();
	double start_time = omp_get_wtime();
	unsigned long long *new_frontier = BFS_sparse(
								frontier,
								graph_vertices,
								graph_edges,
								graph_degrees,
								h_graph_parents,
								frontier_size);
	_mm_free(frontier);
	frontier = new_frontier;
	sparse_time += omp_get_wtime() - last_time;
	last_time = omp_get_wtime();
	
	
	printf("%llu\n",frontier_size);
	unsigned long long out_degree = 0;
	// When update the parents, get the sum of the number of active nodes and their out degree.
#pragma omp parallel for reduction(+: out_degree)
	for (unsigned long long i = 0; i < frontier_size; ++i) {
		unsigned long long end = frontier[i];
		if(end<0 || end>= NNODES)
		//   end=1;
		   continue;
		unsigned long long start = h_graph_parents[end];
		//printf("%llu %llu %llu\n",i, start, end);
		//if(start == 0)
		//   start =1;
		
		h_cost[end] = h_cost[start] + 1;
		out_degree += graph_degrees[end];
	}
	//printf("I'm here?\n");
	update_time += omp_get_wtime() - last_time;
	bool last_is_dense = false;
	// According the sum, determine to run Sparse or Dense, and then change the last_is_dense.
	//unsigned long long bfs_threshold = NEDGES / 20 / T_RATIO; // Determined according to Ligra
	printf("I'm here\n");
	unsigned long long bfs_threshold = NEDGES / T_RATIO; // Determined according to Ligra
	
	while (hop <= max_hop) {
		++hop;
		if (frontier_size + out_degree > bfs_threshold) {
			// Dense
			if (!last_is_dense) {
				last_time = omp_get_wtime();
				to_dense(
					h_graph_mask, 
					is_active_side, 
					frontier, 
					frontier_size);
				to_dense_time += omp_get_wtime() - last_time;
			}
			last_time = omp_get_wtime();
			BFS_dense(
					graph_heads,
					graph_tails,
					h_graph_mask,
					h_updating_graph_mask,
					h_graph_parents,
					h_cost,
					tile_offsets,
					tile_sizes,
					//is_empty_tile,
					is_active_side,
					is_updating_active_side);
			dense_time += omp_get_wtime() - last_time;
			last_is_dense = true;
		} else {
			// Sparse
			if (last_is_dense) {
				last_time = omp_get_wtime();
				new_frontier = to_sparse(
					frontier,
					frontier_size,
					h_graph_mask);
				_mm_free(frontier);
				frontier = new_frontier;
				to_sparse_time += omp_get_wtime() - last_time;
			}
			last_time = omp_get_wtime();
			new_frontier = BFS_sparse(
								frontier,
								graph_vertices,
								graph_edges,
								graph_degrees,
								h_graph_parents,
								frontier_size);
			_mm_free(frontier);
			frontier = new_frontier;
			last_is_dense = false;
			sparse_time += omp_get_wtime() - last_time;
		}
		// Update the parents, also get the sum again.
		last_time = omp_get_wtime();
		if (last_is_dense) {
			update_dense(
					frontier_size,
					out_degree,
					h_graph_mask,
					h_updating_graph_mask,
					is_active_side,
					is_updating_active_side,
					graph_degrees);
			
			if (0 == frontier_size) {
				break;
			}
		} else {
			if (0 == frontier_size) {
				break;
			}
			out_degree = update_sparse(
								frontier,
								frontier_size,
								graph_degrees,
								h_graph_parents,
								h_cost);
		}
		update_time += omp_get_wtime() - last_time;
		//frontier_size--;
		printf("frontier_size: %llu\n", frontier_size);//test
	}
	double end_time = omp_get_wtime();
	printf("I'm here\n");
	//// PAPI results
	//long long values[2];
	//if ((retval = PAPI_stop_counters(values, 2)) < PAPI_OK) {
	//	test_fail(__FILE__, __LINE__, "PAPI_stop_counters", retval);
	//}
	//printf("cache access: %lld, cache misses: %lld, miss rate: %.2f%%\n", values[0], values[1], 100.0* values[1]/values[0]);
	//// End PAPI results
	printf("%llu %lf\n", NUM_THREADS, run_time = (end_time - start_time));
	bot_best_perform.record(run_time, NUM_THREADS);
	print_time();//test
	
	//Store the result into a file

#ifdef ONEDEBUG
	NUM_THREADS = 64;
	omp_set_num_threads(NUM_THREADS);
	unsigned long long num_lines = NNODES / NUM_THREADS;
#pragma omp parallel
{
	unsigned long long tid = omp_get_thread_num();
	unsigned long long offset = tid * num_lines;
	string file_prefix = "path/path";
	string file_name = file_prefix + to_string(tid) + ".txt";
	FILE *fpo = fopen(file_name.c_str(), "w");
	if (!fpo) {
		fprintf(stderr, "Error: cannot open file %s.\n", file_name.c_str());
		exit(1);
	}
	unsigned long long bound_index;
	if (tid != NUM_THREADS - 1) {
		bound_index = offset + num_lines;
	} else {
		bound_index = NNODES;
	}
	for (unsigned long long index = offset; index < bound_index; ++index) {
		fprintf(fpo, "%d) cost:%d\n", index, h_cost[index]);
	}

	fclose(fpo);
}
#endif

	free(frontier);
	free( h_graph_mask);
	free( h_updating_graph_mask);
	free( h_graph_parents);
	free( h_cost);
	free( is_active_side);
	free( is_updating_active_side);
}

///////////////////////////////////////////////////////////////////////////////
// Input data and then apply BFS
///////////////////////////////////////////////////////////////////////////////
void graph_input(
			char *input_f,
			unsigned long long *&graph_heads,
			unsigned long long *&graph_tails,
			unsigned long long *&tile_offsets,
			unsigned long long *&tile_sizes,
			unsigned long long *&graph_vertices,
			unsigned long long *&graph_edges,
			unsigned long long *&graph_degrees)
{
	/////////////////////////////////////////////////////////////////////
	// Input real dataset
	/////////////////////////////////////////////////////////////////////
	//string prefix = string(input_f) + "_untiled";
	//string prefix = string(input_f) + "_coo-tiled-" + to_string(TILE_WIDTH);
	//string file_name_pre = string(input_f) + "_reorder";
	string file_name_pre = string(input_f);
	string prefix = file_name_pre + "_col-" + to_string(ROW_STEP) + "-coo-tiled-" + to_string(TILE_WIDTH);
	//string prefix = string(input_f) + "_col-2-coo-tiled-" + to_string(TILE_WIDTH);
	string fname = prefix + "-0";
	FILE *fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
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
		fscanf(fin, "%llu", tile_offsets + i);
	}
	fclose(fin);
	tile_sizes = (unsigned long long *) malloc(NUM_TILES * sizeof(unsigned long long));
	for (unsigned long long i = 0; i < NUM_TILES; ++i) {
		if (NUM_TILES - 1 != i) {
			tile_sizes[i] = tile_offsets[i + 1] - tile_offsets[i];
		} else {
			tile_sizes[i] = NEDGES - tile_offsets[i];
		}
	}
	graph_heads = (unsigned long long *) malloc(sizeof(unsigned long long) * NEDGES);
	graph_tails = (unsigned long long *) malloc(sizeof(unsigned long long) * NEDGES);

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
		fscanf(fin, "%llu %llu", &NNODES, &NEDGES);
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
		graph_tails[index] = n2;
	}
	fclose(fin);

}


	// For Sparse
	prefix = file_name_pre + "_untiled";
	graph_edges = (unsigned long long *) malloc(sizeof(unsigned long long) * NEDGES);
	graph_degrees = (unsigned long long *) malloc(sizeof(unsigned long long) * NNODES);

	// Read degrees
	fname = prefix + "-nneibor";
	fin = fopen(fname.c_str(), "r");
	if (!fin) {
		fprintf(stderr, "cannot open file: %s\n", fname.c_str());
		exit(1);
	}
	for (unsigned long long i = 0; i < NNODES; ++i) {
		fscanf(fin, "%llu", graph_degrees + i);
	}
	fclose(fin);

	NUM_THREADS = 64;
	edge_bound = NEDGES / NUM_THREADS;
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
		fscanf(fin, "%llu %llu", &NNODES, &NEDGES);
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
		graph_edges[index] = n2;
	}
	fclose(fin);
}
	// CSR
	graph_vertices = (unsigned long long *) malloc(sizeof(unsigned long long) * NNODES);
	unsigned long long edge_start = 0;
	for (unsigned long long i = 0; i < NNODES; ++i) {
		graph_vertices[i] = edge_start;
		edge_start += graph_degrees[i];
	}
	// End Input real dataset
	/////////////////////////////////////////////////////////////////////


}
///////////////////////////////////////////////////////////////////////////////
// Main Program
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
	printf("testing...\n");
	char *input_f;
	
	if(argc > 3){
		input_f = argv[1];
		TILE_WIDTH = strtoul(argv[2], NULL, 0);
		ROW_STEP = strtoul(argv[3], NULL, 0);
	} else {
		//input_f = "/home/zpeng/benchmarks/data/pokec_combine/soc-pokec";
		////input_f = "/sciclone/scr-mlt/zpeng01/pokec_combine/soc-pokec";
		//TILE_WIDTH = 1024;
		//ROW_STEP = 16;
		printf("Usage: ./bfs <data> <tile_width> <stripe_length>\n");
		exit(1);
	}
        printf("start bfs ...\n");
	// Input
	unsigned long long *graph_heads = nullptr;
	unsigned long long *graph_tails = nullptr;
	unsigned long long *tile_offsets = nullptr;
	unsigned long long *tile_sizes = nullptr;
	unsigned long long *graph_vertices = nullptr;
	unsigned long long *graph_edges = nullptr;
	unsigned long long *graph_degrees = nullptr;
        printf("Input start\n");
	//input( argc, argv);
	graph_input(
			input_f,
			graph_heads,
			graph_tails,
			tile_offsets,
			tile_sizes,
			graph_vertices,
			graph_edges,
			graph_degrees);

	unsigned long long source = 0;
        printf("Input finished: %s\n", input_f);
#ifdef ONEDEBUG
	//printf("Input finished: %s\n", input_f);
	unsigned long long run_count = 7;
#else
	unsigned long long run_count = 7;
#endif
	// BFS
	SIZE_BUFFER_MAX = 1024;
	//SIZE_BUFFER_MAX = 512;
	//T_RATIO = 100;
	T_RATIO = 20;
	CHUNK_SIZE = 2048;
	int k_set = 10;
	//printf("tile_size: %llu\n", TILE_WIDTH);
	//printf("stripe_length: %llu\n", ROW_STEP);
	int max_hop=5;
	for (unsigned long long i = 6; i < run_count; ++i) {
		NUM_THREADS = (unsigned long long) pow(2, i);
		bot_best_perform.reset();
	//NUM_THREADS = 64;
#ifndef ONEDEBUG
		//sleep(10);
#endif
		// Re-initializing
		//for (unsigned long long k = 0; k < k_set; ++k) {

		graph_prepare(
				graph_vertices,
				graph_edges,
				graph_heads,
				graph_tails, 
				graph_degrees,
				tile_offsets,
				tile_sizes,
				source,
				max_hop);
//#ifdef ONEDEBUG
		printf("Thread %llu finished.\n", NUM_THREADS);
//#endif
		//}
		bot_best_perform.print_average(NUM_THREADS);
	}

	// cleanup memory
	free( graph_heads);
	free( graph_tails);
	free( graph_edges);
	free( graph_degrees);
	free( graph_vertices);
	free( tile_offsets);
	free( tile_sizes);
}

