/**
 *  @file Huffman.h
 *  @author Sheng Di
 *  @date Aug., 2016
 *  @brief Header file for the exponential segment constructor.
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef RIPPLES_BITMAP_H
#define RIPPLES_BITMAP_H


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <memory>
#include <set>

namespace ripples {

void prtbits(unsigned int a){
    for (int i = 0; i < 32; i++) {
        printf("%d", !!((a << i) & 0x80000000));
    }
    printf("\n");
}

template <typename InItr>
void encodeRR23(InItr in_begin, size_t local_idx, size_t length, size_t n_ints, unsigned int* code_array){
	size_t i = 0;
	size_t vtx_id = 0, byte_offset=0, bit_offset=0;
	unsigned int m = 1;
	for (i = 0;i<length;i++) 
	{
		vtx_id = *(in_begin->begin()+i);
		byte_offset = local_idx / 32;
		bit_offset  = local_idx % 32; 
		m = 1 << bit_offset;
		code_array[vtx_id*n_ints+byte_offset] |= m; //vtx start from zero
	}
}

void countRR(std::vector<std::vector<unsigned int*>> &blockR, const size_t n_vtx, const size_t n_ints,
			 size_t *local_m, size_t *local_v){
	size_t num_threads = omp_get_max_threads();
	size_t n_blocks = blockR.size();
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) //shared(n_vtx)
  	{
  		size_t i = 0, j = 0, k = 0, blk = 0;
  		size_t rank = omp_get_thread_num();
	    int* localcnt=(int*)calloc(n_vtx,sizeof(int));
	    unsigned int* local_r; // = localR[rank];
	    size_t local_max = 0, local_vtx = 0;
#pragma omp for schedule(static)  		    
		for (i = 0;i<n_vtx;i++) 
		{
			for(blk = 0; blk < n_blocks; blk++){
				// auto localR = blockR[blk];
				for(k=0; k<num_threads; k++){
					// local_r = localR[k];
					for (j=0; j<n_ints; j++){
						// localcnt[i] += __builtin_popcount(local_r[i*n_ints + j]);
						localcnt[i] += __builtin_popcount(blockR[blk][k][i*n_ints + j]);
					}
				}
			}
			if (localcnt[i] > local_max){
				local_max = localcnt[i];
				local_vtx = i;
			}
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
	    free(localcnt);		
	}
}

void countRR2(std::vector<std::vector<unsigned int*>> &blockR1, std::vector<std::vector<unsigned int*>> &blockR2, 
			 const size_t n_vtx, const size_t n_ints1, const size_t n_ints2,
			 size_t *local_m, size_t *local_v){
	size_t num_threads = omp_get_max_threads();
	size_t n_blocks1 = blockR1.size(), n_blocks2 = blockR2.size();
	std::cout<<" countRR2 nvtx="<<n_vtx<<" nint1="<<n_ints1 << " nint2=" << n_ints2 << std::endl;
	//*************************************************************
	//* update bitmap : minus u^* from all vtx and popcount again *
	//*************************************************************
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) 
  	{
  		size_t i = 0, j = 0, k = 0, blk = 0;
  		size_t rank = omp_get_thread_num();
	    int* localcnt=(int*)calloc(n_vtx,sizeof(int));
	    unsigned int* local_r; // = localR[rank];
	    unsigned int tmp1 = 0, tmp2 = 0;
	    size_t local_max = 0, local_vtx = 0;
#pragma omp for schedule(static)  		    
		for (i = 0;i<n_vtx;i++) 
		{
			for(blk = 0; blk < n_blocks1; blk++){
				for(k=0; k<num_threads; k++){
					for (j=0; j<n_ints1; j++){
						localcnt[i] += __builtin_popcount(blockR1[blk][k][i*n_ints1 + j]);
					}

				}
			}
			for(blk = 0; blk < n_blocks2; blk++){
				for(k=0; k<num_threads; k++){
					for(j=0; j<n_ints2; j++){
						localcnt[i] += __builtin_popcount(blockR2[blk][k][i*n_ints2 + j]);
					}
				}
			}
			if (localcnt[i] > local_max){
				local_max = localcnt[i];
				local_vtx = i;
			}
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
		#pragma omp critical
	    {
	    	size_t tmp_key = 0;
			for(int r=0;r<num_threads;r++){
				tmp_key = local_v[r];
				std::cout<<tmp_key<<","<<localcnt[tmp_key]<<", ";
			}
			std::cout<<std::endl;
	    }
	    free(localcnt);		
	}
}

void resetAccu(std::vector<std::vector<unsigned int*>> &blockR, const size_t n_vtx, const size_t n_ints){
	size_t num_threads = omp_get_max_threads();
	size_t n_blocks = blockR.size();
	size_t i = 0, j = 0, k = 0, blk = 0;
	for(blk = 0; blk < n_blocks; blk++){
		for(k=0; k<num_threads; k++){
			for (j=0; j<n_ints; j++){
				blockR[blk][k][n_vtx*n_ints + j] &= 0;
			}
		}
	}
}

void selectRR0(std::vector<std::vector<unsigned int*>> &blockR, const size_t n_vtx, const size_t n_ints,
			 size_t *local_m, size_t *local_v, bool *deleteflag, size_t &maxk, size_t &maxv, size_t seedidx){
	size_t num_threads = omp_get_max_threads();
	size_t n_blocks = blockR.size();
	maxk=0; maxv = 0;
	for(int i = 0; i < num_threads; i++ ) {
	    if (local_m[i] > maxv) {
	        maxk = local_v[i];
	        maxv = local_m[i];
	    }
	}
	deleteflag[maxk] = 1;
	std::cout<<" selectRR0 mvtx="<<maxk<<" mmax="<<maxv << std::endl;
	//*************************************************************
	//* update bitmap : minus u^* from all vtx and popcount again *
	//*************************************************************
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close)  shared(maxk)
  	{
  		size_t i = 0, j = 0, k = 0, blk = 0;
  		size_t rank = omp_get_thread_num();
	    int* localcnt=(int*)calloc(n_vtx,sizeof(int));
	    unsigned int* local_r; // = localR[rank];
	    unsigned int tmp1 = 0, tmp2 = 0, tmp3 = 0;
	    size_t local_max = 0, local_vtx = 0;
	    // std::cout<<" selectRR mvtx="<<maxk<<" mmax="<<maxv << " #blocks=" << n_blocks << std::endl;
	    
#pragma omp for schedule(static)  		    
		for (i = 0;i<n_vtx;i++) 
		{
			if(deleteflag[i]== 0){
				for(blk = 0; blk < n_blocks; blk++){
					for(k=0; k<num_threads; k++){
						for (j=0; j<n_ints; j++){
							tmp1 = blockR[blk][k][i*n_ints + j] ^ blockR[blk][k][maxk*n_ints + j];
							blockR[blk][k][i*n_ints + j] &= tmp1;
							localcnt[i] += __builtin_popcount(blockR[blk][k][i*n_ints + j]);
						}
					}
				}
				if (localcnt[i] == 0){
					deleteflag[i] = 1;
				}
				if (localcnt[i] > local_max){
					local_max = localcnt[i];
					local_vtx = i;
				}
			}	
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
	    free(localcnt);		
	}
}

void selectRR02(std::vector<std::vector<unsigned int*>> &blockR1, std::vector<std::vector<unsigned int*>> &blockR2, 
			 const size_t n_vtx, const size_t n_ints1, const size_t n_ints2,
			 size_t *local_m, size_t *local_v, bool *deleteflag, size_t &maxk, size_t &maxv, size_t seedidx){
	size_t num_threads = omp_get_max_threads();
	size_t n_blocks1 = blockR1.size(), n_blocks2 = blockR2.size();
	maxk=0; maxv = 0;
	for(int i = 0; i < num_threads; i++ ) {
	    if (local_m[i] > maxv) {
	        maxk = local_v[i];
	        maxv = local_m[i];
	    }
	}
	deleteflag[maxk] = 1;
	// std::cout<<" selectRR mvtx="<<maxk<<" mmax="<<maxv << " #blocks=" << n_blocks << std::endl;
	//*************************************************************
	//* update bitmap : minus u^* from all vtx and popcount again *
	//*************************************************************
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) shared(maxk)
  	{
  		size_t i = 0, j = 0, k = 0, blk = 0;
  		size_t rank = omp_get_thread_num();
	    int* localcnt=(int*)calloc(n_vtx,sizeof(int));
	    unsigned int* local_r; // = localR[rank];
	    unsigned int tmp1 = 0, tmp2 = 0;
	    size_t local_max = 0, local_vtx = 0;
	    
#pragma omp for schedule(static)  		    
		for (i = 0;i<n_vtx;i++) 
		{
			if(deleteflag[i]== 0){
				for(blk = 0; blk < n_blocks1; blk++){
					for(k=0; k<num_threads; k++){
						for (j=0; j<n_ints1; j++){
							tmp1 = blockR1[blk][k][i*n_ints1 + j] ^ blockR1[blk][k][maxk*n_ints1 + j];
							blockR1[blk][k][i*n_ints1 + j] &= tmp1;
							localcnt[i] += __builtin_popcount(blockR1[blk][k][i*n_ints1 + j]);
						}

					}
				}
				for(blk = 0; blk < n_blocks2; blk++){
					for(k=0; k<num_threads; k++){
						for(j=0; j<n_ints2; j++){
							tmp1 = blockR2[blk][k][i*n_ints2 + j] ^ blockR2[blk][k][maxk*n_ints2 + j];
							blockR2[blk][k][i*n_ints2 + j] &= tmp1;
							localcnt[i] += __builtin_popcount(blockR2[blk][k][i*n_ints2 + j]);
						}

					}
				}
				if (localcnt[i] == 0){
					deleteflag[i] = 1;
				}
				if (localcnt[i] > local_max){
					local_max = localcnt[i];
					local_vtx = i;
				}
			}	
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
	    free(localcnt);		
	}
}

void selectRR(std::vector<std::vector<unsigned int*>> &blockR, const size_t n_vtx, const size_t n_ints,
			 size_t *local_m, size_t *local_v, bool *deleteflag, size_t &maxk, size_t &maxv, size_t seedidx){
	size_t num_threads = omp_get_max_threads();
	size_t n_blocks = blockR.size();
	maxk=0; maxv = 0;
	for(int i = 0; i < num_threads; i++ ) {
	    if (local_m[i] > maxv) {
	        maxk = local_v[i];
	        maxv = local_m[i];
	    }
	}
	deleteflag[maxk] = 1;
	// std::cout<<" selectRR mvtx="<<maxk<<" mmax="<<maxv << " #blocks=" << n_blocks << std::endl;
	//*************************************************************
	//* update bitmap : minus u^* from all vtx and popcount again *
	//*************************************************************
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) shared(maxk)
  	{
  		size_t i = 0, j = 0, k = 0, blk = 0;
  		size_t rank = omp_get_thread_num();
	    int* localcnt=(int*)calloc(n_vtx,sizeof(int));
	    unsigned int* local_r; // = localR[rank];
	    unsigned int tmp1 = 0, tmp2 = 0, tmp3 = 0;
	    size_t local_max = 0, local_vtx = 0;
	    // std::cout<<" selectRR mvtx="<<maxk<<" mmax="<<maxv << " #blocks=" << n_blocks << std::endl;
#pragma omp for schedule(static)  		    
		for (i = 0;i<n_vtx;i++) 
		{
			if(deleteflag[i]== 0){
				for(blk = 0; blk < n_blocks; blk++){
					for(k=0; k<num_threads; k++){
						for (j=0; j<n_ints; j++){
							blockR[blk][k][n_vtx*n_ints + j] |= blockR[blk][k][maxk*n_ints + j];
							tmp1 = blockR[blk][k][i*n_ints + j] ^ blockR[blk][k][n_vtx*n_ints + j];
							tmp2 = blockR[blk][k][i*n_ints + j] & tmp1;
							localcnt[i] += __builtin_popcount(tmp2);
						}
					}
				}
				if (localcnt[i] == 0){
					deleteflag[i] = 1;
				}
				if (localcnt[i] > local_max){
					local_max = localcnt[i];
					local_vtx = i;
				}
			}	
		}
		for(blk = 0; blk < n_blocks; blk++){
			for(k=0; k<num_threads; k++){
				for (j=0; j<n_ints; j++){
					blockR[blk][k][n_vtx*n_ints + j] &= 0;
				}
			}
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
	    free(localcnt);		
	}
}

void selectRR2(std::vector<std::vector<unsigned int*>> &blockR1, std::vector<std::vector<unsigned int*>> &blockR2, 
			 const size_t n_vtx, const size_t n_ints1, const size_t n_ints2,
			 size_t *local_m, size_t *local_v, bool *deleteflag, size_t &maxk, size_t &maxv, size_t seedidx){
	size_t num_threads = omp_get_max_threads();
	size_t n_blocks1 = blockR1.size(), n_blocks2 = blockR2.size();
	maxk=0; maxv = 0;
	for(int i = 0; i < num_threads; i++ ) {
	    if (local_m[i] > maxv) {
	        maxk = local_v[i];
	        maxv = local_m[i];
	    }
	}
	deleteflag[maxk] = 1;
	// std::cout<<" selectRR mvtx="<<maxk<<" mmax="<<maxv << " #blocks=" << n_blocks << std::endl;
	//*************************************************************
	//* update bitmap : minus u^* from all vtx and popcount again *
	//*************************************************************
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) shared(maxk)
  	{
  		size_t i = 0, j = 0, k = 0, blk = 0;
  		size_t rank = omp_get_thread_num();
	    int* localcnt=(int*)calloc(n_vtx,sizeof(int));
	    unsigned int* local_r; // = localR[rank];
	    unsigned int tmp1 = 0, tmp2 = 0;
	    size_t local_max = 0, local_vtx = 0;
	    
#pragma omp for schedule(static)  		    
		for (i = 0;i<n_vtx;i++) 
		{
			if(deleteflag[i]== 0){
				for(blk = 0; blk < n_blocks1; blk++){
					for(k=0; k<num_threads; k++){
						for (j=0; j<n_ints1; j++){
							blockR1[blk][k][n_vtx*n_ints1 + j] |= blockR1[blk][k][maxk*n_ints1 + j];
							tmp1 = blockR1[blk][k][i*n_ints1 + j] ^ blockR1[blk][k][n_vtx*n_ints1 + j];
							tmp2 = blockR1[blk][k][i*n_ints1 + j] & tmp1;
							localcnt[i] += __builtin_popcount(tmp2);
						}

					}
				}
				for(blk = 0; blk < n_blocks2; blk++){
					for(k=0; k<num_threads; k++){
						for(j=0; j<n_ints2; j++){
							blockR2[blk][k][n_vtx*n_ints2 + j] |= blockR2[blk][k][maxk*n_ints2 + j];
							tmp1 = blockR2[blk][k][i*n_ints2 + j] ^ blockR2[blk][k][n_vtx*n_ints2 + j];
							tmp2 = blockR2[blk][k][i*n_ints2 + j] & tmp1;
							localcnt[i] += __builtin_popcount(tmp2);
						}

					}
				}
				if (localcnt[i] == 0){
					deleteflag[i] = 1;
				}
				if (localcnt[i] > local_max){
					local_max = localcnt[i];
					local_vtx = i;
				}
			}	
		}
		local_m[rank]=local_max;
		local_v[rank]=local_vtx;
	    free(localcnt);		
	}
}

template <typename vertex_type, typename RRRset>
void bitmapRRRSets2(std::vector<RRRset> &RRRsets, const int block_idx, const int blockoffset, 
				   std::vector<std::vector<unsigned int*>> &blockR,
				   std::vector<std::vector<unsigned int*>> &blockR_bkp, 
				   const size_t n_vtx, size_t* n_ints) {
  	size_t s1 = RRRsets.size(); //i: current RR's index
  	// std::cout<<" s1="<<s1<<" block-offset="<<blockoffset << std::endl;
  	size_t total_rrr_size = 0;
  	size_t num_threads = omp_get_max_threads();
  	blockR[block_idx].resize(num_threads);
  	blockR_bkp[block_idx].resize(num_threads);
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) //shared(n_vtx)
  	{
  		size_t rank = omp_get_thread_num();
  		size_t n_rows = n_vtx+1; // number of vtx +1, the last is reserved to accumulate bitmap of seeds
  		size_t n_cols = (s1 - blockoffset) / num_threads; // number of RRs
  		if ((s1 - blockoffset) % num_threads > 0) {
  			n_cols += 1;
  		}
  		*n_ints =  n_cols / 32;
  		if (n_cols % 32 > 0){
  			*n_ints += 1;
  		} 
  		blockR[block_idx][rank] = (unsigned int*)malloc((*n_ints)*n_rows*sizeof(unsigned int));
  		memset(blockR[block_idx][rank], 0, (*n_ints)*n_rows*sizeof(unsigned int));
#pragma omp for reduction(+:total_rrr_size) schedule(static)//(dynamic)//
	    for (size_t i=blockoffset; i<s1; i++) {
	        auto in_begin=RRRsets.begin();
	        std::advance(in_begin, i);
	        size_t s2=std::distance(in_begin->begin(),in_begin->end());
	        size_t local_idx = i % n_cols;
	        encodeRR23(in_begin, local_idx, s2, *n_ints, blockR[block_idx][rank]);
	        (*in_begin).clear();	//# check why block+compress is faster than original sampling
	        (*in_begin).shrink_to_fit(); //# check why block+compress is faster than original sampling
	        total_rrr_size += s2;
	    }
	    blockR_bkp[block_idx][rank] = (unsigned int*)malloc((*n_ints)*n_rows*sizeof(unsigned int));
  		memcpy(blockR_bkp[block_idx][rank], blockR[block_idx][rank], (*n_ints)*n_rows*sizeof(unsigned int));
	}
	// std::cout<<" total=" <<total_rrr_size <<" localR.size="<<blockR[block_idx].size()<<std::endl;
}

template <typename vertex_type, typename RRRset>
void bitmapRRRSets3(std::vector<RRRset> &RRRsets, const int block_idx, const int blockoffset, 
				   std::vector<std::vector<unsigned int*>> &blockR, const size_t n_vtx, size_t* n_ints) {
  	size_t s1 = RRRsets.size(); //i: current RR's index
  	// std::cout<<"bitmapRRR3, s1="<<s1<<" block-offset="<<blockoffset << std::endl;
  	size_t total_rrr_size = 0;
  	size_t num_threads = omp_get_max_threads();
  	blockR[block_idx].resize(num_threads);
  	// std::cout<<" ****** here:" <<blockR[block_idx].size()<<","<<std::endl;
#pragma omp parallel num_threads(num_threads) proc_bind(spread) //proc_bind(close) //shared(n_vtx)
  	{
  		size_t rank = omp_get_thread_num();
  		size_t n_rows = n_vtx+1; // number of vtx +1, the last is reserved to accumulate bitmap of seeds
  		size_t n_cols = (s1 - blockoffset) / num_threads; // number of RRs
  		if ((s1 - blockoffset) % num_threads > 0) {
  			n_cols += 1;
  		}
  		*n_ints =  n_cols / 32;
  		if (n_cols % 32 > 0){
  			*n_ints += 1;
  		} 
  		blockR[block_idx][rank] = (unsigned int*)malloc((*n_ints)*n_rows*sizeof(unsigned int));
  		memset(blockR[block_idx][rank], 0, (*n_ints)*n_rows*sizeof(unsigned int));

#pragma omp for reduction(+:total_rrr_size) schedule(static)//(dynamic)//
	    for (size_t i=blockoffset; i<s1; i++) {
	        auto in_begin=RRRsets.begin();
	        std::advance(in_begin, i);
	        size_t s2=std::distance(in_begin->begin(),in_begin->end());
	        size_t local_idx = i % n_cols;
	        encodeRR23(in_begin, local_idx, s2, *n_ints, blockR[block_idx][rank]);
	        (*in_begin).clear();	//# check why block+compress is faster than original sampling
	        (*in_begin).shrink_to_fit(); //# check why block+compress is faster than original sampling
	        total_rrr_size += s2;
	    }
	}
	// std::cout<<" total=" <<total_rrr_size <<" localR.size="<<blockR[block_idx].size()<<std::endl;
}

} //// namespace ripples
#endif //RIPPLES_BITMAP_H
