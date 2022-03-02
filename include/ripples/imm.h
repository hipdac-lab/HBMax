//===------------------------------------------------------------*- C++ -*-===//
//
//             Ripples: A C++ Library for Influence Maximization
//                  Marco Minutoli <marco.minutoli@pnnl.gov>
//                   Pacific Northwest National Laboratory
//
//===----------------------------------------------------------------------===//
//
// Copyright (c) 2019, Battelle Memorial Institute
//
// Battelle Memorial Institute (hereinafter Battelle) hereby grants permission
// to any person or entity lawfully obtaining a copy of this software and
// associated documentation files (hereinafter “the Software”) to redistribute
// and use the Software in source and binary forms, with or without
// modification.  Such person or entity may use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and may permit
// others to do so, subject to the following conditions:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Other than as used herein, neither the name Battelle Memorial Institute or
//    Battelle may be used in any form whatsoever without the express written
//    consent of Battelle.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL BATTELLE OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//===----------------------------------------------------------------------===//

#ifndef RIPPLES_IMM_H
#define RIPPLES_IMM_H

#include <cmath>
#include <cstddef>
#include <limits>
#include <unordered_map>
#include <vector>
#include <numa.h>
#include <math.h>
#include "nlohmann/json.hpp"
#include "trng/lcg64.hpp"
#include "trng/uniform01_dist.hpp"
#include "trng/uniform_int_dist.hpp"

#include "ripples/configuration.h"
#include "ripples/find_most_influential.h"
#include "ripples/generate_rrr_sets.h"
#include "ripples/imm_execution_record.h"
#include "ripples/tim.h"
#include "ripples/utility.h"
#include "ripples/huffman.h"

#include "ripples/streaming_rrr_generator.h"
#include "ripples/streaming_rrr_decompress.h"

#define CUDA_PROFILE 0


namespace ripples {

//! The IMM algorithm configuration descriptor.
struct IMMConfiguration : public TIMConfiguration {
  size_t streaming_workers{0};
  size_t streaming_gpu_workers{0};
  size_t seed_select_max_workers{std::numeric_limits<size_t>::max()};
  size_t seed_select_max_gpu_workers{0};
  std::string gpu_mapping_string{""};
  std::unordered_map<size_t, size_t> worker_to_gpu;

  //! \brief Add command line options to configure IMM.
  //!
  //! \param app The command-line parser object.
  void addCmdOptions(CLI::App &app) {
    TIMConfiguration::addCmdOptions(app);
    app.add_option(
           "--streaming-gpu-workers", streaming_gpu_workers,
           "The number of GPU workers for the CPU+GPU streaming engine.")
        ->group("Streaming-Engine Options");
    app.add_option("--streaming-gpu-mapping", gpu_mapping_string,
                   "A comma-separated set of OpenMP numbers for GPU workers.")
        ->group("Streaming-Engine Options");
    app.add_option("--seed-select-max-workers", seed_select_max_workers,
                   "The max number of workers for seed selection.")
        ->group("Streaming-Engine Options");
    app.add_option("--seed-select-max-gpu-workers", seed_select_max_gpu_workers,
                   "The max number of GPU workers for seed selection.")
        ->group("Streaming-Engine Options");
  }
};

//! Retrieve the configuration parsed from command line.
//! \return the configuration parsed from command line.
ToolConfiguration<ripples::IMMConfiguration> configuration();

//! Approximate logarithm of n chose k.
//! \param n
//! \param k
//! \return an approximation of log(n choose k).
inline double logBinomial(size_t n, size_t k) {
  return n * log(n) - k * log(k) - (n - k) * log(n - k);
}

//! Compute ThetaPrime.
//!
//! \tparam execution_tag The execution policy
//!
//! \param x The index of the current iteration.
//! \param epsilonPrime Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param num_nodes The number of nodes in the input graph.
template <typename execution_tag>
ssize_t ThetaPrime(ssize_t x, double epsilonPrime, double l, size_t k,
                   size_t num_nodes, execution_tag &&) {
  return (2 + 2. / 3. * epsilonPrime) *
         (l * std::log(num_nodes) + logBinomial(num_nodes, k) +
          std::log(std::log2(num_nodes))) *
         std::pow(2.0, x) / (epsilonPrime * epsilonPrime);
}

//! Compute Theta.
//!
//! \param epsilon Parameter controlling the approximation factor.
//! \param l Parameter usually set to 1.
//! \param k The size of the seed set.
//! \param LB The estimate of the lower bound.
//! \param num_nodes The number of nodes in the input graph.
inline size_t Theta(double epsilon, double l, size_t k, double LB,
                    size_t num_nodes) {
  if (LB == 0) return 0;

  double term1 = 0.6321205588285577;  // 1 - 1/e
  double alpha = sqrt(l * std::log(num_nodes) + std::log(2));
  double beta = sqrt(term1 * (logBinomial(num_nodes, k) +
                              l * std::log(num_nodes) + std::log(2)));
  double lamdaStar = 2 * num_nodes * (term1 * alpha + beta) *
                     (term1 * alpha + beta) * pow(epsilon, -2);

  // std::cout << "#### " << lamdaStar << " / " << LB << " = " << lamdaStar / LB << std::endl;
  return lamdaStar / LB;
}

inline auto Entropy(std::vector<uint32_t> &globalcnt) {
  size_t N=globalcnt.size();
  size_t sum = 0;
  std::for_each(globalcnt.begin(), globalcnt.end(), [&] (uint32_t x) {
     sum += x;
  });
  float mean = sum/N;
  float s2 = 0;
  std::for_each(globalcnt.begin(), globalcnt.end(), [&] (uint32_t x) {
     s2 += pow((x-mean),2)/(N-1);
  });
  float s = sqrt(s2);
  float entropy = 0.0;  
  std::for_each(globalcnt.begin(), globalcnt.end(), [&] (uint32_t x) {
    float p_x = (float)x/sum;
    if(p_x>0){
      entropy -= p_x*log2(p_x);
    }
  });
  float skew = 0.0;
  std::for_each(globalcnt.begin(), globalcnt.end(), [&] (uint32_t x) {
     skew += pow((x-mean),3)/(N*pow(s,3));
  });
  float kurt = 0.0;
  std::for_each(globalcnt.begin(), globalcnt.end(), [&] (uint32_t x) {
     kurt += pow((x-mean),4)/(N*pow(s,4));
  });
  kurt-=3; //
  return std::make_tuple(entropy,skew,kurt);
}
//! Collect a set of Random Reverse Reachable set.
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam RRRGeneratorTy The type of the RRR generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param k The size of the seed set.
//! \param epsilon The parameter controlling the approximation guarantee.
//! \param l Parameter usually set to 1.
//! \param generator The rrr sets generator.
//! \param record Data structure storing timing and event counts.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename diff_model_tag, typename execution_tag, typename vertex_type>
auto Sampling5(HuffmanTree* huffmanTree, std::vector<uint32_t> &globalcnt,
              std::vector<unsigned char*> &compR, std::vector<uint32_t> &compBytes, std::vector<uint32_t> &codeCnt,
              std::vector<vertex_type *> &copyR, std::vector<uint32_t> &copyCnt, vertex_type *maxvtx,    
              const GraphTy &G, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  // using vertex_type = typename GraphTy::vertex_type;
  using ex_time_ms = std::chrono::duration<double, std::milli>;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;
  int blocks=CFG.q;
  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  #ifdef ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator(libmemkind::kinds::DAX_KMEM_PREFERRED);
  #else
  RRRsetAllocator<vertex_type> allocator;
  #endif
  std::vector<RRRset<GraphTy>> RR;

  auto xc1 = spdlog::stdout_color_st("xc1:");
  xc1->info("$$$ sampling 5, sortFlag={:s}",CFG.sortFlag);

  double f;
  std::vector<vertex_type> seeds;

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;

  int create_flag = 1, extra_flag=0;
  std::vector<bool> deleteflag;
  vertex_type tmpmax=0, nxtmax=0;
  size_t uncovered=0, freq=0;
  size_t delta_block_sum = 0;
  ex_time_ms elapse;
  double total_sampling=0, total_encode=0, total_decode, total_tree=0;
  float final_cover = 0.0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));
    if(thetaPrime%CFG.q>0){
      thetaPrime+=thetaPrime%CFG.q;
    }
    compR.resize(thetaPrime);
    compBytes.resize(thetaPrime);
    codeCnt.resize(thetaPrime);
    if(CFG.lossyFlag=="N"){
      copyR.resize(thetaPrime);
    }
    copyCnt.resize(thetaPrime);
    size_t delta = thetaPrime - RR.size();
    record.ThetaPrimeDeltas.push_back(delta);

    int delta_block;
    double vm1,vm2;

    for(int i=0;i<blocks;i++){
      delta_block = delta/blocks; 
      // process_mem_usage(vm1);
      auto t0 = std::chrono::high_resolution_clock::now();
      auto timeRRRSets = measure<>::exec_time([&]() {
        RR.insert(RR.end(), delta_block, RRRset<GraphTy>(allocator));

        auto begin = RR.end() - delta_block;

        GenerateRRRSets2(G, generator, begin, RR.end(), record,
                        std::forward<diff_model_tag>(model_tag),
                        std::forward<execution_tag>(ex_tag),
                        globalcnt, maxvtx,
                        CFG.sortFlag, extra_flag, CFG.rthd);
      });
      record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);
      auto t1 = std::chrono::high_resolution_clock::now();
      elapse=t1-t0;
      std::cout<<" gen-block.time=("<<elapse.count()<<")ms";

      if (create_flag==1){
        create_flag = 0;
        auto t3_1 = std::chrono::high_resolution_clock::now();
        auto stats = Entropy(globalcnt);
        auto t3_2 = std::chrono::high_resolution_clock::now();
        elapse=t3_2-t3_1;
        std::cout<<" block-entropy="<<std::get<0>(stats)<<", skewness="<<std::get<1>(stats)<<", kurtosis="<<std::get<2>(stats)<<"."<<std::endl;
        std::cout<<" warm-up-profiling= "<<elapse.count()<<" ms"<<std::endl;
        if(std::get<2>(stats)>6){ //kurtosis >6
          extra_flag = 1;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        process_mem_usage(vm1);
        initByRRRSets3<vertex_type>(huffmanTree, RR);
        process_mem_usage(vm2);
        auto t3 = std::chrono::high_resolution_clock::now();
        elapse=t3-t2;
        std::cout<<" init-huffman.time=("<<elapse.count()<<")ms"<<" mem="<<vm2-vm1<<std::endl;
        printf("===== codelength ======\n");
        size_t maxc=0,minc=0, maxlen=0,minlen=1000;
        for(size_t ii=0;ii<huffmanTree->stateNum;ii++){
          if (huffmanTree->cout[ii]>0 ){
            if (huffmanTree->cout[ii] >= maxlen){
              maxlen=huffmanTree->cout[ii];
              maxc=ii;
            }
            if (huffmanTree->cout[ii] <= minlen){
              minlen=huffmanTree->cout[ii];
              minc=ii;
            }
          }
        }
        std::cout<<"maxlen="<<maxlen<<", maxc="<<maxc<<" minlen="<<minlen<<", minc="<<minc<<std::endl;
      }    
      auto t4 = std::chrono::high_resolution_clock::now();
      encodeRRRSets3<vertex_type>(huffmanTree, RR, delta_block_sum, compR, compBytes, codeCnt, copyR, copyCnt, globalcnt, maxvtx, CFG.lossyFlag);
      auto t5 = std::chrono::high_resolution_clock::now();
      elapse=t5-t4;
      std::cout<<" compress-block.time=("<<elapse.count()<<")ms"<<std::endl;

      delta_block_sum += delta_block;
    }
    
    auto timeMostInfluential = measure<>::exec_time([&]() {  
      deleteflag.resize(compR.size());
      for(int i=0;i<compR.size();i++){
        deleteflag[i]=0;
      }
      uncovered=compR.size();
      tmpmax = *maxvtx;
      auto t6 = std::chrono::high_resolution_clock::now();
      while(seeds.size() < CFG.k && uncovered != 0){
        seeds.push_back(tmpmax);  
        auto t6_1 = std::chrono::high_resolution_clock::now();  
        nxtmax = DecompAndFind4<vertex_type>(huffmanTree, G.num_nodes(),
                compR, codeCnt, copyR, copyCnt, deleteflag, 
                compR.size(), tmpmax, &freq,
                record, std::forward<omp_parallel_tag>(ex_tag),CFG.lossyFlag, 0);
        auto t6_2 = std::chrono::high_resolution_clock::now();
        elapse=t6_2-t6_1;
        std::cout<<" decomp:tmpmax("<<tmpmax<<") freq="<<freq<<" using="<<elapse.count()<<"ms"<<std::endl;
        uncovered-=freq;
        tmpmax=nxtmax;
        f = double(compR.size() - uncovered) / compR.size();
      }
      auto t7 = std::chrono::high_resolution_clock::now();
      elapse=t7-t6;
      total_decode+=elapse.count();
      std::cout<<" decomp-and-find.time=("<<elapse.count()<<")ms"<<std::endl;
    });  
    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    xc1->info("$$$ thetaprime={:d} delta={:d} RR.size={:d} compR.size={:d} f={:f} seeds.size={:d}",thetaPrime, delta, RR.size(), compR.size(), f, seeds.size());
    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }
  
  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;
  spdlog::get("console")->info("Theta {}", theta);
  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > delta_block_sum) {
      if(theta%blocks>0){
        theta+=theta%blocks;
      }
      // extra_flag = 1;
      compR.resize(theta);
      compBytes.resize(theta);
      codeCnt.resize(theta);
      if(CFG.lossyFlag=="N"){
        copyR.resize(theta);
      }
      copyCnt.resize(theta);
      size_t final_delta = theta - delta_block_sum;
      int delta_block;
      for(int i=0;i<blocks;i++){
        delta_block = final_delta/blocks;  
      
        RR.insert(RR.end(), delta_block, RRRset<GraphTy>(allocator));

        auto begin = RR.end() - delta_block;
        auto t10 = std::chrono::high_resolution_clock::now();
        GenerateRRRSets2(G, generator, begin, RR.end(), record,
                        std::forward<diff_model_tag>(model_tag),
                        std::forward<execution_tag>(ex_tag),
                        globalcnt, maxvtx,
                        CFG.sortFlag, extra_flag, CFG.rthd);
        auto t11 = std::chrono::high_resolution_clock::now();
        elapse=t11-t10;
        std::cout<<" extra-gen-block.time=("<<elapse.count()<<")ms";
        encodeRRRSets3<vertex_type>(huffmanTree, RR, delta_block_sum, compR, compBytes, codeCnt, copyR, copyCnt, globalcnt, maxvtx, CFG.lossyFlag);
        auto t12 = std::chrono::high_resolution_clock::now();
        elapse=t12-t11;
        std::cout<<" extra-compress-block.time=("<<elapse.count()<<")ms"<<std::endl;
        delta_block_sum += delta_block;
      }
      seeds.clear();
      deleteflag.clear();

      deleteflag.resize(theta);
      for(int i=0;i<theta;i++){
        deleteflag[i]=0;
      }
      uncovered=theta;
      tmpmax = *maxvtx;
      auto t8 = std::chrono::high_resolution_clock::now();

      while(seeds.size() < CFG.k && uncovered != 0){
        seeds.push_back(tmpmax);  
        auto t8_1 = std::chrono::high_resolution_clock::now();
        nxtmax = DecompAndFind4<vertex_type>(huffmanTree, G.num_nodes(),
                compR, codeCnt, copyR, copyCnt, deleteflag, 
                theta, tmpmax, &freq,
                record, std::forward<omp_parallel_tag>(ex_tag), CFG.lossyFlag, 1);
        
        auto t8_2 = std::chrono::high_resolution_clock::now();
        elapse=t8_2-t8_1;
        std::cout<<" extra-decomp:tmpmax("<<tmpmax<<") freq="<<freq<<" using="<<elapse.count()<<"ms"<<std::endl;
        uncovered-=freq;
        tmpmax=nxtmax;
        f = double(theta - uncovered) / theta;
      }
      for(int i=0; i<theta; i++){
        if(deleteflag[i]==0){
            if(codeCnt[i]>0){
              free(compR[i]);
            }
            if(CFG.lossyFlag=="N"){
              if(copyCnt[i]>0){
                free(copyR[i]);
              }
            }
        }
      }
      auto t9 = std::chrono::high_resolution_clock::now();
      elapse=t9-t8;
      total_decode+=elapse.count();
      std::cout<<" final-decomp-and-find.time=("<<elapse.count()<<")ms"<<" f="<<f<<std::endl;
    } //theta > thetaprime
  });

  compR.shrink_to_fit();
  compBytes.shrink_to_fit();
  codeCnt.shrink_to_fit();
  copyR.shrink_to_fit();
  copyCnt.shrink_to_fit();
  globalcnt.shrink_to_fit();
  std::cout<<" *** generate: rr.size:"<<RR.size()<<" compr.size="<<compR.size()<<" theta:"<<theta<<std::endl;
  std::cout<<", total-encode="<<total_encode<<" total_decode="<<total_decode<<std::endl;
  return seeds;
}


template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename diff_model_tag, typename execution_tag>
auto Sampling(const GraphTy &G, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, execution_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  #ifdef ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator(libmemkind::kinds::DAX_KMEM_PREFERRED);
  #else
  RRRsetAllocator<vertex_type> allocator;
  #endif
  std::vector<RRRset<GraphTy>> RR;

  auto xc1 = spdlog::stdout_color_st("xc1:");
  xc1->info("$$$ sampling 1, sortFlag={:s}",CFG.sortFlag);

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<execution_tag>(ex_tag));

    size_t delta = thetaPrime - RR.size();
    record.ThetaPrimeDeltas.push_back(delta);

    auto timeRRRSets = measure<>::exec_time([&]() {
      RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag),CFG.sortFlag);
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    double f;

    auto timeMostInfluential = measure<>::exec_time([&]() {
      const auto &S =
          FindMostInfluentialSet(G, CFG, RR, record, generator.isGpuEnabled(),
                                 std::forward<execution_tag>(ex_tag));

      f = S.first;
    });

    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    xc1->info("$$$ thetaprime={:d} delta={:d} RR.size={:7d} f={:f}",thetaPrime, delta, RR.size(), f);
    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }
  
  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;
  spdlog::get("console")->info("Theta {}", theta);

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > RR.size()) {
      size_t final_delta = theta - RR.size();
      RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - final_delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<execution_tag>(ex_tag),CFG.sortFlag);
    }
  });
  return RR;
}

template <typename GraphTy, typename ConfTy, typename RRRGeneratorTy,
          typename diff_model_tag>
auto Sampling(const GraphTy &G, const ConfTy &CFG, double l,
              RRRGeneratorTy &generator, IMMExecutionRecord &record,
              diff_model_tag &&model_tag, sequential_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  // sqrt(2) * epsilon
  double epsilonPrime = 1.4142135623730951 * epsilon;

  double LB = 0;
  #ifdef ENABLE_MEMKIND
  RRRsetAllocator<vertex_type> allocator(libmemkind::kinds::DAX_KMEM_PREFERRED);
  #else
  RRRsetAllocator<vertex_type> allocator;
  #endif
  std::vector<RRRset<GraphTy>> RR;

  auto xc2 = spdlog::stdout_color_st("xc2:");
  xc2->info("$$$ sampling 2");

  auto start = std::chrono::high_resolution_clock::now();
  size_t thetaPrime = 0;
  for (ssize_t x = 1; x < std::log2(G.num_nodes()); ++x) {
    // Equation 9
    ssize_t thetaPrime = ThetaPrime(x, epsilonPrime, l, k, G.num_nodes(),
                                    std::forward<sequential_tag>(ex_tag));
    xc2->info("$$$ theta-prime={:5d}, rr-size={:5d}",thetaPrime, RR.size());
    size_t delta = thetaPrime - RR.size();
    record.ThetaPrimeDeltas.push_back(delta);

    auto timeRRRSets = measure<>::exec_time([&]() {
      RR.insert(RR.end(), delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<sequential_tag>(ex_tag),CFG.sortFlag);
    });
    record.ThetaEstimationGenerateRRR.push_back(timeRRRSets);

    double f;

    auto timeMostInfluential = measure<>::exec_time([&]() {
      const auto &S = FindMostInfluentialSet(
          G, CFG, RR, record, false, std::forward<sequential_tag>(ex_tag));

      f = S.first;
    });

    record.ThetaEstimationMostInfluential.push_back(timeMostInfluential);

    if (f >= std::pow(2, -x)) {
      LB = (G.num_nodes() * f) / (1 + epsilonPrime);
      break;
    }
  }

  size_t theta = Theta(epsilon, l, k, LB, G.num_nodes());
  auto end = std::chrono::high_resolution_clock::now();

  record.ThetaEstimationTotal = end - start;

  record.Theta = theta;

  record.GenerateRRRSets = measure<>::exec_time([&]() {
    if (theta > RR.size()) {
      size_t final_delta = theta - RR.size();
      RR.insert(RR.end(), final_delta, RRRset<GraphTy>(allocator));

      auto begin = RR.end() - final_delta;

      GenerateRRRSets(G, generator, begin, RR.end(), record,
                      std::forward<diff_model_tag>(model_tag),
                      std::forward<sequential_tag>(ex_tag),CFG.sortFlag);
    }
  });

  return RR;
}

//! The IMM algroithm for Influence Maximization
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ConfTy The configuration type.
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param CFG The configuration.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.
template <typename GraphTy, typename ConfTy, typename PRNG,
          typename diff_model_tag>
auto IMM(const GraphTy &G, const ConfTy &CFG, double l, PRNG &gen,
         IMMExecutionRecord &record, diff_model_tag &&model_tag,
         sequential_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;

  std::vector<trng::lcg64> generator(1, gen);

  l = l * (1 + 1 / std::log2(G.num_nodes()));

  auto R = Sampling(G, CFG, l, generator, record,
                    std::forward<diff_model_tag>(model_tag),
                    std::forward<sequential_tag>(ex_tag));

#if CUDA_PROFILE
  auto logst = spdlog::stdout_color_st("IMM-profile");
  std::vector<size_t> rrr_sizes;
  for (auto &rrr_set : R) rrr_sizes.push_back(rrr_set.size());
  print_profile_counter(logst, rrr_sizes, "RRR sizes");
#endif
  
  auto start = std::chrono::high_resolution_clock::now();
  const auto &S = FindMostInfluentialSet(G, CFG, R, record, false,
                                         std::forward<sequential_tag>(ex_tag));
  auto end = std::chrono::high_resolution_clock::now();

  record.FindMostInfluentialSet = end - start;

  auto xc3 = spdlog::stdout_color_st("xc3:");
  xc3->info("$$$ IMM-1 f={:f}", S.first);
  // const auto &S1=DumpRRRSets(G,R);
  // DumpRRRSets(G,R);

  return S.second;
}

//! The IMM algroithm for Influence Maximization
//!
//! \tparam GraphTy The type of the input graph.
//! \tparam ConfTy The configuration type
//! \tparam PRNG The type of the parallel random number generator.
//! \tparam diff_model_tag Type-Tag to selecte the diffusion model.
//! \tparam execution_tag Type-Tag to select the execution policy.
//!
//! \param G The input graph.  The graph is transoposed.
//! \param CFG The configuration.
//! \param l Parameter usually set to 1.
//! \param gen The parallel random number generator.
//! \param model_tag The diffusion model tag.
//! \param ex_tag The execution policy tag.

template <typename GraphTy, typename ConfTy, typename GeneratorTy,
          typename diff_model_tag>
auto IMM3(const GraphTy &G, const ConfTy &CFG, double l, GeneratorTy &gen,
         diff_model_tag &&model_tag, omp_parallel_tag &&ex_tag) {
  using vertex_type = typename GraphTy::vertex_type;
  size_t k = CFG.k;
  double epsilon = CFG.epsilon;
  auto &record(gen.execution_record());
  std::vector<vertex_type> seeds;
  l = l * (1 + 1 / std::log2(G.num_nodes()));
  double vm1;
  size_t num_threads = omp_get_max_threads();
  int rank = omp_get_thread_num();
  int numa_node = numa_node_of_cpu(rank);
  int task_nodes = numa_num_task_nodes();
  HuffmanTree* huffmanTree = createHuffmanTree(G.num_nodes());//
  process_mem_usage(vm1);
  std::cout << "##imm3-huffman-tree: " << vm1 << " Mb on cpu("<< rank <<"/"<<num_threads<<")";
  std::cout << "task-nodes="<< task_nodes << std::endl;
  std::vector<unsigned char*> compR;
  std::vector<uint32_t> compBytes;
  std::vector<uint32_t> codeCnt;
  std::vector<vertex_type*> copyR;
  std::vector<uint32_t> copyCnt;
  vertex_type maxvtx=0;
  std::vector<uint32_t> globalcnt;
  globalcnt.resize(G.num_nodes());
  for(int i=0;i<G.num_nodes();i++){
    globalcnt[i]=0;
  }
  process_mem_usage(vm1);
  std::cout << "##imm3-work-vars2: " << vm1 << " Mb" <<std::endl;
  seeds =
      Sampling5(huffmanTree, globalcnt,
              compR, compBytes, codeCnt, copyR, copyCnt, &maxvtx, 
              G, CFG, l, gen, record, 
              std::forward<diff_model_tag>(model_tag),
              std::forward<omp_parallel_tag>(ex_tag));
  process_mem_usage(vm1);
  std::cout << "##imm3-sampling: " << vm1 << " Mb" <<std::endl;    
  SZ_ReleaseHuffman(huffmanTree);
  auto xc4 = spdlog::stdout_color_st("xc4:");
  xc4->info("$$$ IMM3");
  // DumpRRRSets(G,R);
  // DumpCompRRRSets(compR,compBytes,R);
  return seeds;
}

}  // namespace ripples

#endif  // RIPPLES_IMM_H
