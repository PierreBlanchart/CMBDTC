/** CF_finder.cpp
 *
 * Copyright (C) 2021 Pierre BLANCHART
 * pierre.blanchart@cea.fr
 * CEA/LIST/DM2I/SID/LI3A
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 **/


#include "CF_finder.h"


CF_finder::CF_finder(tree_model *mytree_, float *pt_query_, int target_class_, float d2target_sup_, int budget_bt, 
                     const bool check_has_target_, const bool update_sup_bound_) {
  printf("Building CF finder object : target_class = %d ...\n", target_class_);
  mytree = mytree_;
  D = mytree->D;
  pt_query = pt_query_;
  target_class = target_class_;
  d2target_sup = d2target_sup_;
  
  check_has_target = check_has_target_;
  update_sup_bound = update_sup_bound_;
  
  if (d2target_sup > 0.f) {
    subtree = mytree->select_boxes(pt_query, d2target_sup);
    using_subtree = true;
  } else {
    subtree = mytree;
  }
  printf("Number of leaves in subtree: %d [%d]\n", subtree->Nleaves, mytree->Nleaves);
  
  subtree->presort();
  printf("Done presorting ...\n");
  
  if (budget_bt > 0) {
    
  }
  
  nb_threads = tbb::task_scheduler_init::default_num_threads();
  tbb::task_scheduler_init init(nb_threads);
  alloc_mt.resize(nb_threads);
  for (int t=0; t < nb_threads; t++) alloc_mt[t] = new mem_allocator(subtree->Nleaves, mytree->ttype, mytree->Nclass);
  printf("Allocated for %d threads\n", nb_threads);
  
  target = new float[D];
  target_mt.resize(nb_threads);
  for (int t=0; t < nb_threads; t++) {
    target_mt[t] = new float[D];
    std::fill(target_mt[t], target_mt[t]+D, FLT_MAX);
  }
  
  // CF_intervals = makeTensor<float>(nb_threads, 2, D);
  CF_intervals.resize(nb_threads);
  CF_in_intervals.resize(nb_threads);
  d2CF.resize(nb_threads);
}



// chunk_sz = maximum number of pb per thread
void CF_finder::start_width_first(const int max_dim, const int chunk_sz_start) {
  
  int chunk_sz = chunk_sz_start;
  pb_per_dims_mt.resize(D); // D x Nchunks x chunk_sz problems
  
  // level 0
  boost::dynamic_bitset<> is_indexed_leaf(subtree->Nleaves);
  is_indexed_leaf.set();
  
  pb_per_dims_mt[0].resize(1);
  pb_per_dims_mt[0][0].resize(1);
  pb_per_dims_mt[0][0][0] = new interv1d_problem_fast(subtree, subtree->Nleaves, 0, is_indexed_leaf);
  
  Ntot_cur = 1;
  vector<unsigned int> N2chunk(Ntot_cur);
  vector<unsigned int> N2idInChunk(Ntot_cur);
  
  ind_dim_cur = 1;
  while (ind_dim_cur < std::min(max_dim, D) && Ntot_cur < max_per_worker) {
    
    int nb_chunks = Ntot_cur/chunk_sz; if (nb_chunks*chunk_sz < Ntot_cur) nb_chunks++;
    if (nb_chunks > nb_threads*NchunksPerThread) { // make bigger chunks so that the number of chunks doesn't exceed "NchunksPerThread" times the number of threads
      nb_chunks = nb_threads*NchunksPerThread;
      chunk_sz = (int) std::ceil(float(Ntot_cur)/float(nb_chunks));
    }
    // printf("Level %d: using %d chunks\n", ind_dim_cur-1, nb_chunks);
    
    pb_per_dims_mt[ind_dim_cur].resize(nb_chunks);
    temp_Ntot.resize(nb_chunks);
    
    tbb::parallel_for(
      blocked_range<int>(0, nb_chunks),
      [&](tbb::blocked_range<int> r) {
        int thread_id = tbb::this_task_arena::current_thread_index();
        // printf("Launching thread %d\n", thread_id);
        for (int c=r.begin(); c < r.end(); ++c) {
          int ind_start = c*chunk_sz;
          int ind_end = std::min(ind_start+chunk_sz, Ntot_cur);
          int Ntot_cur_c = 0;
          // printf("Thread %d: [%d --> %d]\n", thread_id, ind_start, ind_end);
          for (int n=ind_start; n < ind_end; n++) {
            
            // printf("Treating task %d\n", n);
            interv1d_problem_fast *pb_n = pb_per_dims_mt[ind_dim_cur-1][N2chunk[n]][N2idInChunk[n]];
            pb_n->solve_with_inf(alloc_mt[thread_id]);
            pb_n->d2regions(alloc_mt[thread_id], pt_query[ind_dim_cur-1], true, false);
            
            pb_per_dims_mt[ind_dim_cur][c].resize(Ntot_cur_c+pb_n->nb_seg);
            for (int l=0; l < pb_n->nb_seg; l++) { // adds sub-problems which are inside the sup bound
              // printf("%d : %f\n", Ntot_cur_c, pb_n->dquery2regions[l]);
              if (alloc_mt[thread_id]->dquery2regions[l] <= d2target_sup && pb_n->has_target(alloc_mt[thread_id], target_class, l)) { // pb_n->has_target_binary(alloc_mt[thread_id], target_class, l)) {
                pb_per_dims_mt[ind_dim_cur][c][Ntot_cur_c] = pb_n->create_next_problem(alloc_mt[thread_id], l);
                
                // debug: check partial CF
                // to do
                
                Ntot_cur_c++;
              }
            }
            
            delete pb_n;
            pb_per_dims_mt[ind_dim_cur-1][N2chunk[n]][N2idInChunk[n]] = NULL;
            
          }
          
          temp_Ntot[c] = Ntot_cur_c;
        }
        
      }
      
    );
    
    Ntot_cur = temp_Ntot[0];
    for (int c=1; c < nb_chunks; c++) Ntot_cur += temp_Ntot[c];
    
    N2chunk.resize(Ntot_cur);
    N2idInChunk.resize(Ntot_cur);
    int accum = 0, accum_next;
    for (int c=0; c < nb_chunks; c++) {
      accum_next = accum+temp_Ntot[c];
      std::fill(N2chunk.begin()+accum, N2chunk.begin()+accum_next, (unsigned int) c);
      std::iota(N2idInChunk.begin()+accum, N2idInChunk.begin()+accum_next, 0);
      accum = accum_next;
    }
    
    printf("Created %d problems at level %d\n", Ntot_cur, ind_dim_cur-1);
    ind_dim_cur++;
    
  }
  
  N2chunk.clear();
  N2idInChunk.clear();
  
  recorded_dim_start = ind_dim_cur-1;
  
}



// chunk_sz = maximum number of pb per thread
void CF_finder::start_width_first_regression(const int max_dim, const int chunk_sz_start) {
  
  // printf("Entering width first\n");
  int chunk_sz = chunk_sz_start;
  pb_per_dims_mt.resize(D); // D x Nchunks x chunk_sz problems
  
  // level 0
  boost::dynamic_bitset<> is_indexed_leaf(subtree->Nleaves);
  is_indexed_leaf.set();
  
  pb_per_dims_mt[0].resize(1);
  pb_per_dims_mt[0][0].resize(1);
  pb_per_dims_mt[0][0][0] = new interv1d_problem_fast(subtree, subtree->Nleaves, 0, is_indexed_leaf);
  
  Ntot_cur = 1;
  vector<unsigned int> N2chunk(Ntot_cur);
  vector<unsigned int> N2idInChunk(Ntot_cur);
  
  ind_dim_cur = 1;
  while (ind_dim_cur < std::min(max_dim, D) && Ntot_cur < max_per_worker) {
    
    int nb_chunks = Ntot_cur/chunk_sz; if (nb_chunks*chunk_sz < Ntot_cur) nb_chunks++;
    if (nb_chunks > nb_threads*NchunksPerThread) { // make bigger chunks so that the number of chunks doesn't exceed "NchunksPerThread" times the number of threads
      nb_chunks = nb_threads*NchunksPerThread;
      chunk_sz = (int) std::ceil(float(Ntot_cur)/float(nb_chunks));
    }
    // printf("Level %d: using %d chunks\n", ind_dim_cur-1, nb_chunks);
    
    pb_per_dims_mt[ind_dim_cur].resize(nb_chunks);
    temp_Ntot.resize(nb_chunks);
    
    tbb::parallel_for(
      blocked_range<int>(0, nb_chunks),
      [&](tbb::blocked_range<int> r) {
        int thread_id = tbb::this_task_arena::current_thread_index();
        // printf("Launching thread %d\n", thread_id);
        for (int c=r.begin(); c < r.end(); ++c) {
          int ind_start = c*chunk_sz;
          int ind_end = std::min(ind_start+chunk_sz, Ntot_cur);
          int Ntot_cur_c = 0;
          // printf("Thread %d: [%d --> %d]\n", thread_id, ind_start, ind_end);
          for (int n=ind_start; n < ind_end; n++) {
            
            // printf("Treating task %d\n", n);
            interv1d_problem_fast *pb_n = pb_per_dims_mt[ind_dim_cur-1][N2chunk[n]][N2idInChunk[n]];
            pb_n->solve_with_inf(alloc_mt[thread_id]);
            pb_n->d2regions(alloc_mt[thread_id], pt_query[ind_dim_cur-1], true, false);
            
            pb_per_dims_mt[ind_dim_cur][c].resize(Ntot_cur_c+pb_n->nb_seg);
            for (int l=0; l < pb_n->nb_seg; l++) { // adds sub-problems which are inside the sup bound
              // printf("%d : %f\n", Ntot_cur_c, pb_n->dquery2regions[l]);
              if (alloc_mt[thread_id]->dquery2regions[l] <= d2target_sup && pb_n->has_target_regression(alloc_mt[thread_id], l)) {
                pb_per_dims_mt[ind_dim_cur][c][Ntot_cur_c] = pb_n->create_next_problem(alloc_mt[thread_id], l);
                Ntot_cur_c++;
              }
            }
            
            delete pb_n;
            pb_per_dims_mt[ind_dim_cur-1][N2chunk[n]][N2idInChunk[n]] = NULL;
            
          }
          
          temp_Ntot[c] = Ntot_cur_c;
        }
        
      }
      
    );
    
    Ntot_cur = temp_Ntot[0];
    for (int c=1; c < nb_chunks; c++) Ntot_cur += temp_Ntot[c];
    
    N2chunk.resize(Ntot_cur);
    N2idInChunk.resize(Ntot_cur);
    int accum = 0, accum_next;
    for (int c=0; c < nb_chunks; c++) {
      accum_next = accum+temp_Ntot[c];
      std::fill(N2chunk.begin()+accum, N2chunk.begin()+accum_next, (unsigned int) c);
      std::iota(N2idInChunk.begin()+accum, N2idInChunk.begin()+accum_next, 0);
      accum = accum_next;
    }
    
    printf("Created %d problems at level %d\n", Ntot_cur, ind_dim_cur-1);
    ind_dim_cur++;
    
  }
  
  N2chunk.clear();
  N2idInChunk.clear();
  
  recorded_dim_start = ind_dim_cur-1;
  
}



// version avec un alloc mt per dimension pour éviter les problèmes de corruption dus au parcours depth first
void CF_finder::continue_depth_first(const int budget) {
  
  int nb_chunks = temp_Ntot.size();
  int dim_start = ind_dim_cur-1; // pb_per_dims_mt[ind_dim_cur-1][x][0]->ind_dim;
  
  alloc_mt_df.resize(nb_threads);
  for (int t=0; t < nb_threads; t++) {
    alloc_mt_df[t].resize(D);
    for (int d=dim_start; d < D; d++) {
      alloc_mt_df[t][d] = new mem_allocator(subtree->Nleaves, mytree->ttype, mytree->Nclass);
    }
  }
  
  tbb::atomic<float> d2target_sup_thread = d2target_sup;
  tbb::atomic<int> nb_approx=0;
  
  printf("Starting depth first in dimension %d using %d pb chunks\n", dim_start, nb_chunks);
  
  // tbb::concurrent_vector<bool> temp_found_solution(nb_threads, false);
  vector<bool> temp_found_solution(nb_threads, false);
  
  tbb::parallel_for(
    blocked_range<int>(0, nb_chunks),
    [&](tbb::blocked_range<int> r) {
      int thread_id = tbb::this_task_arena::current_thread_index(); // b
      
      vector<interv1d_problem_fast *> explore_path(D, NULL);
      vector<int *> explore_order(D, NULL);
      unsigned int *cursor = new unsigned int[D];
      vector<int> nb2explore(D);
      
      for (int c=r.begin(); c < r.end(); ++c) {
        
        float temp_d2target_sup_c = -1.f;
        
        // gather all distances to formed regions
        float *temp_d2region = new float[temp_Ntot[c]];
        int max_Ninterv = -1;
        for (int r=0; r < temp_Ntot[c]; r++) {
          temp_d2region[r] = pb_per_dims_mt[dim_start][c][r]->dprev;
          max_Ninterv = std::max(max_Ninterv, (int)pb_per_dims_mt[dim_start][c][r]->Ninterv);
        }
        // order by distance to intersection regions
        int *order_pb = new int[temp_Ntot[c]]; std::iota(order_pb, order_pb+temp_Ntot[c], 0);
        std::sort(order_pb, order_pb+temp_Ntot[c], [&](int i1, int i2) { return temp_d2region[i1] < temp_d2region[i2]; } );
        delete[] temp_d2region;
        
        // pre-allocate explore_order in all dimensions
        for (int d=dim_start+1; d < D; d++) explore_order[d] = new int[2*max_Ninterv];
        
        // we can only explore problems that were created and not solved yet
        explore_order[dim_start] = order_pb;
        nb2explore[dim_start] = temp_Ntot[c];
        // cursor[dim_start] = 0; // using memset later
        // explore_path[dim_start] = pb_per_dims_mt[dim_start][c][order_pb[0]];
        
        int cnt=0;
        
        int dim_cur = dim_start;
        interv1d_problem_fast *pb_cur;
        memset(cursor, 0, sizeof(unsigned int)*D);
        while(cnt < budget) {
          cnt++;
          
          if (cursor[dim_cur] == nb2explore[dim_cur]) { // backtracking
            if (dim_cur == dim_start) {
              // printf("[%d] : No target region in class %d\n", p, target_class);
              break;
            }
            dim_cur--;
            continue;
          }
          
          // load and solve current problem
          int ind_next = explore_order[dim_cur][cursor[dim_cur]];
          if (dim_cur==dim_start) {
            pb_cur = pb_per_dims_mt[dim_start][c][ind_next];
            pb_per_dims_mt[dim_start][c][ind_next] = NULL; // added this to prevent deletion in destructor (will be deleted later in this function)
          } else {
            pb_cur = explore_path[dim_cur-1]->create_next_problem(alloc_mt_df[thread_id][dim_cur-1], ind_next);
          }
          if (pb_cur->dprev < d2target_sup_thread) {
            if (explore_path[dim_cur]) {
              delete explore_path[dim_cur];
              explore_path[dim_cur] = NULL;
            }
            explore_path[dim_cur] = pb_cur;
          } else {
            if (dim_cur == dim_start) break;
            dim_cur--;
            continue;
          }
          
          cursor[dim_cur]++; // move cursor to the next problem
          // pb_cur->printInfo();
          // if (dim_cur==dim_start+1) return;
          // printf("Current thread = %d [%d]\n", thread_id, nb_threads);
          pb_cur->solve_with_inf(alloc_mt_df[thread_id][dim_cur]); // solve pb in dimension dim_cur+1
          
          if (dim_cur == D-1) { // search for potential target region
            pb_cur->do_scoring(alloc_mt_df[thread_id][dim_cur]); // pb_cur->do_scoring_binary(alloc_mt_df[thread_id][dim_cur]);
            if (pb_cur->nb_per_class[target_class]) { // we found a target region
              pb_cur->d2regions(alloc_mt_df[thread_id][dim_cur], pt_query[dim_cur], true);
              int ind_target_seg;
              temp_d2target_sup_c = pb_cur->findClosestTarget(alloc_mt_df[thread_id][dim_cur], target_class, ind_target_seg); // pb_cur->findClosestTarget_classif(alloc_mt_df[thread_id][dim_cur], target_class, ind_target_seg);
              if (temp_d2target_sup_c <= d2target_sup_thread) {
                if (mytree->ttype != multi_classif) {
                  printf("[%d] : Found target region from class %d (%f) at iteration %d | d2target = %f [%d]!\n", thread_id, target_class, alloc_mt_df[thread_id][dim_cur]->scores_binary[ind_target_seg], 
                         cnt, sqrt(temp_d2target_sup_c), ind_target_seg);
                } else {
                  printf("[%d] : Found target region from class %d at iteration %d | d2target = %f [%d]!\n", thread_id, target_class, 
                         cnt, sqrt(temp_d2target_sup_c), ind_target_seg);
                  // for (int ind_c=0; ind_c < mytree->Nclass; ind_c++) {printf("%f | ", alloc_mt_df[thread_id][dim_cur]->scores_multi[ind_c][ind_target_seg]);}; printf("\n");
                }
                for (int d=0; d < D-1; d++) target_mt[thread_id][d] = pb_cur->ptc_from_root[d];
                target_mt[thread_id][D-1] = alloc_mt_df[thread_id][dim_cur]->ptc2seg[ind_target_seg]; // pb_cur->ptc2seg[ind_target_seg];
                
                float **CF_region = makeMatrix<float>(2, D);
                memcpy(CF_region[0], pb_cur->interval_start.data(), sizeof(float)*(D-1)); CF_region[0][D-1] = alloc_mt_df[thread_id][dim_cur]->segments[ind_target_seg];
                memcpy(CF_region[1], pb_cur->interval_end.data(), sizeof(float)*(D-1)); CF_region[1][D-1] = alloc_mt_df[thread_id][dim_cur]->segments[ind_target_seg+1];
                CF_intervals[thread_id].push_back(CF_region);
                float *temp_CF_in_interval = new float[D]; memcpy(temp_CF_in_interval, target_mt[thread_id], sizeof(float)*D);
                CF_in_intervals[thread_id].push_back(temp_CF_in_interval);
                d2CF[thread_id].push_back(temp_d2target_sup_c);
                
                // debug
                // boost::dynamic_bitset<> temp_intersect;
                // if (mytree->ttype != multi_classif) {
                //   printf("Prediction = %f\n", subtree->check_leaves(target_mt[thread_id], temp_intersect));
                // } else {
                //   printVec(subtree->check_leaves_multi(target_mt[thread_id], temp_intersect), mytree->Nclass);
                // }
                // compare_leaf_sets(alloc_mt_df[thread_id][dim_cur]->intersect_bitset[ind_target_seg], temp_intersect);
                
                // d2target_sup_thread = min(d2target_sup_thread, temp_d2target_sup_c);
                if (update_sup_bound) {
                  update_minimum(d2target_sup_thread, temp_d2target_sup_c); // had time to change in the other threads, so take min again. lock-free operation !
                }
                temp_found_solution[thread_id] = true;
                continue;
              } else { // backtracking
                // dim_cur--;
                dim_cur = std::max(dim_start, dim_cur-1); // useful when dim_start == D-1
                continue;
              }
            } else { // backtracking
              // dim_cur--;
              dim_cur = std::max(dim_start, dim_cur-1); // useful when dim_start == D-1
              continue;
            }
          }
          
          // compute distance of query to the intersection regions
          pb_cur->d2regions(alloc_mt_df[thread_id][dim_cur], pt_query[dim_cur], true);
          
          // order by distance to intersection regions
          int *order_cur = explore_order[dim_cur+1];
          
          // std::iota(order_cur, order_cur+pb_cur->nb_seg, 0);
          // checking if segments of pb_cur have a potential to be target regions
          int temp_nb2explore = 0;
          for (int p=0; p < pb_cur->nb_seg; p++) {
            if (alloc_mt_df[thread_id][dim_cur]->dquery2regions[p] <= d2target_sup_thread) {
              if (check_has_target) {
                if (pb_cur->has_target(alloc_mt_df[thread_id][dim_cur], target_class, p)) { // pb_cur->has_target_binary(alloc_mt_df[thread_id][dim_cur], target_class, p)) {
                  order_cur[temp_nb2explore] = p;
                  temp_nb2explore++;
                }
              } else {
                order_cur[temp_nb2explore] = p;
                temp_nb2explore++;
              }
            }
          }
          
          // std::sort(order_cur, order_cur+pb_cur->nb_seg, [&](int i1, int i2) { return pb_cur->dquery_seg[i1] < pb_cur->dquery_seg[i2]; } );
          std::sort(order_cur, order_cur+temp_nb2explore, //+pb_cur->nb_seg, 
                    [&](int i1, int i2) { return alloc_mt_df[thread_id][dim_cur]->dquery2regions[i1] < alloc_mt_df[thread_id][dim_cur]->dquery2regions[i2]; } );
          // int temp_nb2explore = pb_cur->nb_seg;
          // while (temp_nb2explore && alloc_mt_df[thread_id][dim_cur]->dquery2regions[order_cur[temp_nb2explore-1]] > d2target_sup_thread) temp_nb2explore--;
          
          // stack regions in order for potential future exploration
          if (temp_nb2explore) {
            cursor[dim_cur+1] = 0;
            nb2explore[dim_cur+1] = temp_nb2explore;
            dim_cur++;
          }
          
        }
        
        if (cnt >= budget) {
          nb_approx++;
          // printf("[%d] : Max budget reached: %d seg explored [%d] - [%d]\n", c, cursor[dim_start], temp_Ntot[c], cnt);
        } else {
          // printf("[%d] : all %d seg explored\n", c, temp_Ntot[c]);
        }
        
        delete[] order_pb;
        for (int d=dim_start+1; d < D; d++) delete[] explore_order[d];
        
      }
      
      explore_path.clear();
      explore_order.clear();
      delete[] cursor;
      nb2explore.clear();
      
    }
    
  );
  
  // found which one is the best solution in target_mt
  for (int t=0; t < nb_threads; t++) {
    if (temp_found_solution[t]) {
      found_solution = true;
      break;
    }
  }
  
  if (!nb_approx && found_solution) {
    printf("Exact solution found [%f]\n", sqrt((float)d2target_sup_thread));
    is_approx = false;
  } else {
    is_approx = true;
  }
  
  if (found_solution) {
    float dist_t;
    int ind_opt = 0;
    for (int t=0; t < nb_threads; t++) {
      float dist_t = dist_sq(target_mt[t], pt_query, D);
      if (!t) {
        dist_CF2query = dist_t;
      } else {
        if (dist_t < dist_CF2query) {
          dist_CF2query = dist_t;
          ind_opt = t;
        }
      }
    }
    memcpy(target, target_mt[ind_opt], sizeof(float)*D);
    printf("dist_CF2query = %f [%d]\n", sqrt(dist_CF2query), ind_opt);
  } else {
    printf("No solution found for the data sup bound and the thresh_dec value provided. Try to increase the allocated search budget, make the sup bound less tight, or change the thresh_dec value\n");
  }
  
  for (int t=0; t < nb_threads; t++) {
    for (int d=dim_start; d < D; d++) {
      delete alloc_mt_df[t][d];
    }
    alloc_mt_df[t].clear();
  }
  alloc_mt_df.clear();
  
  temp_found_solution.clear();
}



// version avec un alloc mt per dimension pour éviter les problèmes de corruption dus au parcours depth first
void CF_finder::continue_depth_first_regression(const int budget) {
  
  int nb_chunks = temp_Ntot.size();
  int dim_start = ind_dim_cur-1; // pb_per_dims_mt[ind_dim_cur-1][x][0]->ind_dim;
  
  alloc_mt_df.resize(nb_threads);
  for (int t=0; t < nb_threads; t++) {
    alloc_mt_df[t].resize(D);
    for (int d=dim_start; d < D; d++) {
      alloc_mt_df[t][d] = new mem_allocator(subtree->Nleaves, mytree->ttype, mytree->Nclass);
    }
  }
  
  tbb::atomic<float> d2target_sup_thread = d2target_sup;
  tbb::atomic<int> nb_approx=0;
  
  printf("Starting depth first in dimension %d using %d pb chunks\n", dim_start, nb_chunks);
  
  // tbb::concurrent_vector<bool> temp_found_solution(nb_threads, false);
  vector<bool> temp_found_solution(nb_threads, false);
  
  tbb::parallel_for(
    blocked_range<int>(0, nb_chunks),
    [&](tbb::blocked_range<int> r) {
      int thread_id = tbb::this_task_arena::current_thread_index(); // b
      
      vector<interv1d_problem_fast *> explore_path(D, NULL);
      vector<int *> explore_order(D, NULL);
      unsigned int *cursor = new unsigned int[D];
      vector<int> nb2explore(D);
      
      for (int c=r.begin(); c < r.end(); ++c) {
        
        float temp_d2target_sup_c = -1.f;
        
        // gather all distances to formed regions
        float *temp_d2region = new float[temp_Ntot[c]];
        int max_Ninterv = -1;
        for (int r=0; r < temp_Ntot[c]; r++) {
          temp_d2region[r] = pb_per_dims_mt[dim_start][c][r]->dprev;
          max_Ninterv = std::max(max_Ninterv, (int)pb_per_dims_mt[dim_start][c][r]->Ninterv);
        }
        // order by distance to intersection regions
        int *order_pb = new int[temp_Ntot[c]]; std::iota(order_pb, order_pb+temp_Ntot[c], 0);
        std::sort(order_pb, order_pb+temp_Ntot[c], [&](int i1, int i2) { return temp_d2region[i1] < temp_d2region[i2]; } );
        delete[] temp_d2region;
        
        // pre-allocate explore_order in all dimensions
        for (int d=dim_start+1; d < D; d++) explore_order[d] = new int[2*max_Ninterv];
        
        // we can only explore problems that were created and not solved yet
        explore_order[dim_start] = order_pb;
        nb2explore[dim_start] = temp_Ntot[c];
        // cursor[dim_start] = 0; // using memset later
        // explore_path[dim_start] = pb_per_dims_mt[dim_start][c][order_pb[0]];
        
        int cnt=0;
        
        int dim_cur = dim_start;
        interv1d_problem_fast *pb_cur;
        memset(cursor, 0, sizeof(unsigned int)*D);
        while(cnt < budget) {
          cnt++;
          
          if (cursor[dim_cur] == nb2explore[dim_cur]) { // backtracking
            if (dim_cur == dim_start) {
              // printf("[%d] : No target region in class %d\n", p, target_class);
              break;
            }
            dim_cur--;
            continue;
          }
          
          // load and solve current problem
          int ind_next = explore_order[dim_cur][cursor[dim_cur]];
          if (dim_cur==dim_start) {
            pb_cur = pb_per_dims_mt[dim_start][c][ind_next];
            pb_per_dims_mt[dim_start][c][ind_next] = NULL; // added this to prevent deletion in destructor (will be deleted later in this function)
          } else {
            pb_cur = explore_path[dim_cur-1]->create_next_problem(alloc_mt_df[thread_id][dim_cur-1], ind_next);
          }
          if (pb_cur->dprev < d2target_sup_thread) {
            if (explore_path[dim_cur]) {
              delete explore_path[dim_cur];
              explore_path[dim_cur] = NULL;
            }
            explore_path[dim_cur] = pb_cur;
          } else {
            if (dim_cur == dim_start) break;
            dim_cur--;
            continue;
          }
          
          cursor[dim_cur]++; // move cursor to the next problem
          // pb_cur->printInfo();
          // if (dim_cur==dim_start+1) return;
          // printf("Current thread = %d [%d]\n", thread_id, nb_threads);
          pb_cur->solve_with_inf(alloc_mt_df[thread_id][dim_cur]); // solve pb in dimension dim_cur+1
          
          if (dim_cur == D-1) { // search for potential target region
            pb_cur->do_scoring_regression(alloc_mt_df[thread_id][dim_cur]);
            if (pb_cur->nb_target) { // we found target regions
              pb_cur->d2regions(alloc_mt_df[thread_id][dim_cur], pt_query[dim_cur], true);
              int ind_target_seg;
              temp_d2target_sup_c = pb_cur->findClosestTarget_regression(alloc_mt_df[thread_id][dim_cur], ind_target_seg);
              if (temp_d2target_sup_c <= d2target_sup_thread) {
                printf("[%d] : Found target region (%f) at iteration %d | d2target = %f [%d]!\n", thread_id, alloc_mt_df[thread_id][dim_cur]->scores_binary[ind_target_seg], 
                       cnt, sqrt(temp_d2target_sup_c), ind_target_seg);
                for (int d=0; d < D-1; d++) target_mt[thread_id][d] = pb_cur->ptc_from_root[d];
                target_mt[thread_id][D-1] = alloc_mt_df[thread_id][dim_cur]->ptc2seg[ind_target_seg]; // pb_cur->ptc2seg[ind_target_seg];
                
                float **CF_region = makeMatrix<float>(2, D);
                memcpy(CF_region[0], pb_cur->interval_start.data(), sizeof(float)*(D-1)); CF_region[0][D-1] = alloc_mt_df[thread_id][dim_cur]->segments[ind_target_seg];
                memcpy(CF_region[1], pb_cur->interval_end.data(), sizeof(float)*(D-1)); CF_region[1][D-1] = alloc_mt_df[thread_id][dim_cur]->segments[ind_target_seg+1];
                CF_intervals[thread_id].push_back(CF_region);
                float *temp_CF_in_interval = new float[D]; memcpy(temp_CF_in_interval, target_mt[thread_id], sizeof(float)*D);
                CF_in_intervals[thread_id].push_back(temp_CF_in_interval);
                d2CF[thread_id].push_back(temp_d2target_sup_c);
                
                // d2target_sup_thread = min(d2target_sup_thread, temp_d2target_sup_c);
                if (update_sup_bound) {
                  update_minimum(d2target_sup_thread, temp_d2target_sup_c); // had time to change in the other threads, so take min again. lock-free operation !
                }
                temp_found_solution[thread_id] = true;
                continue;
              } else { // backtracking
                // dim_cur--;
                dim_cur = std::max(dim_start, dim_cur-1); // useful when dim_start == D-1
                continue;
              }
            } else { // backtracking
              // dim_cur--;
              dim_cur = std::max(dim_start, dim_cur-1); // useful when dim_start == D-1
              continue;
            }
          }
          
          // compute distance of query to the intersection regions
          pb_cur->d2regions(alloc_mt_df[thread_id][dim_cur], pt_query[dim_cur], true);
          
          // order by distance to intersection regions
          int *order_cur = explore_order[dim_cur+1];
          
          // std::iota(order_cur, order_cur+pb_cur->nb_seg, 0);
          // checking if segments of pb_cur have a potential to be target regions
          int temp_nb2explore = 0;
          for (int p=0; p < pb_cur->nb_seg; p++) {
            if (alloc_mt_df[thread_id][dim_cur]->dquery2regions[p] <= d2target_sup_thread) {
              if (check_has_target) {
                if (pb_cur->has_target_regression(alloc_mt_df[thread_id][dim_cur], p)) {
                  order_cur[temp_nb2explore] = p;
                  temp_nb2explore++;
                }
              } else {
                order_cur[temp_nb2explore] = p;
                temp_nb2explore++;
              }
            }
          }
          
          // std::sort(order_cur, order_cur+pb_cur->nb_seg, [&](int i1, int i2) { return pb_cur->dquery_seg[i1] < pb_cur->dquery_seg[i2]; } );
          std::sort(order_cur, order_cur+temp_nb2explore, //+pb_cur->nb_seg, 
                    [&](int i1, int i2) { return alloc_mt_df[thread_id][dim_cur]->dquery2regions[i1] < alloc_mt_df[thread_id][dim_cur]->dquery2regions[i2]; } );
          // int temp_nb2explore = pb_cur->nb_seg;
          // while (temp_nb2explore && alloc_mt_df[thread_id][dim_cur]->dquery2regions[order_cur[temp_nb2explore-1]] > d2target_sup_thread) temp_nb2explore--;
          
          // stack regions in order for potential future exploration
          if (temp_nb2explore) {
            cursor[dim_cur+1] = 0;
            nb2explore[dim_cur+1] = temp_nb2explore;
            dim_cur++;
          }
          
        }
        
        if (cnt >= budget) {
          nb_approx++;
          // printf("[%d] : Max budget reached: %d seg explored [%d] - [%d]\n", c, cursor[dim_start], temp_Ntot[c], cnt);
        } else {
          // printf("[%d] : all %d seg explored\n", c, temp_Ntot[c]);
        }
        
        delete[] order_pb;
        for (int d=dim_start+1; d < D; d++) delete[] explore_order[d];
        
      }
      
      explore_path.clear();
      explore_order.clear();
      delete[] cursor;
      nb2explore.clear();
      
    }
    
  );
  
  // found which one is the best solution in target_mt
  for (int t=0; t < nb_threads; t++) {
    if (temp_found_solution[t]) {
      found_solution = true;
      break;
    }
  }
  
  if (!nb_approx && found_solution) {
    printf("Exact solution found [%f]\n", sqrt((float)d2target_sup_thread));
    is_approx = false;
  } else {
    is_approx = true;
  }
  
  if (found_solution) {
    float dist_t;
    int ind_opt = 0;
    for (int t=0; t < nb_threads; t++) {
      float dist_t = dist_sq(target_mt[t], pt_query, D);
      if (!t) {
        dist_CF2query = dist_t;
      } else {
        if (dist_t < dist_CF2query) {
          dist_CF2query = dist_t;
          ind_opt = t;
        }
      }
    }
    memcpy(target, target_mt[ind_opt], sizeof(float)*D);
    printf("dist_CF2query = %f [%d]\n", sqrt(dist_CF2query), ind_opt);
  } else {
    printf("No solution found for the data sup bound and the thresh_dec value provided. Try to increase the allocated search budget, make the sup bound less tight, or change the thresh_dec value\n");
  }
  
  for (int t=0; t < nb_threads; t++) {
    for (int d=dim_start; d < D; d++) {
      delete alloc_mt_df[t][d];
    }
    alloc_mt_df[t].clear();
  }
  alloc_mt_df.clear();
  
  temp_found_solution.clear();
}


