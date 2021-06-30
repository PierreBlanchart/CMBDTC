/** CF_finder.h
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


#ifndef DEF_CF_FINDER_H
#define DEF_CF_FINDER_H

#include "interv1d_problem_fast.h"


class CF_finder {
public:
  CF_finder(tree_model *mytree_, float *pt_query_, int target_class_, float d2target_sup_=-1.f, int budget_bt=1e6, 
            const bool check_has_target_=true, const bool update_sup_bound_=true);
  
  CF_finder(tree_model *mytree_, float *pt_query_, float *target_interval_, float d2target_sup_=-1.f, int budget_bt=1e6, 
            bool check_has_target_=true, const bool update_sup_bound_=true, const bool do_multiclass_=false) : CF_finder(mytree_, pt_query_, 
            -1, d2target_sup_, budget_bt, check_has_target_, update_sup_bound_) {
    target_interval = target_interval_;
  };
  
  ~CF_finder() {
    printf("Cleaning CF object\n");
    
    if (mytree) delete mytree;
    if (using_subtree && subtree) delete subtree;
    
    for (int t=0; t < nb_threads; t++) {
      delete[] target_mt[t];
    }
    target_mt.clear();
    
    for (int t=0; t < nb_threads; t++) {
      for (int l=0; l < CF_intervals[t].size(); l++) {
        deleteMatrix<float>(CF_intervals[t][l]);
        delete[] CF_in_intervals[t][l];
      }
      CF_intervals[t].clear();
      CF_in_intervals[t].clear();
      d2CF[t].clear();
    }
    CF_intervals.clear();
    CF_in_intervals.clear();
    d2CF.clear();
    
    // deleteTensor<float>(nb_threads, CF_intervals);
    
    for (int t=0; t < nb_threads; t++) delete alloc_mt[t];
    alloc_mt.clear();
    
    for (int d=0; d < recorded_dim_start+1; d++) {
      // for (int c=0; c < pb_per_dims_mt[d].size(); c++) {
      tbb::parallel_for(
        blocked_range<int>(0, pb_per_dims_mt[d].size()),
        [&](tbb::blocked_range<int> r) {
          for (int c=r.begin(); c < r.end(); ++c) {
            for (int n=0; n < pb_per_dims_mt[d][c].size(); n++) {
              if (pb_per_dims_mt[d][c][n]) {
                // pb_per_dims_mt[d][c][n]->index.clear();
                delete pb_per_dims_mt[d][c][n];
              }
            }
            pb_per_dims_mt[d][c].clear();
          }
        });
      pb_per_dims_mt[d].clear();
    }
    pb_per_dims_mt.clear(); // D x Nchunks x chunk_sz problems
    temp_Ntot.clear();
    
    if (target) delete[] target;
  };
  
  void start_width_first(const int max_dim, const int chunk_sz_start=1000);
  
  void start_width_first_regression(const int max_dim, const int chunk_sz_start=1000);
  
  void continue_depth_first(const int budget);
  
  void continue_depth_first_regression(const int budget);
  
public:
  tree_model *mytree=NULL, *subtree=NULL;
  bool using_subtree = false;
  int D;
  float *pt_query=NULL;
  int target_class=-1;
  float *target_interval=NULL; // targeted prediction interval (regression)
  
  bool check_has_target=true;
  bool update_sup_bound=true;
  
  float d2target_sup=-1.f;
  float *target=NULL, dist_CF2query = -1.f;
  bool is_approx = false, found_solution=false;
  
  unsigned int NchunksPerThread = 4; // maximum number of chunks that a thread can treat (useful to dimension chunks)
  unsigned int max_per_worker = 1e6;
  
  int ind_dim_cur = 0, Ntot_cur=0;
  int recorded_dim_start = -1; // when to start depth first exploration
  
  // multithread stuffs
  // tbb::concurrent_vector<int> temp_Ntot;
  // tbb::concurrent_vector<tbb::concurrent_vector<vector<interv1d_problem_fast*> > > pb_per_dims_mt; // D x Nchunks x chunk_sz problems
  vector<int> temp_Ntot;
  vector< vector< vector< interv1d_problem_fast*> > > pb_per_dims_mt; // D x Nchunks x chunk_sz problems
  
  
  int nb_threads;
  // tbb::concurrent_vector<mem_allocator *> alloc_mt;
  // tbb::concurrent_vector<vector<mem_allocator *>> alloc_mt_df; // for depth-first search
  // tbb::concurrent_vector<float *> target_mt;
  vector<mem_allocator *> alloc_mt;
  vector<vector<mem_allocator *>> alloc_mt_df; // for depth-first search
  vector<float *> target_mt;
  
  // CF sets
  // float ***CF_intervals=NULL;
  vector< vector<float **> > CF_intervals; // definition of sets as multi-dimensional intervals
  vector< vector<float *> > CF_in_intervals; // CF example associated with each multi-dimensional interval
  vector< vector<float> > d2CF; // distance to the CF example for each of the multi-dimensional intervals
  
  // debug
  void compare_leaf_sets(boost::dynamic_bitset<> &set1, boost::dynamic_bitset<> &set2) {
    int cnt1 = 0, cnt_diff1 = 0, cnt_diff2 = 0;
    for (int n=0; n < set1.size(); n++) {
      if (set1[n]) {
        if (set2[n]) {
          cnt1++;
        } else {
          cnt_diff1++;
        }
      } else {
        if (set2[n]) cnt_diff2++;
      }
    }
    
    printf("cnt_same = %d | cnt_diff1 = %d | cnt_diff2 = %d | [%d]\n", cnt1, cnt_diff1, cnt_diff2, (int)set1.size());
  };
};


#endif
