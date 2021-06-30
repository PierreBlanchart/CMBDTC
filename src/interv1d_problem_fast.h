/** interv1d_problem_fast.h
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


#ifndef DEF_INTERV1D_PROBLEM_FAST_H
#define DEF_INTERV1D_PROBLEM_FAST_H

#include "tree_model.h"
#include "mem_allocator.h"


class interv1d_problem_fast {
public:
  interv1d_problem_fast(tree_model *mytree_, const int Ninterv_, const int ind_dim_, const boost::dynamic_bitset<> &index_);

  ~interv1d_problem_fast(void);

public:
  tree_model *mytree=NULL;

  int ind_dim=0;
  
  unsigned int Ninterv;
  boost::dynamic_bitset<> index;

  unsigned int nb_seg=0;
  
  float dprev=0.f; // accumulated (squared) distance of query to regions propagated from previous dimensions
  // vector<float> dquery_seg; // contains distance of query to segments in current problem dimension
  // vector<float> ptc2seg; // contains closest point of each segment to the query in the current dimension
  
  bool computed_d2regions = false;
  // float *dquery2regions=NULL; // contains (squared) distance of query to the formed "ind_dim"-dimensional regions
  
  // unsigned int *labels;
  vector<int> nb_per_class, labels;
  int nb_target = 0;

  bool is_solved=false;
  
  vector<float> ptc_from_root;
  vector<float> interval_start, interval_end;

public:
  void solve_with_inf(mem_allocator *thread_alloc, const bool verbose_=false); // using mem_allocator
  
  interv1d_problem_fast* create_next_problem(mem_allocator *thread_alloc, const int &ind_seg);
  
  int d2seg(mem_allocator *thread_alloc, const float &query, bool verbose=false);
  
  void d2regions(mem_allocator *thread_alloc, float query=0.f, bool compute_d2seg=false, bool verbose=false);
  
  void do_scoring(mem_allocator *); // interface function
  void do_scoring_binary(mem_allocator *);
  void do_scoring_multi(mem_allocator *);
  void do_scoring_regression(mem_allocator *);
  
  bool has_target(mem_allocator *thread_alloc, const int &target_class, const int &ind_seg); // interface function
  bool has_target_binary(mem_allocator *thread_alloc, const int &target_class, const int &ind_seg);
  bool has_target_multi(mem_allocator *thread_alloc, const int &target_class, const int &ind_seg);
  bool has_target_regression(mem_allocator *thread_alloc, const int &ind_seg);
  
  float findClosestTarget(mem_allocator *thread_alloc, const int &target_class, int &ind_closest_seg); // interface function
  float findClosestTarget_classif(mem_allocator *thread_alloc, const int &target_class, int &ind_closest_seg); // both binary and multiclass settings
  float findClosestTarget_regression(mem_allocator *thread_alloc, int &ind_closest_seg);
  
  void printInfo() {
    printf("Problem defined in dimension %d | problem status %d\n", ind_dim, (int)is_solved);
  }

};



#endif
