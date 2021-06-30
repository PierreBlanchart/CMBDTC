/** interv1d_problem_fast.cpp
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


#include "interv1d_problem_fast.h"


interv1d_problem_fast::interv1d_problem_fast(tree_model *mytree_, const int Ninterv_, const int ind_dim_, const boost::dynamic_bitset<> &index_) {
  mytree = mytree_;
  
  Ninterv = Ninterv_;
  ind_dim = ind_dim_;
  
  index = index_; // deep copy
}



interv1d_problem_fast::~interv1d_problem_fast() {
  // ptc2seg.clear();
  // delete[] labels;
  labels.clear();
  nb_per_class.clear();
  ptc_from_root.clear();
  interval_start.clear();
  interval_end.clear();
  index.clear();
}



// uses mem_allocator
void interv1d_problem_fast::solve_with_inf(mem_allocator *thread_alloc, const bool verbose_) {
  
  bool **temp_is_inf = mytree->is_inf[ind_dim];
  unsigned int *pos_index_dim = mytree->pos_in_index[ind_dim];
  float **temp_model_intervals = mytree->model_intervals[ind_dim];
  
  thread_alloc->temp_intersect.reset();
  thread_alloc->is_start.reset();
  
  unsigned int nb_neg_inf = 0, nb_inf = 0, nb_ninf= 0;
  size_t l = index.find_first();
  // int cnt_dbg = 0;
  while (l != string::npos) {
    
    // start of interval "l"
    if (temp_is_inf[0][l]) {
      thread_alloc->temp_intersect[l] = 1;
      nb_neg_inf++;
    } else {
      thread_alloc->is_start[nb_ninf] = 1;
      thread_alloc->temp_index[nb_ninf] = l;
      thread_alloc->temp_intervals[nb_ninf] = temp_model_intervals[0][l];
      thread_alloc->temp_pos[nb_ninf] = pos_index_dim[2*l];
      nb_ninf++;
    }
    
    // end of interval "l"
    if (temp_is_inf[1][l]) {
      nb_inf++;
    } else {
      thread_alloc->temp_index[nb_ninf] = l;
      thread_alloc->temp_intervals[nb_ninf] = temp_model_intervals[1][l];
      thread_alloc->temp_pos[nb_ninf] = pos_index_dim[2*l + 1];
      nb_ninf++;
    }
    
    l = index.find_next(l);
  }
  
  // printf("cnt_dbg = %d\n", cnt_dbg);
  
  // faire un tuple ou pair<element, pair>> pour éviter tout ça !!!!
  // sort temp_intervals by sorting temp_pos (boost::integer_sort)
  std::iota(thread_alloc->sorted_index, thread_alloc->sorted_index+nb_ninf, 0);
  sort(thread_alloc->sorted_index, thread_alloc->sorted_index+nb_ninf, 
       [&](unsigned int i1, unsigned int i2) { return thread_alloc->temp_pos[i1] < thread_alloc->temp_pos[i2]; } );
  
  for (int n=0; n < nb_ninf; n++) {
    thread_alloc->sorted_temp_intervals[n] = thread_alloc->temp_intervals[thread_alloc->sorted_index[n]];
    thread_alloc->sorted_is_start[n] = thread_alloc->is_start[thread_alloc->sorted_index[n]];
    thread_alloc->sorted_temp_index[n] = thread_alloc->temp_index[thread_alloc->sorted_index[n]];
  }
  
  bool is_started = false;
  if (nb_neg_inf) {
    thread_alloc->segments[0] = neg_inf;
    is_started = true;
  }
  
  float pos_cur, pos_prev = neg_inf;
  nb_seg = 0;
  
  unsigned int cnt = nb_neg_inf;
  for (int n=0; n < nb_ninf; n++) {
    pos_cur = thread_alloc->sorted_temp_intervals[n];
    if (!is_started) {
      thread_alloc->segments[0] = pos_prev = pos_cur;
      is_started = true;
    }
    if (pos_cur > pos_prev) {
      pos_prev = pos_cur;
      thread_alloc->intersect_bitset[nb_seg].clear(); // added this here
      thread_alloc->intersect_bitset[nb_seg] = thread_alloc->temp_intersect; // deep copy
      thread_alloc->nb_intersect[nb_seg] = cnt;
      thread_alloc->segments[nb_seg+1] = pos_cur;
      nb_seg++;
    }
    thread_alloc->temp_intersect[thread_alloc->sorted_temp_index[n]] = thread_alloc->sorted_is_start[n];
    if (thread_alloc->sorted_is_start[n]) {
      cnt++;
    } else {
      cnt--;
    }
  }
  
  // printf("nb_inf = %d\n", nb_inf);
  if (nb_inf) {
    thread_alloc->intersect_bitset[nb_seg].clear();
    thread_alloc->intersect_bitset[nb_seg] = thread_alloc->temp_intersect;
    thread_alloc->nb_intersect[nb_seg] = nb_inf;
    thread_alloc->segments[nb_seg+1] = inf;
    nb_seg++;
  }
  
  if (verbose_) {
    printf("dim %d: %d segments found\n", ind_dim, nb_seg);
    for (int s=0; s < nb_seg; s++) {
      int nb_s = thread_alloc->intersect_bitset[s].count();
      printf("segment %d: %d [%f --> %f]\n", s, nb_s, thread_alloc->segments[s], thread_alloc->segments[s+1]);
    }
  }
  
  is_solved = true;
  
  index.clear();
}



interv1d_problem_fast* interv1d_problem_fast::create_next_problem(mem_allocator *thread_alloc, const int &ind_seg) {
  interv1d_problem_fast *next_pb = new interv1d_problem_fast(mytree, thread_alloc->nb_intersect[ind_seg], ind_dim+1, thread_alloc->intersect_bitset[ind_seg]);
  if (computed_d2regions) {
    next_pb->dprev = thread_alloc->dquery2regions[ind_seg];
    if (ind_dim) {
      next_pb->ptc_from_root = this->ptc_from_root;
      next_pb->interval_start = this->interval_start;
      next_pb->interval_end = this->interval_end;
    }
    next_pb->ptc_from_root.push_back(thread_alloc->ptc2seg[ind_seg]);
    next_pb->interval_start.push_back(thread_alloc->segments[ind_seg]);
    next_pb->interval_end.push_back(thread_alloc->segments[ind_seg+1]);
  }
  
  // thread_alloc->intersect_bitset[ind_seg].clear();
  
  return next_pb;
}



int interv1d_problem_fast::d2seg(mem_allocator *thread_alloc, const float &query, bool verbose) {
  
  // dquery_seg.resize(nb_seg);
  // ptc2seg.resize(nb_seg);
  
  int n=0, ind_query=-1;
  float *segments = thread_alloc->segments;
  float *dquery_seg = thread_alloc->dquery_seg;
  float *ptc2seg = thread_alloc->ptc2seg;
  
  /**
   for (int n=0; n < nb_seg; n++) {
   if (query >= segments[n]) {
   if (query < segments[n+1]) {
   dquery_seg[n] = 0;
   ptc2seg[n] = query;
   ind_query = n;
   } else {
   dquery_seg[n] = query-(segments[n+1]-1e-5f);
   ptc2seg[n] = segments[n+1]-1e-5f;
   }
   } else {
   dquery_seg[n] = segments[n]-query;
   ptc2seg[n] = segments[n];
   }
   }
   if (ind_query < 0) {
   ind_query = nb_seg;
   }
   **/
  
  while (n < nb_seg && query >= segments[n+1]) {
    dquery_seg[n] = query-(segments[n+1]-1e-5f);
    ptc2seg[n] = segments[n+1]-1e-5f;
    n++;
  }
  
  if (n < nb_seg) {
    dquery_seg[n] = 0;
    ptc2seg[n] = query;
    ind_query = n;
    n++;
    for (; n < nb_seg; n++) {
      dquery_seg[n] = segments[n]-query;
      ptc2seg[n] = segments[n];
    }
  } else {
    ind_query = nb_seg;
  }
  
  return ind_query;
  
}



void interv1d_problem_fast::d2regions(mem_allocator *thread_alloc, float query, bool compute_d2seg, bool verbose) {
  if (compute_d2seg) d2seg(thread_alloc, query, verbose);
  // dquery2regions = new float[nb_seg];
  for (int n=0; n < nb_seg; n++) {
    thread_alloc->dquery2regions[n] = dprev + (thread_alloc->dquery_seg[n]*thread_alloc->dquery_seg[n]);
  }
  computed_d2regions = true;
}



void interv1d_problem_fast::do_scoring_binary(mem_allocator *thread_alloc) {
  nb_per_class.resize(2, 0);
  labels.resize(nb_seg);
  
  float temp_sum;
  size_t l;
  for (int r=0; r < nb_seg; r++) {
    l = thread_alloc->intersect_bitset[r].find_first();
    temp_sum = 0.f;
    while (l != string::npos) {
      temp_sum += mytree->score_leaves[l];
      l = thread_alloc->intersect_bitset[r].find_next(l);
    }
    thread_alloc->scores_binary[r] = mytree->aggr_function(temp_sum);
    
    if (thread_alloc->scores_binary[r] > mytree->thresh_dec) {
      labels[r] = 1;
      nb_per_class[1]++;
    } else {
      nb_per_class[0]++;
    }
  }
  
}



void interv1d_problem_fast::do_scoring_multi(mem_allocator *thread_alloc) {
  nb_per_class.resize(mytree->Nclass, 0);
  labels.resize(nb_seg);
  for (int c=0; c < mytree->Nclass; c++) memset(thread_alloc->scores_multi[c], 0, sizeof(float)*nb_seg);
  
  size_t l;
  int cat_max;
  float val_max;
  for (int r=0; r < nb_seg; r++) {
    l = thread_alloc->intersect_bitset[r].find_first();
    while (l != string::npos) { // looping over intervals present in segment
      thread_alloc->scores_multi[mytree->index_class_leaves[l]][r] += mytree->score_leaves[l];
      l = thread_alloc->intersect_bitset[r].find_next(l);
    }
    
    cat_max = 0;
    val_max = thread_alloc->scores_multi[0][r] = mytree->aggr_function(thread_alloc->scores_multi[0][r]);
    for (int c=1; c < mytree->Nclass; c++) {
      thread_alloc->scores_multi[c][r] = mytree->aggr_function(thread_alloc->scores_multi[c][r]);
      if (thread_alloc->scores_multi[c][r] > val_max) {
        val_max = thread_alloc->scores_multi[c][r];
        cat_max = c;
      }
    }
    labels[r] = cat_max;
    nb_per_class[cat_max]++;
  }
  
}



void interv1d_problem_fast::do_scoring_regression(mem_allocator *thread_alloc) {
  labels.resize(nb_seg);
  
  float temp_sum;
  size_t l;
  for (int r=0; r < nb_seg; r++) {
    l = thread_alloc->intersect_bitset[r].find_first();
    temp_sum = 0.f;
    while (l != string::npos) {
      temp_sum += mytree->score_leaves[l];
      l = thread_alloc->intersect_bitset[r].find_next(l);
    }
    
    thread_alloc->scores_binary[r] = mytree->aggr_function(temp_sum);
    if (thread_alloc->scores_binary[r] >= mytree->target_interval[0] && thread_alloc->scores_binary[r] <= mytree->target_interval[1]) {
      labels[r] = 1;
      nb_target++;
    }
  }
  
}



void interv1d_problem_fast::do_scoring(mem_allocator *thread_alloc) {
  
  switch(mytree->ttype) {
  case binary_classif: return this->do_scoring_binary(thread_alloc);
  case multi_classif : return this->do_scoring_multi(thread_alloc);
  case regression_task: return this->do_scoring_regression(thread_alloc);
  default: break;
  }
  
}



bool interv1d_problem_fast::has_target_binary(mem_allocator *thread_alloc, const int &target_class, const int &ind_seg) {
  
  bool has_target = false;
  float thresh_sum = mytree->thresh_sum; // inv_sigmoid(thresh_dec)
  
  size_t l = thread_alloc->intersect_bitset[ind_seg].find_first();
  if (target_class) {
    
    if (thresh_sum <= 0) {
      while (l != string::npos) {
        if (mytree->score_leaves[l] > thresh_sum) {
          has_target = true;
          break;
        }
        l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
      }
    } else {
      float accum = 0.f;
      while (l != string::npos) {
        if (mytree->score_leaves[l] > thresh_sum) {
          has_target = true;
          break;
        }
        if (mytree->score_leaves[l] > 0) accum += mytree->score_leaves[l];
        if (accum > thresh_sum) {
          has_target = true;
          break;
        }
        l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
      }
    }
    
  } else {
    
    if (thresh_sum >= 0) {
      while (l != string::npos) {
        if (mytree->score_leaves[l] <= thresh_sum) {
          has_target = true;
          break;
        }
        l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
      }
    } else { // thresh_sum < 0
      float accum = 0.f;
      while (l != string::npos) {
        if (mytree->score_leaves[l] <= thresh_sum) {
          has_target = true;
          break;
        }
        if (mytree->score_leaves[l] < 0) accum += mytree->score_leaves[l];
        if (accum <= thresh_sum) {
          has_target = true;
          break;
        }
        l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
      }
    }
    
  }
  
  return has_target;
}



bool interv1d_problem_fast::has_target_multi(mem_allocator *thread_alloc, const int &target_class, const int &ind_seg) {
  
  bool has_target_class = false;
  
  size_t l = thread_alloc->intersect_bitset[ind_seg].find_first();
  while (l != string::npos) { // looping over intervals present in "ind_seg" segment
    if (mytree->index_class_leaves[l]==target_class) {
      has_target_class = true;
      break;
    }
    l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
  }
  
  return has_target_class;
}



// only works for "id_fun" aggregation function (squared error loss)
// replace bounds with inv_sigmoid(target_interval) for logistic regression
bool interv1d_problem_fast::has_target_regression(mem_allocator *thread_alloc, const int &ind_seg) {
  
  if (mytree->fun_aggr != id_fun) return true;
  
  bool has_interval = false;
  float *target_interval = mytree->target_interval;
  // printf("%f, %f\n", target_interval[0], target_interval[1]);
  
  size_t l = thread_alloc->intersect_bitset[ind_seg].find_first();
  
  // checking lower bound
  if (target_interval[0] <= 0) {
    while (l != string::npos) {
      if (mytree->score_leaves[l] > target_interval[0]) {
        has_interval = true;
        break;
      }
      l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
    }
  } else {
    float accum = 0.f;
    while (l != string::npos) {
      if (mytree->score_leaves[l] > target_interval[0]) {
        has_interval = true;
        break;
      }
      if (mytree->score_leaves[l] > 0) accum += mytree->score_leaves[l];
      if (accum > target_interval[0]) {
        has_interval = true;
        break;
      }
      l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
    }
  }
  
  if (!has_interval) return false;
  
  
  // checking upper bound
  has_interval = false;
  if (target_interval[1] >= 0) {
    while (l != string::npos) {
      if (mytree->score_leaves[l] <= target_interval[1]) {
        has_interval = true;
        break;
      }
      l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
    }
  } else { // target_interval[1] < 0
    float accum = 0.f;
    while (l != string::npos) {
      if (mytree->score_leaves[l] <= target_interval[1]) {
        has_interval = true;
        break;
      }
      if (mytree->score_leaves[l] < 0) accum += mytree->score_leaves[l];
      if (accum <= target_interval[1]) {
        has_interval = true;
        break;
      }
      l = thread_alloc->intersect_bitset[ind_seg].find_next(l);
    }
  }
  
  
  return has_interval;
}



bool interv1d_problem_fast::has_target(mem_allocator *thread_alloc, const int &target_class, const int &ind_seg) {
  
  switch(mytree->ttype) {
  case binary_classif: return this->has_target_binary(thread_alloc, target_class, ind_seg);
  case multi_classif : return this->has_target_multi(thread_alloc, target_class, ind_seg);
  case regression_task: return this->has_target_regression(thread_alloc, ind_seg);
  default: return false;
  }
  
}



float interv1d_problem_fast::findClosestTarget_classif(mem_allocator *thread_alloc, const int &target_class, int &ind_closest_seg) {
  float dclosest = inf;
  ind_closest_seg = -1;
  for (int r=0; r < nb_seg; r++) {
    if (labels[r] == target_class) {
      // if ((target_class && scores_binary[r] > mytree->thresh_dec) || (!target_class && scores_binary[r] < mytree->thresh_dec)) {
      if (thread_alloc->dquery2regions[r] < dclosest) {
        dclosest = thread_alloc->dquery2regions[r];
        ind_closest_seg = r;
      }
    }
  }
  return dclosest;
}



float interv1d_problem_fast::findClosestTarget_regression(mem_allocator *thread_alloc, int &ind_closest_seg) {
  float dclosest = inf;
  ind_closest_seg = -1;
  for (int r=0; r < nb_seg; r++) {
    if (labels[r]) {
      if (thread_alloc->dquery2regions[r] < dclosest) {
        dclosest = thread_alloc->dquery2regions[r];
        ind_closest_seg = r;
      }
    }
  }
  return dclosest;
}



float interv1d_problem_fast::findClosestTarget(mem_allocator *thread_alloc, const int &target_class, int &ind_closest_seg) {
  
  switch(mytree->ttype) {
  case binary_classif: 
  case multi_classif : return this->findClosestTarget_classif(thread_alloc, target_class, ind_closest_seg);
  case regression_task: return this->findClosestTarget_regression(thread_alloc, ind_closest_seg);
  default: return -1.f;
  }
  
}

