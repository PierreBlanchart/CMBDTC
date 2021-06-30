/** tree_model.cpp
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


#include "tree_model.h"


void tree_model::init(int Nleaves_) {
  Nleaves = Nleaves_;
  model_intervals = makeTensor<float>(D, 2, Nleaves);
  for (int d=0; d < D; d++) {
    std::fill(model_intervals[d][0], model_intervals[d][0]+Nleaves, neg_inf);
    std::fill(model_intervals[d][1], model_intervals[d][1]+Nleaves, inf);
  }
  is_inf = makeTensor<bool>(D, 2, Nleaves, true);
  
  index_class_leaves.resize(Nleaves);
  score_leaves.resize(Nleaves, 0.f);
}



void tree_model::presort() {
  sorted_intervals = makeMatrix<float>(D, 2*Nleaves);
  pos_in_index = makeMatrix<unsigned int>(D, 2*Nleaves);
  
  float *temp_val = new float[2*Nleaves];
  unsigned int *ind_sorted = new unsigned int[2*Nleaves];
  
  for (int d=0; d < D; d++) {
    
    int ind_cur = 0;
    for (int n=0; n < Nleaves; n++) {
      temp_val[ind_cur] = model_intervals[d][0][n];
      temp_val[ind_cur+1] = model_intervals[d][1][n];
      ind_cur+=2;
    }
    
    std::iota(ind_sorted, ind_sorted + 2*Nleaves, 0);
    std::stable_sort(ind_sorted, ind_sorted + 2*Nleaves, [&](int i1, int i2) { return temp_val[i1] < temp_val[i2]; });
    
    for (int n=0; n < 2*Nleaves; n++) {
      sorted_intervals[d][n] = temp_val[ind_sorted[n]];
      pos_in_index[d][ind_sorted[n]] = n;
    }
    
  }
  
  delete[] temp_val;
  delete[] ind_sorted;
}



// select boxes within range fixed by thresh_dsq
tree_model* tree_model::select_boxes(float *query, float thresh_dsq) {
  
  bool *is_in_range = new bool[Nleaves]; memset(is_in_range, 0, sizeof(bool)*Nleaves);
  int nb_in_range = 0;
  
  float dist_n, temp_diff;
  for (int n=0; n < Nleaves; n++) {
    dist_n = 0.f;
    for (int d=0; d < D; d++) {
      if (query[d] < model_intervals[d][0][n]) {
        temp_diff = model_intervals[d][0][n]-query[d];
      } else {
        if (query[d] > model_intervals[d][1][n]) {
          temp_diff = query[d]-model_intervals[d][1][n];
        } else {
          temp_diff = 0.f;
        }
      }
      dist_n += temp_diff*temp_diff;
    }
    
    if (dist_n <= thresh_dsq) {
      is_in_range[n] = true;
      nb_in_range++;
    }
  }
  
  // build new tree model
  tree_model *new_tree;
  if (Nclass > 1) {
    new_tree = new tree_model(D, MAX_DEPTH, Nclass, thresh_dec, fun_aggr);
  } else {
    new_tree = new tree_model(D, MAX_DEPTH, target_interval, fun_aggr);
  }
  new_tree->init(nb_in_range);
  
  int ind_cur = 0;
  for (int n=0; n < Nleaves; n++) {
    if (is_in_range[n]) {
      for (int d=0; d < D; d++) {
        new_tree->model_intervals[d][0][ind_cur] = this->model_intervals[d][0][n];
        new_tree->model_intervals[d][1][ind_cur] = this->model_intervals[d][1][n];
        new_tree->is_inf[d][0][ind_cur] = this->is_inf[d][0][n];
        new_tree->is_inf[d][1][ind_cur] = this->is_inf[d][1][n];
      }
      new_tree->index_class_leaves[ind_cur] = this->index_class_leaves[n];
      new_tree->score_leaves[ind_cur] = this->score_leaves[n];
      ind_cur++;
    }
  }
  
  delete[] is_in_range;
  
  return new_tree;
  
}



tree_model* tree_model::preselect_with_fixed_dimensions(float *fixed_val, unsigned int *fixed_dim, int nb_fixed) {
  
  tree_model *new_tree;
  unsigned int nb_non_fixed = D-nb_fixed;
  unsigned int *host_find_non_fixed = new unsigned int[nb_non_fixed];
  
  // supposing fixed_dim sorted in ascending order
  int ind_cur = 0, ind_cur_fixed = 0;
  for (int d=0; d < D; d++) {
    if (ind_cur_fixed >= nb_fixed || d < fixed_dim[ind_cur_fixed]) {
      host_find_non_fixed[ind_cur] = d;
      ind_cur++;
    } else {
      ind_cur_fixed++;
    }
  }
  // printf("%d | %d\n", nb_non_fixed, ind_cur);
  // return NULL;
  
  if (Nclass > 1) {
    new_tree = new tree_model(nb_non_fixed, MAX_DEPTH, Nclass, thresh_dec, fun_aggr);
  } else {
    new_tree = new tree_model(nb_non_fixed, MAX_DEPTH, target_interval, fun_aggr);
  }
  
  unsigned int nb_in_range, *host_find_in_range;
  
#ifdef ARRAYFIRE
  af::array ind_fixed(nb_fixed, fixed_dim);
  af::array af_model_intervals = reorder(af::array(dim4(Nleaves, 2, D), model_intervals[0][0]), 2, 1, 0); // D x 2 x Nleaves
  af::array af_fixed_intervals = af_model_intervals(ind_fixed, span, span).copy();
  
  af::array af_fixed_val(nb_fixed, fixed_val);
  af::array test1 = tile(af_fixed_val, 1, 1, Nleaves) >= af_fixed_intervals(span, 0, span);
  af::array test2 = tile(af_fixed_val, 1, 1, Nleaves) < af_fixed_intervals(span, 1, span);
  af::array test12 = test1 & test2; // nb_fixed x 1 x Nleaves
  af::array ind_in_range = (sum(test12*1.f, 0) == (float)nb_fixed); // 1 x 1 x Nleaves
  
  // define subtree composed of leaves that are inside "ind_in_range", and dimensions that are not in "fixed_dim"
  af::array ind_not_fixed = constant(true, D, b8); ind_not_fixed(ind_fixed) = false;
  af::array af_selected_model_intervals = af_model_intervals(ind_not_fixed, span, ind_in_range).copy();
  
  // // fill new model
  // af::array seq_D(dim4(D), u32); seq_D(span) = af::seq(0, D-1);
  // af::array find_non_fixed = seq_D(ind_not_fixed).copy();
  // find_non_fixed.host(host_find_non_fixed);
  
  nb_in_range = af_selected_model_intervals.dims(2);
  af::array seq_leaves(dim4(Nleaves), u32); seq_leaves(span) = af::seq(0, Nleaves-1);
  af::array find_in_range = seq_leaves(ind_in_range).copy();
  host_find_in_range = new unsigned int[nb_in_range];
  find_in_range.host(host_find_in_range);
  
  // new_tree->init(nb_in_range);
  // reorder(af_selected_model_intervals, 2, 1, 0).host(new_tree->model_intervals[0][0]);
#else
  
  boost::dynamic_bitset<> temp_intersect;
  temp_intersect.resize(Nleaves);
  temp_intersect.set();
  int ind_fixed_d;
  for (int d=0; d < nb_fixed; d++) {
    ind_fixed_d = fixed_dim[d];
    for (int n=0; n < Nleaves; n++) {
      if (fixed_val[d] < model_intervals[ind_fixed_d][0][n] || fixed_val[d] >= model_intervals[ind_fixed_d][1][n]) {
        temp_intersect[n] = 0;
      }
    }
  }
  
  nb_in_range = temp_intersect.count();
  host_find_in_range = new unsigned int[nb_in_range];
  
  int cnt_inside = 0;
  size_t l = temp_intersect.find_first();
  while (l != string::npos) {
    host_find_in_range[cnt_inside] = l;
    l = temp_intersect.find_next(l);
    cnt_inside++;
  }
  temp_intersect.clear();
  
#endif
  
  printf("nb_in_range = %d\n", nb_in_range);
  new_tree->init(nb_in_range);
  
  int old_n, old_d;
  for (int n=0; n < nb_in_range; n++) {
    old_n = host_find_in_range[n];
    for (int d=0; d < nb_non_fixed; d++) {
      old_d = host_find_non_fixed[d];
      new_tree->model_intervals[d][0][n] = this->model_intervals[old_d][0][old_n];
      new_tree->model_intervals[d][1][n] = this->model_intervals[old_d][1][old_n];
      new_tree->is_inf[d][0][n] = this->is_inf[old_d][0][old_n];
      new_tree->is_inf[d][1][n] = this->is_inf[old_d][1][old_n];
    }
    new_tree->index_class_leaves[n] = this->index_class_leaves[old_n];
    new_tree->score_leaves[n] = this->score_leaves[old_n];
  }
  
  delete[] host_find_non_fixed;
  delete[] host_find_in_range;
  
  printf("%d leaves preselected - %d dimensions remaining\n", nb_in_range, nb_non_fixed);
  
  return new_tree;
}



#ifdef ARRAYFIRE
// data2predict: D x Npredict
// consider predicting batches of leaves if the number of leaves is too high
// predicting batches of data as well
af::array tree_model::predict(const af::array &data2predict, const bool do_softmax, const int batch_size) {
  af::array af_model_intervals = reorder(af::array(dim4(Nleaves, 2, D), model_intervals[0][0]), 2, 1, 0); // D x 2 x Nleaves
  // printDims(af_model_intervals);
  
  int Npredict = data2predict.dims(1);
  af::array test1 = tile(data2predict, 1, 1, Nleaves) >= tile(af_model_intervals(span, 0, span), 1, Npredict, 1);
  af::array test2 = tile(data2predict, 1, 1, Nleaves) < tile(af_model_intervals(span, 1, span), 1, Npredict, 1);
  af::array test12 = test1 & test2; // D x Npredict x Nleaves
  af::array is_inside = (sum(test12*1.f, 0) == (float)D)*1.f; // 1 x Npredict x Nleaves
  // printDims(is_inside);
  
  af::array af_score_leaves(Nleaves, score_leaves.data());
  af::array full_score = constant(0.f, Nclass, 1, Nleaves);
  af::array af_index_class_leaves(Nleaves, index_class_leaves.data());
  full_score(af::array(af::seq(0, Nleaves-1))*Nclass + af_index_class_leaves) = af_score_leaves;
  
  af::array sum_score_leaves = sum(
    tile(full_score, 1, Npredict, 1)*tile(is_inside, Nclass, 1, 1), 
    2
  );
  
  af::array af_pred;
  switch(fun_aggr) {
    case sigmoid_fun: {
      af_pred = af::sigmoid(sum_score_leaves);
      break;
    }
    case id_fun: {
      af_pred = sum_score_leaves;
      break;
    }
    default: {
      break;
    }
  }
  
  if (do_softmax) {
    af::array af_exp_pred = exp(af_pred);
    af_pred = af_exp_pred/tile(sum(af_exp_pred, 0), Nclass);
  }
  
  return af_pred;
}
#endif



float tree_model::check_leaves(float *pt, boost::dynamic_bitset<> &temp_intersect) {
  
  temp_intersect.resize(Nleaves);
  temp_intersect.set();
  for (int d=0; d < D; d++) {
    for (int n=0; n < Nleaves; n++) {
      if (pt[d] < model_intervals[d][0][n] || pt[d] >= model_intervals[d][1][n]) {
        temp_intersect[n] = 0;
      }
    }
  }
  
  size_t l = temp_intersect.find_first();
  float temp_sum = 0.f;
  while (l != string::npos) {
    temp_sum += score_leaves[l];
    l = temp_intersect.find_next(l);
  }
  
  return this->aggr_function(temp_sum);
}



float tree_model::check_leaves(float *pt, const bool verbose) {
  boost::dynamic_bitset<> temp_intersect;
  float res = this->check_leaves(pt, temp_intersect);
  temp_intersect.clear();
  
  if (verbose) {
    printf("Prediction = %f\n", res);
  }
  
  return res;
}



float *tree_model::check_leaves_multi(float *pt, boost::dynamic_bitset<> &temp_intersect, float *buffer_res, const bool do_softmax) {
  
  temp_intersect.resize(Nleaves);
  temp_intersect.set();
  for (int d=0; d < D; d++) {
    for (int n=0; n < Nleaves; n++) {
      if (pt[d] < model_intervals[d][0][n] || pt[d] >= model_intervals[d][1][n]) {
        temp_intersect[n] = 0;
      }
    }
  }
  
  if (!buffer_res) buffer_res = new float[Nclass];
  memset(buffer_res, 0, sizeof(float)*Nclass);
  
  size_t l = temp_intersect.find_first();
  while (l != string::npos) {
    buffer_res[index_class_leaves[l]] += score_leaves[l];
    l = temp_intersect.find_next(l);
  }
  
  if (!do_softmax) {
    for (int c=0; c < Nclass; c++) buffer_res[c] = this->aggr_function(buffer_res[c]);
  } else {
    float temp_sum = 0.f;
    for (int c=0; c < Nclass; c++) {
      buffer_res[c] = exp(this->aggr_function(buffer_res[c]));
      temp_sum += buffer_res[c];
    }
    for (int c=0; c < Nclass; c++) buffer_res[c] /= temp_sum;
  }
  
  return buffer_res;
}



float *tree_model::check_leaves_multi(float *pt, float *buffer_res, const bool do_softmax) {
  boost::dynamic_bitset<> temp_intersect;
  this->check_leaves_multi(pt, temp_intersect, buffer_res, do_softmax);
  temp_intersect.clear();
  return buffer_res;
}


