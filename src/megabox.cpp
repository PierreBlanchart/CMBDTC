/** megabox.cpp
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


#include "megabox.h"
#include "CF_finder.h"


void compute_leaves_single_tree(arma::mat &tree_mat, tree_model *mytree, int index_class, int index_leaves_cur=0) {
  
  int max_depth = mytree->MAX_DEPTH;
  
  // form leaves
  stack<int> ind_explore; ind_explore.push(0); // LIFO stack
  
  int *depth = new int[tree_mat.n_rows]; depth[0] = 0;
  bool *isLeftNode = new bool[tree_mat.n_rows]; memset(isLeftNode, 0, sizeof(bool)*tree_mat.n_rows);
  
  float *split_path = new float[max_depth+1];
  int *dim_path = new int[max_depth+1];
  bool *isLeft_path = new bool[max_depth+1];
  
  int depth_cur, ind_left, ind_right;
  int node_cur, index_leaf = index_leaves_cur;
  while (!ind_explore.empty()) {
    node_cur = ind_explore.top();
    ind_explore.pop();
    depth_cur = depth[node_cur];
    
    isLeft_path[depth_cur] = isLeftNode[node_cur];
    
    // printf("Exploring node %d at depth %d [%d, %d]\n", node_cur, depth_cur, (int)tree_mat(node_cur, 0), (int)tree_mat(node_cur, 1));
    
    if (tree_mat(node_cur, 0) < 0) { // we arrived in a leaf
      for (int d=0; d < depth_cur; d++) {
        // printf("dim_path[%d] = %d | index_leaf = %d\n", d, dim_path[d], index_leaf);
        if (isLeft_path[d+1]) {
          if (split_path[d] < mytree->model_intervals[dim_path[d]][1][index_leaf]) {
            mytree->model_intervals[dim_path[d]][1][index_leaf] = split_path[d];
            mytree->is_inf[dim_path[d]][1][index_leaf] = false;
          }
        } else {
          if (split_path[d] > mytree->model_intervals[dim_path[d]][0][index_leaf]) {
            mytree->model_intervals[dim_path[d]][0][index_leaf] = split_path[d];
            mytree->is_inf[dim_path[d]][0][index_leaf] = false;
          }
        }
      }
      mytree->index_class_leaves[index_leaf] = index_class;
      mytree->score_leaves[index_leaf] = tree_mat(node_cur, 4);
      index_leaf++;
    } else {
      // current path updating
      dim_path[depth_cur] = tree_mat(node_cur, 5) - 1;
      split_path[depth_cur] = tree_mat(node_cur, 3);
      
      // stacking left and right nodes (LIFO order)
      ind_left = tree_mat(node_cur, 0);
      ind_right = tree_mat(node_cur, 1);
      
      ind_explore.push(ind_right);
      ind_explore.push(ind_left);
      
      isLeftNode[ind_left] = true;
      depth[ind_left] = depth[ind_right] = depth_cur+1;
    }
  }
  
  delete[] depth;
  delete[] isLeftNode;
  
  delete[] split_path;
  delete[] dim_path;
  delete[] isLeft_path;
  
}



// [[Rcpp::export]]
Rcpp::List compute_all(Rcpp::List &tree_list, int fs, int max_depth, int nb_class, int nb_trees_per_class, float thresh_dec) {
  
  tree_model *mytree = new tree_model(fs, max_depth, nb_class, thresh_dec);
  
  int Ntrees = tree_list.size();
  int max_nodes = 0;
  for (int t=0; t < Ntrees; t++) {
    max_nodes = std::max(max_nodes, Rcpp::as<NumericMatrix>(tree_list[t]).nrow());
  }
  printf("Model contains %d trees with at most %d nodes\n", Ntrees, max_nodes);
  
  // compute number of leaves per tree
  vector<int> nb_leaves(Ntrees), cum_nb_leaves(Ntrees);
  
  int *ind2explore = new int[max_nodes];
  int cursor;
  int node_cur, index_leaf;
  int nb_leaves_tot = 0;
  for (int t=0; t < Ntrees; t++) {
    // printf("Treating tree %d\n", t);
    arma::mat tree_mat = Rcpp::as<arma::mat>(tree_list[t]);
    // printf("%d |%d | %d\n", tree_mat.n_rows, tree_mat.n_cols, (int)tree_mat(0, 0));
    
    // count number of leaves
    cursor = index_leaf = 0;
    ind2explore[cursor] = 0;
    while (cursor >= 0) {
      node_cur = ind2explore[cursor];
      // printf("%d --> %d\n", cursor, node_cur);
      if (tree_mat(node_cur, 0) < 0) { // we arrived in a leaf
        index_leaf++;
        cursor--;
      } else {
        ind2explore[cursor] = tree_mat(node_cur, 0);
        cursor++;
        ind2explore[cursor] = tree_mat(node_cur, 1);
      }
    }
    nb_leaves[t] = index_leaf;
    
    // change instructions below if parallelizing
    cum_nb_leaves[t] = nb_leaves_tot;
    nb_leaves_tot += nb_leaves[t];
    
    // printf("Tree %d has %d leaves\n", t, nb_leaves[t]);
  }
  
  delete[] ind2explore;
  
  // extract all leaves from all trees
  mytree->init(nb_leaves_tot);
  if (nb_class > 2) {
    for (int t=0; t < Ntrees; t++) {
      // printf("Treating tree %d\n", t);
      arma::mat tree_mat = Rcpp::as<arma::mat>(tree_list[t]);
      compute_leaves_single_tree(tree_mat, mytree, t%nb_class, cum_nb_leaves[t]);
    }
  } else {
    for (int t=0; t < Ntrees; t++) {
      // printf("Treating tree %d\n", t);
      arma::mat tree_mat = Rcpp::as<arma::mat>(tree_list[t]);
      compute_leaves_single_tree(tree_mat, mytree, 0, cum_nb_leaves[t]);
    }
  }
  
  // format model dump
  arma::mat arma_leaves(fs*2, nb_leaves_tot);
  arma::vec arma_labels(nb_leaves_tot);
  arma::vec arma_scores(nb_leaves_tot);
  for (int i=0; i < nb_leaves_tot; i++) {
    for (int d=0; d < fs; d++) {
      arma_leaves(d, i) = mytree->model_intervals[d][0][i];
      arma_leaves(fs+d, i) = mytree->model_intervals[d][1][i];
    }
    arma_labels(i) = mytree->index_class_leaves[i];
    arma_scores(i) = mytree->score_leaves[i];
  }
  
  Rcpp::List res_leaves = List::create(_["leaves"] = arma_leaves, _["labels"] = arma_labels, _["scores"] = arma_scores);
  
  return res_leaves;
  
}



tree_model * dump_leaves(Rcpp::List &tree_list, int fs, int max_depth, int nb_class, int nb_trees_per_class, float thresh_dec, float *target_intervals) {
  
  tree_model *mytree;
  if (nb_class > 1) {
    mytree = new tree_model(fs, max_depth, nb_class, thresh_dec);
  } else {
    if (!target_intervals) printf("Target interval should be provided at some point for regression: unspecified for now\n");
    mytree = new tree_model(fs, max_depth, target_intervals);
  }
  
  int Ntrees = tree_list.size();
  int max_nodes = 0;
  for (int t=0; t < Ntrees; t++) {
    max_nodes = std::max(max_nodes, Rcpp::as<NumericMatrix>(tree_list[t]).nrow());
  }
  printf("Model contains %d trees with at most %d nodes\n", Ntrees, max_nodes);
  
  // compute number of leaves per tree
  vector<int> nb_leaves(Ntrees), cum_nb_leaves(Ntrees);
  
  int *ind2explore = new int[max_nodes];
  int cursor;
  int node_cur, index_leaf;
  int nb_leaves_tot = 0;
  for (int t=0; t < Ntrees; t++) {
    // printf("Treating tree %d\n", t);
    arma::mat tree_mat = Rcpp::as<arma::mat>(tree_list[t]);
    // printf("%d |%d | %d\n", tree_mat.n_rows, tree_mat.n_cols, (int)tree_mat(0, 0));
    
    // count number of leaves
    cursor = index_leaf = 0;
    ind2explore[cursor] = 0;
    while (cursor >= 0) {
      node_cur = ind2explore[cursor];
      // printf("%d --> %d\n", cursor, node_cur);
      if (tree_mat(node_cur, 0) < 0) { // we arrived in a leaf
        index_leaf++;
        cursor--;
      } else {
        ind2explore[cursor] = tree_mat(node_cur, 0);
        cursor++;
        ind2explore[cursor] = tree_mat(node_cur, 1);
      }
    }
    nb_leaves[t] = index_leaf;
    
    // change instructions below if parallelizing
    cum_nb_leaves[t] = nb_leaves_tot;
    nb_leaves_tot += nb_leaves[t];
    
    // printf("Tree %d has %d leaves\n", t, nb_leaves[t]);
  }
  
  delete[] ind2explore;
  
  mytree->Nleaves = nb_leaves_tot;
  
  // extract all leaves from all trees
  mytree->init(nb_leaves_tot);
  if (nb_class > 2) {
    for (int t=0; t < Ntrees; t++) {
      // printf("Treating tree %d\n", t);
      arma::mat tree_mat = Rcpp::as<arma::mat>(tree_list[t]);
      compute_leaves_single_tree(tree_mat, mytree, t%nb_class, cum_nb_leaves[t]);
    }
  } else {
    for (int t=0; t < Ntrees; t++) {
      // printf("Treating tree %d\n", t);
      arma::mat tree_mat = Rcpp::as<arma::mat>(tree_list[t]);
      compute_leaves_single_tree(tree_mat, mytree, 0, cum_nb_leaves[t]);
    }
  }
  
  return mytree;
  
}



// [[Rcpp::export]]
arma::mat predict_model(const arma::mat &data2predict, Rcpp::List &tree_list, int max_depth, int nb_class, 
                        int nb_trees_per_class, float thresh_dec, const std::string model_type) {
  int fs = data2predict.n_rows;
  tree_model *mytree = dump_leaves(tree_list, fs, max_depth, nb_class, nb_trees_per_class, thresh_dec);
  
  switch(map_model[model_type]) {
  case reg_logistic: {
    mytree->fun_aggr = sigmoid_fun;
    break;
  }
  case multi_softmax: {
    mytree->fun_aggr = id_fun;
    break;
  }
  case reg_squarederror: {
    mytree->fun_aggr = id_fun;
    break;
  }
  default: {
    fprintf(stderr, "Unknown model type\n");
    return 0;
  }
  }
  
#ifdef ARRAYFIRE
  // printf("Converting to af format\n");
  af::array af_data2predict = arma_mat2af(data2predict);
  printDims(af_data2predict);
  
  af::array af_pred = mytree->predict(af_data2predict, mytree->ttype==multi_classif);
  return af2arma_mat(af_pred);
#else
  int nb2predict = data2predict.n_cols;
  arma::fmat fdata2predict = arma::conv_to<arma::fmat>::from(data2predict);
  float *ptr_fdata2predict = fdata2predict.memptr();
  
  float *temp_res;
  arma::fmat fres;
  if (mytree->ttype != multi_classif) { // binary classif or regression
    temp_res = new float[nb2predict];
    for (int n=0; n < nb2predict; n++) temp_res[n] = mytree->check_leaves(&ptr_fdata2predict[n*fs]);
    fres = arma::fmat(temp_res, 1, nb2predict, true, true);
  } else {
    temp_res = new float[nb_class*nb2predict];
    for (int n=0; n < nb2predict; n++) mytree->check_leaves_multi(&ptr_fdata2predict[n*fs], &temp_res[n*nb_class], true);
    fres = arma::fmat(temp_res, nb_class, nb2predict, true, true);
  }
  delete[] temp_res;
  
  arma::mat res = arma::conv_to<arma::mat>::from(fres);
  return res;
  
#endif
}



// [[Rcpp::export]]
Rcpp::List CF_find(arma::vec &query, int predicted_class, int target_class, Rcpp::List &tree_list, int max_depth, int nb_class, int nb_trees_per_class,
                   float thresh_dec, float sup_d2query_dataset, int budget, int max_dim_width_first, 
                   bool check_has_target, bool update_sup_bound, const std::string model_type) {
  
  int fs = query.size();
  tree_model *mytree = dump_leaves(tree_list, fs, max_depth, nb_class, nb_trees_per_class, thresh_dec);
  printf("Dumped model to %d %d-dimensional intervals\n", mytree->Nleaves, fs);
  // printVec(mytree->score_leaves.data(), mytree->Nleaves);
  
  switch(map_model[model_type]) {
  case reg_logistic:  {
    mytree->fun_aggr = sigmoid_fun;
    break;
  }
  case multi_softmax: {
    mytree->fun_aggr = id_fun;
    break;
  }
  case reg_squarederror: {
    mytree->fun_aggr = id_fun;
    break;
  }
  default: {
    fprintf(stderr, "Unknown model type\n");
    return 0;
  }
  }
  
  float *fquery = new float[fs];
  for (int d=0; d < fs; d++) fquery[d] = (float)query(d);
  
  int target_class_;
  if (nb_class == 2) {
    target_class_ = (predicted_class==1) ? 0 : 1;
  } else {
    target_class_ = target_class;
  }
  
  CF_finder *CF_obj = new CF_finder(mytree, fquery, target_class_, sup_d2query_dataset, budget, check_has_target, update_sup_bound);
  CF_obj->start_width_first(max_dim_width_first);
  // delete CF_obj; return List::create(_["found_solution"] = false);
  CF_obj->continue_depth_first(budget);
  
  delete[] fquery;
  
  // format result
  arma::vec res(fs);
  float prediction_CF = NAN;
  if (CF_obj->found_solution) {
    for (int d=0; d < fs; d++) res(d) = CF_obj->target[d];
    if (mytree->ttype != multi_classif) {
      prediction_CF = CF_obj->subtree->check_leaves(CF_obj->target);
      printf("Prediction CF example = %f\n", prediction_CF);
    } else {
      
    }
  } else {
    for (int d=0; d < fs; d++) res(d) = NAN;
    prediction_CF = NAN;
  }
  
  // format CF sets
  int cnt_CF = 0;
  for (int t=0; t < CF_obj->nb_threads; t++) cnt_CF += CF_obj->CF_intervals[t].size();
  
  arma::cube CF_interv(fs, 2, cnt_CF);
  arma::mat CF_in_interv(fs, cnt_CF);
  arma::vec d2CF(cnt_CF);
  for (int t=0, ind = 0; t < CF_obj->nb_threads; t++) {
    for (int l=0; l < CF_obj->CF_intervals[t].size(); l++, ind++) {
      for (int d=0; d < fs; d++) {
        CF_interv(d, 0, ind) = CF_obj->CF_intervals[t][l][0][d];
        CF_interv(d, 1, ind) = CF_obj->CF_intervals[t][l][1][d];
        CF_in_interv(d, ind) = CF_obj->CF_in_intervals[t][l][d];
      }
      d2CF(ind) = CF_obj->d2CF[t][l];
    }
  }
  
  List list_res = List::create(_["CF_example"] = res, 
                               _["found_solution"] = CF_obj->found_solution, 
                               _["is_approx"] = CF_obj->is_approx,
                               _["dist_CF2query"] = CF_obj->dist_CF2query, 
                               _["prediction"] = prediction_CF, 
                               _["CF_sets"] = CF_interv,
                               _["CF_in_sets"] = CF_in_interv,
                               _["d2CF_sets"] = d2CF
  );
  
  delete CF_obj;
  
  return list_res;
}



// [[Rcpp::export]]
Rcpp::List CF_find_with_mask(arma::vec &query, int predicted_class, int target_class, arma::vec &mask_fixed_features, 
                             Rcpp::List &tree_list, int max_depth, int nb_class, int nb_trees_per_class,
                             float thresh_dec, float masked_sup_d2query_dataset, int budget, int max_dim_width_first, 
                             bool check_has_target, bool update_sup_bound, const std::string model_type) {
  
  int fs = query.size();
  tree_model *mytree = dump_leaves(tree_list, fs, max_depth, nb_class, nb_trees_per_class, thresh_dec);
  printf("Dumped model to %d %d-dimensional intervals\n", mytree->Nleaves, fs);
  // printVec(mytree->score_leaves.data(), mytree->Nleaves);
  
  switch(map_model[model_type]) {
  case reg_logistic:  {
    mytree->fun_aggr = sigmoid_fun;
    break;
  }
  case multi_softmax: {
    mytree->fun_aggr = id_fun;
    break;
  }
  case reg_squarederror: {
    mytree->fun_aggr = id_fun;
    break;
  }
  default: {
    fprintf(stderr, "Unknown model type\n");
    return 0;
  }
  }
  
  int target_class_;
  if (nb_class == 2) {
    target_class_ = (predicted_class==1) ? 0 : 1;
  } else {
    target_class_ = target_class;
  }
  
  int nb_fixed = sum(mask_fixed_features);
  if (nb_fixed == fs) {
    List list_res = List::create(_["CF_example"] = query, _["found_solution"] = true, _["is_approx"] = false, _["dist_CF2query"] = 0.f);
    return list_res;
  }
  
  printf("%d fixed features\n", nb_fixed);
  tree_model *masked_tree;
  float *masked_query = new float[fs-nb_fixed];
  if (nb_fixed) {
    unsigned int *ind_fixed = new unsigned int[nb_fixed];
    float *val_fixed = new float[nb_fixed];
    int ind_cur = 0, ind_cur2 = 0;
    for (int i=0; i < fs; i++) {
      if (mask_fixed_features(i)) {
        ind_fixed[ind_cur] = i;
        val_fixed[ind_cur] = query(i);
        ind_cur++;
      } else {
        masked_query[ind_cur2] = query(i);
        ind_cur2++;
      }
    }
    masked_tree = mytree->preselect_with_fixed_dimensions(val_fixed, ind_fixed, nb_fixed);
    
    delete[] ind_fixed;
    delete[] val_fixed;
  } else {
    for (int i=0; i < fs; i++) masked_query[i] = query(i);
    masked_tree = mytree;
  }
  
  CF_finder *CF_obj = new CF_finder(masked_tree, masked_query, target_class_, masked_sup_d2query_dataset, budget, check_has_target, update_sup_bound);
  CF_obj->start_width_first(max_dim_width_first);
  CF_obj->continue_depth_first(budget);
  
  delete[] masked_query;
  
  // format result
  arma::vec res(fs);
  float prediction_CF;
  if (CF_obj->found_solution) {
    int ind_cur = 0;
    for (int d=0; d < fs; d++) {
      if (!mask_fixed_features(d)) {
        res(d) = CF_obj->target[ind_cur];
        ind_cur++;
      } else {
        res(d) = query(d);
      }
    }
    prediction_CF = CF_obj->subtree->check_leaves(CF_obj->target);
    printf("Prediction CF example = %f\n", prediction_CF);
  } else {
    for (int d=0; d < fs; d++) res(d) = NAN;
    prediction_CF = NAN;
  }
  
  List list_res = List::create(_["CF_example"] = res, 
                               _["found_solution"] = CF_obj->found_solution, 
                               _["is_approx"] = CF_obj->is_approx,
                               _["dist_CF2query"] = CF_obj->dist_CF2query,
                               _["prediction"] = prediction_CF
  );
  
  delete CF_obj;
  
  return list_res;
}


// [[Rcpp::export]]
Rcpp::List  CF_find_regression(arma::vec &query, arma::vec &target_interval, Rcpp::List &tree_list, int max_depth, int nb_trees, 
                               float sup_d2query_dataset, int budget, int max_dim_width_first, 
                               bool check_has_target, bool update_sup_bound, 
                               const std::string model_type) {
  
  int fs = query.size();
  float *ftarget_interval = new float[2];
  ftarget_interval[0] = target_interval(0); ftarget_interval[1] = target_interval(1);
  tree_model *mytree = dump_leaves(tree_list, fs, max_depth, 1, nb_trees, -1.f, ftarget_interval);
  printf("Dumped model to %d %d-dimensional intervals\n", mytree->Nleaves, fs);
  
  switch(map_model[model_type]) {
  case reg_logistic:  {
    mytree->fun_aggr = sigmoid_fun;
    break;
  }
  case multi_softmax: {
    mytree->fun_aggr = id_fun;
    break;
  }
  case reg_squarederror: {
    mytree->fun_aggr = id_fun;
    break;
  }
  default: {
    fprintf(stderr, "Unknown model type\n");
    return 0;
  }
  }
  
  float *fquery = new float[fs];
  for (int d=0; d < fs; d++) fquery[d] = (float)query(d);
  
  CF_finder *CF_obj = new CF_finder(mytree, fquery, ftarget_interval, sup_d2query_dataset, budget, check_has_target, update_sup_bound);
  CF_obj->start_width_first_regression(max_dim_width_first);
  CF_obj->continue_depth_first_regression(budget);
  
  delete[] fquery;
  
  // format result
  arma::vec res(fs);
  float prediction_CF;
  if (CF_obj->found_solution) {
    for (int d=0; d < fs; d++) res(d) = CF_obj->target[d];
    prediction_CF = CF_obj->subtree->check_leaves(CF_obj->target);
    printf("Prediction CF example = %f\n", prediction_CF);
  } else {
    for (int d=0; d < fs; d++) res(d) = NAN;
    prediction_CF = NAN;
  }
  
  // format CF sets
  printf("Formatting CF sets\n");
  int cnt_CF = 0;
  for (int t=0; t < CF_obj->nb_threads; t++) cnt_CF += CF_obj->CF_intervals[t].size();
  
  arma::cube CF_interv(fs, 2, cnt_CF);
  arma::mat CF_in_interv(fs, cnt_CF);
  arma::vec d2CF(cnt_CF);
  for (int t=0, ind = 0; t < CF_obj->nb_threads; t++) {
    for (int l=0; l < CF_obj->CF_intervals[t].size(); l++, ind++) {
      for (int d=0; d < fs; d++) {
        CF_interv(d, 0, ind) = CF_obj->CF_intervals[t][l][0][d];
        CF_interv(d, 1, ind) = CF_obj->CF_intervals[t][l][1][d];
        CF_in_interv(d, ind) = CF_obj->CF_in_intervals[t][l][d];
      }
      d2CF(ind) = CF_obj->d2CF[t][l];
    }
  }
  
  List list_res = List::create(_["CF_example"] = res, 
                               _["found_solution"] = CF_obj->found_solution, 
                               _["is_approx"] = CF_obj->is_approx,
                               _["dist_CF2query"] = CF_obj->dist_CF2query,
                               _["prediction"] = prediction_CF, 
                               _["CF_sets"] = CF_interv,
                               _["CF_in_sets"] = CF_in_interv,
                               _["d2CF_sets"] = d2CF
  );
  
  delete CF_obj;
  
  return list_res;
}



// [[Rcpp::export]]
Rcpp::List CF_find_with_mask_regression(arma::vec &query, arma::vec &target_interval, arma::vec &mask_fixed_features, Rcpp::List &tree_list, int max_depth, int nb_trees,
                                        float masked_sup_d2query_dataset, int budget, int max_dim_width_first, bool check_has_target, 
                                        bool update_sup_bound, const std::string model_type) {
  
  int fs = query.size();
  float *ftarget_interval = new float[2];
  ftarget_interval[0] = target_interval(0); ftarget_interval[1] = target_interval(1);
  tree_model *mytree = dump_leaves(tree_list, fs, max_depth, 1, nb_trees, -1.f, ftarget_interval);
  printf("Dumped model to %d %d-dimensional intervals\n", mytree->Nleaves, fs);
  
  switch(map_model[model_type]) {
  case reg_logistic:  {
    mytree->fun_aggr = sigmoid_fun;
    break;
  }
  case multi_softmax: {
    mytree->fun_aggr = id_fun;
    break;
  }
  case reg_squarederror: {
    mytree->fun_aggr = id_fun;
    break;
  }
  default: {
    fprintf(stderr, "Unknown model type\n");
    return 0;
  }
  }
  
  int nb_fixed = sum(mask_fixed_features);
  if (nb_fixed == fs) {
    List list_res = List::create(_["CF_example"] = query, _["found_solution"] = true, _["is_approx"] = false, _["dist_CF2query"] = 0.f);
    return list_res;
  }
  
  printf("%d fixed features\n", nb_fixed);
  unsigned int *ind_fixed = new unsigned int[nb_fixed];
  float *val_fixed = new float[nb_fixed];
  float *masked_query = new float[fs-nb_fixed];
  int ind_cur = 0, ind_cur2 = 0;
  for (int i=0; i < fs; i++) {
    if (mask_fixed_features(i)) {
      ind_fixed[ind_cur] = i;
      val_fixed[ind_cur] = query(i);
      ind_cur++;
    } else {
      masked_query[ind_cur2] = query(i);
      ind_cur2++;
    }
  }
  
  tree_model *masked_tree = mytree->preselect_with_fixed_dimensions(val_fixed, ind_fixed, nb_fixed);
  masked_tree->check_leaves(masked_query, true);
  // masked_tree->print_info();
  // return List::create(_["found_solution"] = false);
  
  CF_finder *CF_obj = new CF_finder(masked_tree, masked_query, ftarget_interval, masked_sup_d2query_dataset, budget, check_has_target, update_sup_bound);
  CF_obj->start_width_first_regression(max_dim_width_first);
  CF_obj->continue_depth_first_regression(budget);
  
  delete[] ind_fixed;
  delete[] val_fixed;
  delete[] masked_query;
  
  // format result
  arma::vec res(fs);
  float prediction_CF;
  if (CF_obj->found_solution) {
    ind_cur = 0;
    for (int d=0; d < fs; d++) {
      if (!mask_fixed_features(d)) {
        res(d) = CF_obj->target[ind_cur];
        ind_cur++;
      } else {
        res(d) = query(d);
      }
    }
    prediction_CF = CF_obj->subtree->check_leaves(CF_obj->target);
    printf("Prediction CF example = %f\n", prediction_CF);
  } else {
    for (int d=0; d < fs; d++) res(d) = NAN;
    prediction_CF = NAN;
  }
  
  List list_res = List::create(_["CF_example"] = res, 
                               _["found_solution"] = CF_obj->found_solution, 
                               _["is_approx"] = CF_obj->is_approx,
                               _["dist_CF2query"] = CF_obj->dist_CF2query,
                               _["prediction"] = prediction_CF
  );
  
  delete CF_obj;
  
  return list_res;
}



// [[Rcpp::export]]
Rcpp::List  CF_find_surrogate(const arma::vec &query, const float target, Rcpp::List &tree_list, int max_depth, int nb_class, 
                              int nb_trees_per_class, float thresh_dec, float sigma, int Niter, float lr, float lambda_distortion,
                              const std::string model_type, const std::string optim_crit) {
  
  int fs = query.size();
  tree_model *mytree = dump_leaves(tree_list, fs, max_depth, nb_class, nb_trees_per_class, thresh_dec);
  
  switch(map_model[model_type]) {
  case reg_logistic: {
    mytree->fun_aggr = sigmoid_fun;
    break;
  }
  case multi_softmax: {
    mytree->fun_aggr = id_fun;
    break;
  }
  case reg_squarederror: {
    mytree->fun_aggr = id_fun;
    break;
  }
  default: {
    fprintf(stderr, "Unknown model type\n");
    return 0;
  }
  }
  
  float *fquery = new float[fs];
  for (int d=0; d < fs; d++) fquery[d] = (float)query(d);
  
  surrogate *Smodel = new surrogate(*mytree, map_crit[optim_crit]);
  Smodel->sigma = sigma;
  
  // Smodel->forward(fquery);
  Smodel->alpha = lr;
  Smodel->lambda_distortion = lambda_distortion;
  Smodel->compute_CF_binary(fquery, target, Niter);
  
  arma::vec res(fs);
  if (Smodel->found_solution) {
    for (int d=0; d < fs; d++) {
      res(d) = Smodel->CF_example[d];
    }
  }
  
  List list_res = List::create(_["CF_example"] = res, 
                               _["found_solution"] = Smodel->found_solution, 
                               _["dist_CF2query"] = Smodel->dist_CF2query,
                               _["prediction"] = Smodel->prediction[0]
  );
  
  delete[] fquery;
  delete Smodel;
  
  return list_res;
}


