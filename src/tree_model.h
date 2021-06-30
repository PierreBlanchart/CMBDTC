/** tree_model.h
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


#ifndef DEF_TREE_MODEL_H
#define DEF_TREE_MODEL_H

#include "utils.h"


class tree_model {
public:
  int D; // feature space size
  int Nclass;
  float ***model_intervals=NULL; // model leaves: D x 2 x Nleaves
  bool ***is_inf=NULL;
  
  task_type ttype=binary_classif;
  aggr_fun fun_aggr=sigmoid_fun;
  
  float *eps_per_dim = NULL;

  float **sorted_intervals=NULL;
  unsigned int **pos_in_index=NULL; // position of each element in the sorted structure

  vector<int> index_class_leaves; // class index of each leaf
  vector<float> score_leaves; // score associated with each leaf

  int MAX_DEPTH;
  int Nleaves;

  float thresh_dec=0.5f;
  float thresh_sum=0;
  float *target_interval=NULL;

public:
  tree_model(int D_, int MAX_DEPTH_, int Nclass_, float thresh_dec_=0.5f, aggr_fun fun_aggr_=sigmoid_fun) {
    printf("Building classification tree model\n");
    D = D_;
    MAX_DEPTH = MAX_DEPTH_;
    Nclass = Nclass_;
    if (Nclass > 2) {
      ttype = multi_classif;
    } else {
      ttype = binary_classif;
    }
    
    thresh_dec = thresh_dec_;
    if (thresh_dec != 0.5f) { // useful for binary classification
      thresh_sum = std::log(thresh_dec/(1.f-thresh_dec));
      printf("thresh_sum is %f\n", thresh_sum);
    } else {
      thresh_sum = thresh_dec; // id_fun as aggregation function in a binary classification scenario ? in case, it exists ...
    }
    
    fun_aggr=fun_aggr_;
  };
  
  // regression tree
  tree_model(int D_, int MAX_DEPTH_, float *target_interval_, aggr_fun fun_aggr_=id_fun) {
    printf("Building regression tree model\n");
    D = D_;
    MAX_DEPTH = MAX_DEPTH_;
    Nclass = 1;
    ttype = regression_task;
    target_interval = target_interval_;
    fun_aggr=fun_aggr_;
  };

  ~tree_model(void) {
    if (model_intervals) deleteTensor<float>(D, model_intervals);
    if (is_inf) deleteTensor<bool>(D, is_inf);
    
    if (sorted_intervals) deleteMatrix<float>(sorted_intervals);
    if (pos_in_index) deleteMatrix<unsigned int>(pos_in_index);
    
    index_class_leaves.clear();
    score_leaves.clear();
  };
  
  void compute_eps() {
    eps_per_dim = new float[D];
    float max_d;
    bool init_d;
    int nb_signif_d;
    for (int d=0; d < D; d++) {
      init_d = false;
      for (int n=0; n < Nleaves; n++) {
        if (!is_inf[d][0][n]) {
          if (!init_d) {
            max_d = abs(model_intervals[d][0][n]);
            init_d = true;
          } else {
            max_d = max(max_d, abs(model_intervals[d][0][n]));
          }
        }
      }
      nb_signif_d = floor(log10(max_d))+1;
      eps_per_dim[d] = std::pow(10.0, nb_signif_d-7);
    }
  };

  void init(int Nleaves_);
  
  void presort();

  tree_model* select_boxes(float *query, float thresh_dsq);
  
  float aggr_function(const float &val) {
    switch(fun_aggr) {
      case sigmoid_fun: {
        return sigmoid(val);
      }
      case id_fun: {
        return val;
      }
      default: {
        return val;
      }
    }
  }
  
  tree_model *preselect_with_fixed_dimensions(float *fixed_val, unsigned int *fixed_dim, int nb_fixed);
  
#ifdef ARRAYFIRE
  af::array predict(const af::array &data2predict, const bool do_softmax=false, const int batch_size=128);
#endif
  
  // debug
  float check_leaves(float *pt, const bool verbose=false);
  float check_leaves(float *pt, boost::dynamic_bitset<> &temp_intersect);
  float *check_leaves_multi(float *pt, float *buffer_res=NULL, const bool do_softmax=false);
  float *check_leaves_multi(float *pt, boost::dynamic_bitset<> &temp_intersect, float *buffer_res=NULL, const bool do_softmax=false);
  
  void print_info() {
    printf("Nleaves = %d | fun_aggr = %d\n", Nleaves, (int)fun_aggr);
  };
};

#endif
