/** surrogate.h
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


#ifndef DEF_SURROGATE_H
#define DEF_SURROGATE_H

#include "tree_model.h"


#ifdef ARRAYFIRE
inline af::array sigmoid2(const af::array &val, float sigma) {
  return 1.f/(1.f + exp(-sigma*val));
}
#endif


class surrogate: public tree_model {
public:
  float sigma=1.f;
  float lambda_distortion = 1.f; //1.f;
  
  float alpha = 1e-3f;
  float lrd=1e-7f, eps=1e-8f, beta1=0.9f, beta2=0.999f;
  
  crit_type crit_optim=mse;
  
public:
  surrogate(tree_model &mytree, crit_type crit_optim_=mse) : tree_model(mytree) {
    crit_optim = crit_optim_;
    
#ifdef ARRAYFIRE
    af_model_intervals = reorder(af::array(dim4(Nleaves, 2, D), model_intervals[0][0]), 2, 1, 0); // D x 2 x Nleaves
    
    af::array af_score_leaves(Nleaves, score_leaves.data());
    if (Nclass > 2) {
      full_score = constant(0.f, Nclass, 1, Nleaves);
      af::array af_index_class_leaves(Nleaves, index_class_leaves.data());
      full_score(af::array(af::seq(0, Nleaves-1))*Nclass + af_index_class_leaves) = af_score_leaves; // Nclass x 1 x Nleaves
    } else {
      full_score = moddims(af_score_leaves, 1, 1, Nleaves);
    }
#else
    is_inside_left = makeMatrix<float>(D, Nleaves);
    is_inside_right = makeMatrix<float>(D, Nleaves);
    is_inside = new float[Nleaves];
    prod_is_inside = new float[Nleaves];
    
    if (ttype == multi_classif) {
      temp_sum_score_leaves = new float[Nclass];
      prediction = new float[Nclass];
      bm = new float[Nclass];
      pred_multi = new float[Nclass];
    } else {
      prediction = new float[1];
      bm = new float[1];
    }
    
    bm_data = new float[D];
#endif
  };
  
  ~surrogate() {
#ifdef ARRAYFIRE
    af_device_gc();
#else
    deleteMatrix<float>(is_inside_left);
    deleteMatrix<float>(is_inside_right);
    delete[] is_inside;
    delete[] prod_is_inside;
    delete[] bm_data;
    delete[] CF_example;
    delete[] prediction;
    delete[] bm;
    if (temp_sum_score_leaves) delete[] temp_sum_score_leaves;
    if (pred_multi) delete[] pred_multi;
#endif
  }
  
#ifdef ARRAYFIRE
  int Npredict;
  af::array af_model_intervals, full_score;
  af::array is_inside_left, is_inside_right, is_inside_scored, prediction;
  void forward(const af::array &data2predict);
  
  af::array bm_data;
  void backward_CF(const af::array &bm);
  
  void compute_CF_binary(const af::array &query, const float target, const int Niter=48);
#else
  float **is_inside_left=NULL, **is_inside_right=NULL, *is_inside=NULL, *prod_is_inside=NULL;
  float *prediction=NULL, *temp_sum_score_leaves=NULL;
  void forward(float *pt);
  
  float *bm=NULL, *bm_data=NULL;
  void backward_CF();
  
  bool found_solution=false;
  float *pred_multi=NULL;
  float dist_CF2query, *CF_example=NULL;
  void compute_CF_binary(float *query, const float target_score, const int Niter=48);
#endif
};



#endif
