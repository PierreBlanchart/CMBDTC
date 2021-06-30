/** surrogate.cpp
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


#include "surrogate.h"


#ifdef ARRAYFIRE
// data2predict: D x Npredict
void surrogate::forward(const af::array &data2predict) {
  
  Npredict = data2predict.dims(1);
  af::array diff_left = tile(data2predict, 1, 1, Nleaves) - tile(af_model_intervals(span, 0, span), 1, Npredict, 1);
  af::array diff_right = tile(data2predict, 1, 1, Nleaves) - tile(af_model_intervals(span, 1, span), 1, Npredict, 1);
  is_inside_left = sigmoid2(diff_left, sigma);
  is_inside_right = sigmoid2(-diff_right, sigma);
  af::array is_inside = is_inside_left*is_inside_right; // D x Npredict x Nleaves
  af::array prod_is_inside = af::product(is_inside, 0); // 1 x Npredict x Nleaves
  
  if (Nclass == 2) {
    is_inside_scored = tile(full_score, 1, Npredict, 1)*prod_is_inside; // 1 x Npredict x Nleaves
  } else {
    is_inside_scored = tile(full_score, 1, Npredict, 1)*tile(prod_is_inside, Nclass, 1, 1); // Nclass x Npredict x Nleaves
  }
  af::array sum_score_leaves = sum(is_inside_scored, 2); // Nclass x Npredict | 1 x Npredict
  
  switch(fun_aggr) {
  case sigmoid_fun: {
    prediction = af::sigmoid(sum_score_leaves);
    break;
  }
  case id_fun: {
    prediction = sum_score_leaves;
    break;
  }
  default: {
    fprintf(stderr, "Unknow aggregation score type\n");
    break;
  }
  }
  
}
# else
void surrogate::forward(float *pt) {
  float diff_left, diff_right;
  if (ttype != multi_classif) {
    
    float sum_score_leaves = 0.f;
    for (int n=0; n < Nleaves; n++) {
      prod_is_inside[n] = score_leaves[n];
      for (int d=0; d < D; d++) {
        diff_left = pt[d] - model_intervals[d][0][n];
        diff_right = pt[d] - model_intervals[d][1][n];
        is_inside_left[d][n] = sigmoid(diff_left, sigma);
        is_inside_right[d][n] = sigmoid(-diff_right, sigma);
        is_inside[n] = is_inside_left[d][n]*is_inside_right[d][n];
        prod_is_inside[n] *= is_inside[n];
      }
      sum_score_leaves += prod_is_inside[n];
    }
    
    // if (fun_aggr == sigmoid_fun) {
    //   prediction[0] = sigmoid(sum_score_leaves);
    // } else {
    //   prediction[0] = sum_score_leaves;
    // }
    prediction[0] = aggr_function(sum_score_leaves);
    
  } else {
    
    memset(temp_sum_score_leaves, 0, sizeof(float)*Nclass);
    for (int n=0; n < Nleaves; n++) {
      prod_is_inside[n] = score_leaves[n];
      for (int d=0; d < D; d++) {
        diff_left = pt[d] - model_intervals[d][0][n];
        diff_right = pt[d] - model_intervals[d][1][n];
        is_inside_left[d][n] = sigmoid(diff_left, sigma);
        is_inside_right[d][n] = sigmoid(-diff_right, sigma);
        is_inside[n] = is_inside_left[d][n]*is_inside_right[d][n];
        prod_is_inside[n] *= is_inside[n];
      }
      temp_sum_score_leaves[index_class_leaves[n]] += prod_is_inside[n];
    }
    
    // softmax
    float temp_sum = 0.f;
    for (int c=0; c < Nclass; c++) {
      prediction[c] = exp(temp_sum_score_leaves[c]);
      temp_sum += prediction[c];
    }
    for (int c=0; c < Nclass; c++) prediction[c] /= temp_sum;
    
  }
}
#endif



#ifdef ARRAYFIRE
// bm: 1 x Npredict
void surrogate::backward_CF(const af::array &bm) {
  
  af::array temp_bm;
  switch(fun_aggr) {
  case sigmoid_fun: {
    temp_bm = bm*(prediction*(1.f-prediction));
    break;
  }
  case id_fun: {
    temp_bm = bm;
    break;
  }
  default: {
    fprintf(stderr, "Unknow aggregation score type\n");
    break;
  }
  }
  
  // sum(
  //   tile(temp_bm, D, 1, Nleaves) * 
  //     tile(full_score, 1, Npredict, 1) * (tile(prod_is_inside, D) / is_inside) * [is_inside_left*is_inside_right*(1.f-is_inside_right)*(-sigma) +
  //     is_inside_right*is_inside_left*(1.f-is_inside_left)*sigma], 
  //     2
  // );
  
  bm_data = tile(temp_bm, D) * (sigma*sum(tile(is_inside_scored, D)*(is_inside_right - is_inside_left), 2)); // D x Npredict
}
#else
void surrogate::backward_CF() {
  
  float temp_d, prod_n;
  memset(bm_data, 0, sizeof(float)*D);
  
  if (ttype != multi_classif) {
    
    for (int n=0; n < Nleaves; n++) {
      prod_n = prod_is_inside[n] * sigma;
      for (int d=0; d < D; d++) {
        // temp_d = (prod_is_inside[n]/is_inside[n]) * (
        //   (sigma*is_inside_left[d][n]*(1.f-is_inside_left[d][n]) * is_inside_right[d][n]) +
        //     (is_inside_left[d][n] * (-sigma)*is_inside_right[d][n]*(1.f-is_inside_right[d][n]))
        // );
        // temp_d = prod_is_inside[n] * sigma * (1.f-is_inside_left[d][n] - (1.f-is_inside_right[d][n]));
        temp_d = prod_n * (is_inside_right[d][n]-is_inside_left[d][n]);
        bm_data[d] += temp_d;
      }
    }
    
    for (int d=0; d < D; d++) bm_data[d] *= bm[0];
    
  } else {
    
    float bm_n;
    for (int n=0; n < Nleaves; n++) {
      prod_n = prod_is_inside[n] * sigma;
      bm_n = bm[index_class_leaves[n]];
      for (int d=0; d < D; d++) {
        temp_d = prod_n * (is_inside_right[d][n]-is_inside_left[d][n]);
        bm_data[d] += bm_n*temp_d;
      }
    }
    
  }
  
}
#endif



#ifdef ARRAYFIRE
// query: D x 1
void surrogate::compute_CF_binary(const af::array &query, const float target_score, const int Niter) {
  
  int state_t = 0;
  float beta1_t = beta1;
  float beta2_t = beta2;
  af::array m_parameters = constant(0.f, D);
  af::array v_parameters = constant(0.f, D);
  
  float pred_query;
  af::array CF_example = query.copy();
  for (int n=0; n < Niter; n++) {
    this->forward(CF_example);
    prediction.host(&pred_query);
    // if (pred_query <= thresh_dec) break; // we found a CF example
    printf("iter %d: %f [%f]\n", n, pred_query, target_score);
    
    af::array diff_CF = CF_example-query;
    af::array loss_i = lambda_distortion*sum(diff_CF*diff_CF);
    
    af::array update_i = lambda_distortion*diff_CF;
    // if (pred_query > target_score) {
    loss_i += prediction;
    this->backward_CF(prediction-target_score);
    update_i += bm_data;
    // }
    
    // gradient update step
    CF_example -= alpha*update_i;
    
    // adam update step
    // float clr = alpha/(1.f + ((float)state_t)*lrd); // learning rate decay (annealing)
    // state_t++;
    // 
    // float biasCorrection1 = 1.f - beta1_t;
    // float biasCorrection2 = 1.f - beta2_t;
    // float step_size = clr*sqrt(biasCorrection2)/biasCorrection1;
    // float OneMinusBeta1 = 1.f-beta1;
    // float OneMinusBeta2 = 1.f-beta2;
    // 
    // beta1_t *= beta1;
    // beta2_t *= beta2;
    // 
    // m_parameters = (beta1*m_parameters) + (OneMinusBeta1*update_i);
    // v_parameters = (beta2*v_parameters) + (OneMinusBeta2*(update_i*update_i));
    // CF_example -= (step_size*(m_parameters/(sqrt(v_parameters)+eps)));
  }
  
}
#else
void surrogate::compute_CF_binary(float *query, const float target, const int Niter) {
  int state_t = 0;
  float beta1_t = beta1;
  float beta2_t = beta2;
  vector<float> m_parameters(D);
  vector<float> v_parameters(D);
  
  found_solution = false;
  CF_example = new float[D]; memcpy(CF_example, query, sizeof(float)*D);
  dist_CF2query = NAN;
  
  float *update_n = new float[D];
  
  float biasCorrection1, biasCorrection2, step_size;
  float diff_d, loss_n;
  
  int target_int;
  int ind_max;
  float val_max;
  if (ttype == multi_classif) target_int = int(target);
  
  for (int n=0; n < Niter; n++) {
    
    this->forward(CF_example);
    
    switch(ttype) {
      case binary_classif: {
        if (prediction[0] <= thresh_dec) {
          if (this->check_leaves(CF_example) <= thresh_dec) { // be careful that check_leaves(CF_example) != prediction[0] (first is the exact model prediction - second is the surrogate prediction)
            found_solution = true;
            dist_CF2query = std::sqrt(dist_sq(CF_example, query, D));
            printf("Found a CF example (%f) at a distance %.2f\n", prediction[0], dist_CF2query);
            break;
          }
        }
        break;
      }
      case regression_task: {
        
        break;
      }
      case multi_classif: {
        ind_max = 0;
        val_max = prediction[0];
        for (int c=1; c < Nclass; c++) {
          if (prediction[c] > val_max) {
            ind_max = c;
            val_max = prediction[c];
          }
        }
        if (ind_max == target_int) {
          this->check_leaves_multi(CF_example, pred_multi, true);
          ind_max = 0;
          val_max = prediction[0];
          for (int c=1; c < Nclass; c++) {
            if (pred_multi[c] > val_max) {
              ind_max = c;
              val_max = pred_multi[c];
            }
          }
          if (ind_max == target_int) {
            found_solution = true;
            dist_CF2query = std::sqrt(dist_sq(CF_example, query, D));
            printf("Found a CF example (%f) at a distance %.2f\n", prediction[target_int], dist_CF2query);
            break;
          }
        }
        break;
      }
      default: break;
    }
    
    if (found_solution) break;
    
    switch(crit_optim) {
      case mse: {
        bm[0] = prediction[0]-target;
        if (fun_aggr == sigmoid_fun) {
          bm[0] *= prediction[0]*(1.f-prediction[0]);
        }
        break;
      }
      case binary_CE: { // supposing fun_aggr == sigmoid_fun | target = 0 or 1
        bm[0] = target ? prediction[0]-1.f : prediction[0];
        break;
      }
      case CE: { // target in [0, Nclass-1]
        for (int c=0; c < Nclass; c++) {
          if (c!=target_int) bm[c] = prediction[c];
        }
        bm[target_int] = prediction[target_int]-1.f;
        break;
      }
      default: break;
    }
    
    this->backward_CF();
    
    loss_n = 0.f;
    for (int d=0; d < D; d++) {
      diff_d = CF_example[d]-query[d];
      update_n[d] = this->bm_data[d] + lambda_distortion*diff_d;
      loss_n += diff_d*diff_d;
    }
    // loss_n = d2target*d2target + lambda_distortion*loss_n;
    printf("iter %d: prediction = %.3f [%f] -- distortion = %.2f\n", n, prediction[0], target, loss_n);
    
    // gradient update step
    // for (int d=0; d < D; d++) CF_example[d] -= alpha*update_n[d];
    
    // adam update step
    float clr = alpha/(1.f + ((float)state_t)*lrd); // learning rate decay (annealing)
    state_t++;
    
    biasCorrection1 = 1.f - beta1_t;
    biasCorrection2 = 1.f - beta2_t;
    step_size = clr*sqrt(biasCorrection2)/biasCorrection1;
    
    beta1_t *= beta1;
    beta2_t *= beta2;
    
    for (int d=0; d < D; d++) {
      m_parameters[d] = (beta1*m_parameters[d]) + ((1.f-beta1)*update_n[d]);
      v_parameters[d] = (beta2*v_parameters[d]) + ((1.f-beta2)*(update_n[d]*update_n[d]));
      CF_example[d] -= (step_size*(m_parameters[d]/(sqrt(v_parameters[d])+eps)));
    }
    
  }
  
  delete[] update_n;
  
}
#endif

