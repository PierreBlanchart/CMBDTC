/** utils.h
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


#ifndef DEF_UTILS_H
#define DEF_UTILS_H

#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <chrono>
#include <time.h>
#include <limits>
#include <ctype.h>
#include <float.h>
#include <stdarg.h>
#include <vector>
#include <random>
#include <algorithm>
#include <stack>
#include <queue>

#include "boost/dynamic_bitset.hpp"
// #include <boost/sort/spreadsort/spreadsort.hpp>
// using namespace boost::sort::spreadsort;
// #include <boost/compute/algorithm/sort_by_key.hpp>

typedef std::chrono::high_resolution_clock myclock;
typedef myclock::time_point timepoint;
using namespace std::chrono;
using namespace std;

// parallel library "Intel TBB"
#include "tbb/tbb.h"
#include "tbb/task.h"
#include "tbb/parallel_for.h"
#include "tbb/concurrent_vector.h"
#include "tbb/task_scheduler_init.h"
using namespace tbb;

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

#ifdef ARRAYFIRE
#include <arrayfire.h>
using namespace af;
#endif

const float inf = std::numeric_limits<float>::infinity();
const float neg_inf = - std::numeric_limits<float>::infinity();

enum model_type {reg_logistic=0, multi_softmax=1, reg_squarederror=2};
static std::map<std::string, model_type> map_model {
  {"reg:logistic", reg_logistic},
  {"binary:logistic", reg_logistic}, 
  {"multi:softmax", multi_softmax}, 
  {"multi:softprob", multi_softmax}, 
  {"reg:squarederror", reg_squarederror}
};

enum aggr_fun {sigmoid_fun=0, id_fun=1};
enum task_type {binary_classif=0, multi_classif=1, regression_task=2};

enum crit_type {mse=0, binary_CE=1, CE=2};
static std::map<std::string, crit_type> map_crit {
  {"mse", mse},
  {"binary_CE", binary_CE},
  {"CE", CE}
};

template<typename T>
inline T **makeMatrix(int nrows, int ncols, T fill_value=0) {
  int sz = nrows*ncols;
  T *buffer = new T[sz];
  std::fill(buffer, buffer+sz, fill_value);
  
  T **mat = new T*[nrows];
  for (int i=0, ind=0; i < nrows; i++, ind+=ncols) {
    mat[i] = &buffer[ind];
  }
  return mat;
}

template<typename T>
inline T **makeMatrix(T *buffer, int nrows, int ncols) {
  T **mat = new T*[nrows];
  for (int i=0, ind=0; i < nrows; i++, ind+=ncols) {
    mat[i] = &buffer[ind];
  }
  return mat;
}

template<typename T>
inline void deleteMatrix(T **mat) {
  delete[] mat[0];
  delete[] mat;
}

template<typename T>
inline T ***makeTensor(int nLayers, int nrows, int ncols, T fill_value=0) {
  
  int dec = nrows*ncols;
  T *buffer = new T[nLayers*dec]; // continuous storage : take this into account when deleting !
  std::fill(buffer, buffer + nLayers*dec, fill_value);
  
  T ***tensor = new T**[nLayers];
  for (int n=0, ind_n=0; n < nLayers; n++, ind_n+=dec) {
    tensor[n] = makeMatrix<T>(&buffer[ind_n], nrows, ncols);
  }
  
  return tensor;
}

template<typename T>
inline void deleteTensor(int nLayers, T ***tensor) {
  delete[] tensor[0][0];
  for (int n=0; n < nLayers; n++) {
    delete[] tensor[n];
  }
  delete[] tensor;
}


typedef std::pair<int, bool> pair_interv;
typedef std::pair<float, pair_interv> mypairf; // <value, index>
inline bool compFunf(const mypairf& l, const mypairf& r) { return l.first < r.first; }
inline void sortWithIndex2(int nb_values, float *values, vector<mypairf> &obj_sorted) {
  for (int i=0; i < nb_values; i++) obj_sorted[i] = mypairf(values[i], pair_interv(i, true));
  for (int i=nb_values, j=0; i < 2*nb_values; i++, j++) obj_sorted[i] = mypairf(values[i], pair_interv(j, false));
  std::sort(obj_sorted.begin(), obj_sorted.begin() + 2*nb_values, compFunf);
}


inline void sortWithIndex2(int nb_values, float *values, bool *val_inf, vector<mypairf> &obj_sorted,
                           int &nb_inf_start, int &nb_inf_end, int &nb_ninf,
                           int* &ind_start) { //, int* &ind_end) {
  
  nb_inf_start = 0;
  nb_ninf = 0;
  ind_start = new int[nb_values];
  for (int i=0; i < nb_values; i++) {
    if (val_inf[i]) {
      ind_start[nb_inf_start] = i;
      nb_inf_start++;
    } else {
      obj_sorted[nb_ninf] = mypairf(values[i], pair_interv(i, true));
      nb_ninf++;
    }
  }
  // for (int i=0; i < nb_values; i++) obj_sorted[i] = mypairf(values[i], pair_interv(i, true));
  
  nb_inf_end = 0;
  // ind_end = new int[nb_values];
  for (int i=nb_values, j=0; i < 2*nb_values; i++, j++) {
    if (val_inf[i]) {
      // ind_end[nb_inf_end] = j;
      nb_inf_end++;
    } else {
      obj_sorted[nb_ninf] = mypairf(values[i], pair_interv(j, false));
      nb_ninf++;
    }
  }
  // for (int i=nb_values, j=0; i < 2*nb_values; i++, j++) obj_sorted[i] = mypairf(values[i], pair_interv(j, false));
  
  std::sort(obj_sorted.begin(), obj_sorted.begin() + nb_ninf, compFunf);
  
}


inline void printVec(vector<int> &vec, int N=-1) {
  if (N < 0) N = vec.size();
  for (int i=0; i < N-1; i++) printf("%d | ", vec[i]);
  printf("%d\n", vec[N-1]);
}


inline void printVec(float *vec, int N) {
  for (int i=0; i < N-1; i++) printf("%f | ", vec[i]);
  printf("%f\n", vec[N-1]);
}


inline void printVec(double *vec, int N, bool cum=false) {
  if (!cum) {
    for (int i=0; i < N-1; i++) printf("%f | ", vec[i]);
    printf("%f\n", vec[N-1]);
  } else {
    double accum=0.;
    for (int i=0; i < N-1; i++) {
      accum += vec[i];
      printf("%f | ", accum);
    }
    accum += vec[N-1];
    printf("%f\n", accum);
  }
}


inline void printMat(float **mat, int nb1, int nb2) {
  for (size_t i=0; i < nb1; i++) {
    printf("[%d] : ", (int)i);
    for (size_t j=0; j < nb2; j++) {
      printf("%.2f ", mat[i][j]);
    }
    printf("\n");
  }
}


inline float sigmoid(const float &x) {
  return 1.f / (1.f + exp(-x));
}


inline float sigmoid(const float &x, const float &lambda) {
  return 1.f / (1.f + exp(-lambda*x));
}


inline float dist_sq(float *pt1, float *pt2, const int &D) {
  float res = 0.f, tmp_diff;
  for (int d=0; d < D; d++) {
    tmp_diff = pt1[d]-pt2[d];
    res += (tmp_diff*tmp_diff);
  }
  return res;
}


inline float dist_sq(float *pt1, float *pt2, const int &D, const int &rm) {
  float res = 0.f, tmp_diff;
  for (int d=0; d < D; d++) {
    if (d!=rm) {
      tmp_diff = pt1[d]-pt2[d];
      res += (tmp_diff*tmp_diff);
    }
  }
  return res;
}


#ifdef ARRAYFIRE
inline void printDims(const af::array &a) {
  for (int n=0; n < a.numdims(); n++) {
    if (n < a.numdims()-1) printf("%d x ", (int)a.dims(n)); else printf("%d\n", (int)a.dims(n));
  }
}


// converts an arma double matrix to an arrayfire float one
inline af::array arma_mat2af(const arma::mat &arma_mat) {
  arma::fmat farma_mat = arma::conv_to<arma::fmat>::from(arma_mat);
  af::array af_mat = af::array(arma_mat.n_rows, arma_mat.n_cols, farma_mat.memptr());
  return af_mat;
}


inline arma::mat af2arma_mat(const af::array &af_mat) {
  float *af_data = new float[af_mat.elements()];
  af_mat.host(af_data);
  int nDims = af_mat.numdims();
  int N = af_mat.dims(nDims-1);
  int fs = af_mat.elements()/N;
  arma::fmat farma_mat = arma::fmat(af_data, fs, N, true, true);
  arma::mat res = arma::conv_to<arma::mat>::from(farma_mat);
  return res;
}
#endif


inline void update_minimum(tbb::atomic<float>& minimum_value, const float &value) {
  float min_value;
  do {
    min_value = minimum_value;
    if (min_value <= value) break;
  } while(minimum_value.compare_and_swap(value, min_value) != min_value);
}


#endif
