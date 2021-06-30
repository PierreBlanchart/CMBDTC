/** mem_allocator.h
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


#ifndef DEF_MEM_ALLOCATOR_H
#define DEF_MEM_ALLOCATOR_H

#include "utils.h"


class mem_allocator {
public:
  mem_allocator(const int Nleaves_, const task_type ttype_=binary_classif, const int Nclass_=-1) : Nleaves(Nleaves_) {
    temp_intervals = new float[2*Nleaves];
    is_start.resize(2*Nleaves);
    temp_intersect.resize(Nleaves);
    temp_index = new unsigned int[2*Nleaves];
    temp_pos = new unsigned int[2*Nleaves];
    sorted_index = new unsigned int[2*Nleaves];
    sorted_temp_intervals = new float[2*Nleaves];
    sorted_is_start.resize(2*Nleaves);
    sorted_temp_index = new unsigned int[2*Nleaves];
    
    nb_intersect = new unsigned int[2*Nleaves];
    segments = new float[2*Nleaves];
    intersect_bitset = new boost::dynamic_bitset<>[2*Nleaves];
    
    dquery_seg = new float[2*Nleaves];
    ptc2seg = new float[2*Nleaves];
    dquery2regions = new float[2*Nleaves];
    
    switch(ttype_) {
      case binary_classif:
      case regression_task: {
        scores_binary = new float[2*Nleaves]; // used for both binary classification and regression
        break;
      }
      case multi_classif: {
        scores_multi = makeMatrix<float>(Nclass_, 2*Nleaves);
        break;
      }
      default: break;
    }
  };
  
  ~mem_allocator() {
    delete[] temp_intervals;
    is_start.clear();
    temp_intersect.clear();
    delete[] temp_index;
    delete[] temp_pos;
    delete[] sorted_index;
    delete[] sorted_temp_intervals;
    sorted_is_start.clear();
    delete[] sorted_temp_index;
    
    delete[] nb_intersect;
    delete[] segments;
    
    for (int i=0; i < 2*Nleaves; i++) intersect_bitset[i].clear();
    delete[] intersect_bitset;
    
    delete[] dquery_seg;
    delete[] ptc2seg;
    delete[] dquery2regions;
    if (scores_binary) delete[] scores_binary;
    if (scores_multi) deleteMatrix<float>(scores_multi);
  };
public:
  const int Nleaves;
  float *temp_intervals = NULL;
  boost::dynamic_bitset<> is_start;
  boost::dynamic_bitset<> temp_intersect;
  unsigned int *temp_index = NULL;
  unsigned int *temp_pos = NULL;
  unsigned int *sorted_index = NULL;
  float *sorted_temp_intervals = NULL;
  boost::dynamic_bitset<> sorted_is_start;
  unsigned int *sorted_temp_index = NULL;
  
  unsigned int *nb_intersect = NULL;
  float *segments = NULL;
  boost::dynamic_bitset<> *intersect_bitset = NULL;
  
  float *dquery_seg = NULL;
  float *ptc2seg = NULL;
  float *dquery2regions = NULL;
  float *scores_binary = NULL;
  float **scores_multi = NULL;
};



#endif
