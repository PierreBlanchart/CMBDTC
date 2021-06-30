# /** utils_CF.R
# *
# * Copyright (C) 2021 Pierre BLANCHART
# * pierre.blanchart@cea.fr
# * CEA/LIST/DM2I/SID/LI3A
# * This program is free software: you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation, either version 3 of the License, or
# * (at your option) any later version.
# * 
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# * 
# * You should have received a copy of the GNU General Public License
# * along with this program.  If not, see <https://www.gnu.org/licenses/>.
# **/


#' @export CF_search
CF_search <- function(query, 
                      predicted_class, # predicted class of the query
                      target_class = -1, # targeted class for the CF example
                      tree_list, 
                      max_depth, 
                      nb_class=2, 
                      nb_trees_per_class, 
                      thresh_dec=0.5, 
                      sup_d2query_dataset, # use square distance value here !
                      budget=2e7, 
                      max_dim_width_first=12, 
                      check_has_target=TRUE, 
                      update_sup_bound=TRUE, 
                      model_type = "binary:logistic", 
                      nb_rec_iter=2, 
                      tol=1e-7, 
                      cheap_try=-1
) {
  
  sup_cur <- sup_d2query_dataset
  
  do_cheap <- (cheap_try > 0 && cheap_try < sup_d2query_dataset)
  if (do_cheap) {
    res.cheap <- CF_find(query=query, 
                         predicted_class=predicted_class, # predicted class of the query
                         target_class=target_class, 
                         tree_list=tree_list, 
                         max_depth=max_depth, 
                         nb_class=nb_class, 
                         nb_trees_per_class=nb_trees_per_class, 
                         thresh_dec=thresh_dec, 
                         sup_d2query_dataset=cheap_try, # use square distance value here !
                         budget=budget, 
                         max_dim_width_first=max_dim_width_first, 
                         check_has_target=check_has_target, 
                         update_sup_bound=update_sup_bound, 
                         model_type=model_type
    )
    
    if (res.cheap$found_solution) {
      if (!res.cheap$is_approx) {
        return(res.cheap)
      } else {
        sup_cur <- res.cheap$dist_CF2query + 1e-4
      }
    }
    
  }
  
  has.best.res <- FALSE
  if (do_cheap && res.cheap$found_solution) {
    best.res <- res.cheap
    has.best.res <- TRUE
  }
  for (i in 1:nb_rec_iter) {
    res.i <- CF_find(query=query, 
                     predicted_class=predicted_class, # predicted class of the query
                     target_class=target_class, 
                     tree_list=tree_list, 
                     max_depth=max_depth, 
                     nb_class=nb_class, 
                     nb_trees_per_class=nb_trees_per_class, 
                     thresh_dec=thresh_dec, 
                     sup_d2query_dataset=sup_cur, # use square distance value here !
                     budget=budget, 
                     max_dim_width_first=max_dim_width_first, 
                     check_has_target=check_has_target, 
                     update_sup_bound=update_sup_bound, 
                     model_type=model_type
    )
    
    if (!res.i$found_solution) {
      if (has.best.res) {
        return(best.res)
      } else {
        return(res.i)
      }
    }
    
    if (!res.i$is_approx || (has.best.res && abs(best.res$dist_CF2query - res.i$dist_CF2query) < tol)) {
      # print(paste0("Found CF example at a distance ", sqrt(res.i$dist_CF2query)))
      return(res.i)
    } else {
      print(paste0("iter rec ", i, ": Found CF example at a distance ", sqrt(res.i$dist_CF2query)))
      if (has.best.res) {
        if (res.i$dist_CF2query < best.res$dist_CF2query) {
          best.res <- res.i
        }
      } else {
        best.res <- res.i
        has.best.res <- TRUE
      }
      sup_cur <- best.res$dist_CF2query + 1e-4
    }
    
  }
  
  print("No exact solution found within the allocated number of search iterations")
  return(best.res)
}



#' @export CF_search_with_mask
CF_search_with_mask <- function(query, 
                                predicted_class, # predicted class of the query
                                target_class = -1, # targeted class for the CF example
                                mask_fixed_features, 
                                tree_list, 
                                max_depth, 
                                nb_class=2, 
                                nb_trees_per_class, 
                                thresh_dec=0.5, 
                                masked_sup_d2query_dataset, # use square distance value here !
                                budget=2e7, 
                                max_dim_width_first=12, 
                                check_has_target=TRUE, 
                                update_sup_bound=TRUE, 
                                model_type = "binary:logistic", 
                                nb_rec_iter=2, 
                                tol=1e-7, 
                                cheap_try=-1
) {
  
  sup_cur <- masked_sup_d2query_dataset
  
  do_cheap <- (cheap_try > 0 && cheap_try < masked_sup_d2query_dataset)
  if (do_cheap) {
    res.cheap <- CF_find_with_mask(query=query, 
                                   predicted_class=predicted_class, # predicted class of the query
                                   target_class=target_class, 
                                   mask_fixed_features=mask_fixed_features, 
                                   tree_list=tree_list, 
                                   max_depth=max_depth, 
                                   nb_class=nb_class, 
                                   nb_trees_per_class=nb_trees_per_class, 
                                   thresh_dec=thresh_dec, 
                                   masked_sup_d2query_dataset=cheap_try, # use square distance value here !
                                   budget=budget, 
                                   max_dim_width_first=max_dim_width_first, 
                                   check_has_target=check_has_target, 
                                   update_sup_bound=update_sup_bound, 
                                   model_type=model_type
    )
    
    if (res.cheap$found_solution) {
      if (!res.cheap$is_approx) {
        return(res.cheap)
      } else {
        sup_cur <- res.cheap$dist_CF2query + 1e-4
      }
    }
    
  }
  
  has.best.res <- FALSE
  if (do_cheap && res.cheap$found_solution) {
    best.res <- res.cheap
    has.best.res <- TRUE
  }
  for (i in 1:nb_rec_iter) {
    res.i <- CF_find_with_mask(query=query, 
                               predicted_class=predicted_class, # predicted class of the query
                               target_class=target_class,
                               mask_fixed_features=mask_fixed_features, 
                               tree_list=tree_list, 
                               max_depth=max_depth, 
                               nb_class=nb_class, 
                               nb_trees_per_class=nb_trees_per_class, 
                               thresh_dec=thresh_dec, 
                               masked_sup_d2query_dataset=sup_cur, # use square distance value here !
                               budget=budget, 
                               max_dim_width_first=max_dim_width_first, 
                               check_has_target=check_has_target, 
                               update_sup_bound=update_sup_bound, 
                               model_type=model_type
    )
    
    if (!res.i$found_solution) {
      if (has.best.res) {
        return(best.res)
      } else {
        return(res.i)
      }
    }
    
    if (!res.i$is_approx || (has.best.res && abs(best.res$dist_CF2query - res.i$dist_CF2query) < tol)) {
      # print(paste0("Found CF example at a distance ", sqrt(res.i$dist_CF2query)))
      return(res.i)
    } else {
      print(paste0("iter rec ", i, ": Found CF example at a distance ", sqrt(res.i$dist_CF2query)))
      if (has.best.res) {
        if (res.i$dist_CF2query < best.res$dist_CF2query) {
          best.res <- res.i
        }
      } else {
        best.res <- res.i
        has.best.res <- TRUE
      }
      sup_cur <- best.res$dist_CF2query + 1e-4
    }
    
  }
  
  print("No exact solution found within the allocated number of search iterations")
  return(best.res)
}



#' @export CF_search_regression
CF_search_with_mask_regression <- function(query, 
                                           target_interval, # targeted interval for the CF example
                                           mask_fixed_features, 
                                           tree_list, 
                                           max_depth, 
                                           nb_trees, 
                                           masked_sup_d2query_dataset, # use square distance value here !
                                           budget=2e7, 
                                           max_dim_width_first=12, 
                                           check_has_target = TRUE, 
                                           update_sup_bound = TRUE, 
                                           model_type = "reg:squarederror",
                                           nb_rec_iter=2, 
                                           tol=1e-7, 
                                           cheap_try=-1
) {
  
  sup_cur <- masked_sup_d2query_dataset
  
  do_cheap <- (cheap_try > 0 && cheap_try < masked_sup_d2query_dataset)
  if (do_cheap) {
    res.cheap <- CF_find_with_mask_regression(query=query, 
                                              target_interval=target_interval,
                                              mask_fixed_features=mask_fixed_features,
                                              tree_list=tree_list, 
                                              max_depth=max_depth, 
                                              nb_trees=nb_trees, 
                                              masked_sup_d2query_dataset=cheap_try, # use square distance value here !
                                              budget=budget, 
                                              max_dim_width_first=max_dim_width_first, 
                                              check_has_target = check_has_target, 
                                              update_sup_bound = update_sup_bound, 
                                              model_type=model_type
    )
    
    if (res.cheap$found_solution) {
      if (!res.cheap$is_approx) {
        return(res.cheap)
      } else {
        sup_cur <- res.cheap$dist_CF2query + 1e-4
      }
    }
    
  }
  
  has.best.res <- FALSE
  if (do_cheap && res.cheap$found_solution) {
    best.res <- res.cheap
    has.best.res <- TRUE
  }
  for (i in 1:nb_rec_iter) {
    res.i <- CF_find_with_mask_regression(query=query, 
                                          target_interval=target_interval,
                                          mask_fixed_features=mask_fixed_features,
                                          tree_list=tree_list, 
                                          max_depth=max_depth, 
                                          nb_trees=nb_trees, 
                                          masked_sup_d2query_dataset=sup_cur, # use square distance value here !
                                          budget=budget, 
                                          max_dim_width_first=max_dim_width_first, 
                                          check_has_target=check_has_target, 
                                          update_sup_bound = update_sup_bound, 
                                          model_type=model_type
    )
    
    if (!res.i$found_solution) {
      if (has.best.res) {
        return(best.res)
      } else {
        return(res.i)
      }
    }
    
    if (!res.i$is_approx || (has.best.res && abs(best.res$dist_CF2query - res.i$dist_CF2query) < tol)) {
      # print(paste0("Found CF example at a distance ", sqrt(res.i$dist_CF2query)))
      return(res.i)
    } else {
      print(paste0("iter rec ", i, ": Found CF example at a distance ", sqrt(res.i$dist_CF2query)))
      if (has.best.res) {
        if (res.i$dist_CF2query < best.res$dist_CF2query) {
          best.res <- res.i
        }
      } else {
        best.res <- res.i
        has.best.res <- TRUE
      }
      sup_cur <- best.res$dist_CF2query + 1e-4
    }
    
  }
  
  print("No exact solution found within the allocated number of search iterations")
  return(best.res)
}



