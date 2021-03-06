# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

compute_all <- function(tree_list, fs, max_depth, nb_class, nb_trees_per_class, thresh_dec) {
    .Call('_cmbdtc_compute_all', PACKAGE = 'cmbdtc', tree_list, fs, max_depth, nb_class, nb_trees_per_class, thresh_dec)
}

predict_model <- function(data2predict, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, model_type) {
    .Call('_cmbdtc_predict_model', PACKAGE = 'cmbdtc', data2predict, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, model_type)
}

CF_find <- function(query, predicted_class, target_class, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type) {
    .Call('_cmbdtc_CF_find', PACKAGE = 'cmbdtc', query, predicted_class, target_class, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type)
}

CF_find_with_mask <- function(query, predicted_class, target_class, mask_fixed_features, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, masked_sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type) {
    .Call('_cmbdtc_CF_find_with_mask', PACKAGE = 'cmbdtc', query, predicted_class, target_class, mask_fixed_features, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, masked_sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type)
}

CF_find_regression <- function(query, target_interval, tree_list, max_depth, nb_trees, sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type) {
    .Call('_cmbdtc_CF_find_regression', PACKAGE = 'cmbdtc', query, target_interval, tree_list, max_depth, nb_trees, sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type)
}

CF_find_with_mask_regression <- function(query, target_interval, mask_fixed_features, tree_list, max_depth, nb_trees, masked_sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type) {
    .Call('_cmbdtc_CF_find_with_mask_regression', PACKAGE = 'cmbdtc', query, target_interval, mask_fixed_features, tree_list, max_depth, nb_trees, masked_sup_d2query_dataset, budget, max_dim_width_first, check_has_target, update_sup_bound, model_type)
}

CF_find_surrogate <- function(query, target, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, sigma, Niter, lr, lambda_distortion, model_type, optim_crit) {
    .Call('_cmbdtc_CF_find_surrogate', PACKAGE = 'cmbdtc', query, target, tree_list, max_depth, nb_class, nb_trees_per_class, thresh_dec, sigma, Niter, lr, lambda_distortion, model_type, optim_crit)
}

