# /** demo_surrogate_multiclass.R
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


library(cmbdtc)
library(data.table)
library(xgboost)


################################################ load MNIST ################################################
print("loading MNIST ...")
mnist <- readRDS("./data/MNIST_dataset.rds")
data.train <- t(mnist$train$x) / 255
labels.train <- mnist$train$y
data.test <- t(mnist$test$x) / 255
labels.test <- mnist$test$y
mode(data.train) <- "numeric"
mode(data.test) <- "numeric"

thresh.PCA <- 0.7
obj.PCA <- RPCA_fun(data.train, NA, thresh.PCA, normalize=FALSE)
data.train <- obj.PCA$coords
fs <- nrow(data.train)
data.test <- t(obj.PCA$axis)%*%(data.test - obj.PCA$centering)
Ntest <- ncol(data.test)

ind2keep.train <- seq(1, ncol(data.train), by=20)
data.train <- t(data.train[, ind2keep.train])
labels.train <- labels.train[ind2keep.train]
Ntrain <- length(labels.train)
nb.per.class <- as.numeric(table(labels.train))
nClass <- length(nb.per.class)


###################################### training xgboost model ##############################################
dtrain <- xgb.DMatrix(data.train, label=labels.train)
dtest <- xgb.DMatrix(t(data.test), label=labels.test)


depth <- 3 # 3
nTrees <- 10
params.xgb <- list(
  objective = "multi:softprob", # "multi:softmax",
  eval_metric = "mlogloss",
  eta = 0.2,
  gamma=1e-1,
  min_child_weight=10,
  max_depth = depth,
  colsample_bytree = 0.8,
  num_class=nClass, 
  base_score=0
)


m1 <- xgb.train(data = dtrain, params.xgb, nrounds = nTrees, watchlist = list(mod = dtrain, val = dtest))
feature_names <- paste0('F', 1:fs)
dt_tree.matrix <- dump_model(m1, feature_names)

# compute accuracy and isolate miss-classified data
proba.test <- predict(m1, dtest, reshape = TRUE) # t(matrix(X, nClass)) -- Ntest x nClass
pred.test <- apply(proba.test, MARGIN=c(1), FUN=function(row) which.max(row)-1)
accuracy <- sum(labels.test == pred.test) / Ntest
print(paste("accuracy =", accuracy*100, "%"))


################################################################################################################################
# demonstrating model prediction (multi-class classification model) using cmbdtc4app pkg
pred.comp <- predict_model(
  data2predict=data.test[, 1, drop=FALSE],
  tree_list=dt_tree.matrix,
  max_depth=depth,
  nb_class=nClass,
  nb_trees_per_class=nTrees,
  thresh_dec=0.5,
  model_type="multi:softprob"
)
pred.xgboost <- predict(m1, t(data.test[, 1, drop=FALSE]))
print(rbind(pred.xgboost, as.numeric(pred.comp)))


################################################################################################################################
ind.queries <- which(labels.test != pred.test)
nb.queries <- length(ind.queries)
CF_class <- 0

ind.query <- ceiling(runif(1)*nb.queries) # between 1 and length(ind.queries)
ind.query.data.test <- ind.queries[ind.query]
query <- as.numeric(data.test[, ind.query.data.test])
if (CF_class == pred.test[ind.query.data.test]) stop("Choose different classes for predicted data and CF example")

print(paste0("label test = ", labels.test[ind.query.data.test]))
res.surrogate <- CF_find_surrogate(
  query=query,
  target=CF_class, 
  tree_list=dt_tree.matrix,
  max_depth=depth,
  nb_class=nClass,
  nb_trees_per_class=nTrees,
  thresh_dec=0.4,
  sigma = 8,
  Niter = 1024, 
  lr = 1e-3, # learning rate
  lambda_distortion = 1e0, 
  model_type="multi:softprob", 
  optim_crit="CE"
)
pred.sur <- predict(m1, matrix(res.surrogate$CF_example, nrow=1))


######################################################################################################################
# demonstrating exact CF example search using our method initialized with a search range computed from the surrogate
# find closest train data belonging to target class (0)
if (res.surrogate$found_solution) {
  sup.d2query <- res.surrogate$dist_CF2query^2 + 1e-2
  cheap_try <- sup.d2query/1
} else {
  data.train.target <- data.train[labels.train == 0, ]
  temp.diff <- t(data.train.target) - query
  sup.d2query <- min(colSums(temp.diff^2)) # should be adjusted to take the threshold for the CF query into account
  cheap_try <- sup.d2query/10
}

budget <- 1e6
max_dim_width_first <- 15
res <- CF_search(query=query, 
                 predicted_class=pred.test[ind.query.data.test], # predicted class of the query
                 target_class=CF_class, 
                 tree_list=dt_tree.matrix, 
                 max_depth=depth, 
                 nb_class=nClass, 
                 nb_trees_per_class=nTrees, 
                 sup_d2query_dataset=sup.d2query, # use square distance value here !
                 budget=budget, 
                 max_dim_width_first=max_dim_width_first,
                 check_has_target = TRUE, 
                 update_sup_bound = TRUE, 
                 model_type = "multi:softmax", 
                 nb_rec_iter=2, # 4, 
                 cheap_try=cheap_try
)
if (res$found_solution) {
  
  print(paste0("Found CF example at a distance ", sqrt(res$dist_CF2query)))
  comp <- rbind(query, as.numeric(res$CF_example))
  rownames(comp) <- c("query", "CF example")
  
  # plot reconstruction of CF example found from PCA base
  CF.rec <- obj.PCA$axis%*%as.numeric(res$CF_example) + obj.PCA$centering
  CF.rec <- pmax(pmin(CF.rec, 1), 0) # > thresh.binary
  par(pty="s", xaxt='n', yaxt='n', cex=0.7) # make axes square
  image(matrix(CF.rec, 28)[, 28:1])
  title("CF example")
}

