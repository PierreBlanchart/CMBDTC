# /** demo_surrogate_binary.R
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

ind.classes <- round(runif(2)*9)
if (ind.classes[1] == ind.classes[2]) stop("Choose two different classes for two-class clssification problem")
nClass <- 2

ind2keep.train <- which(labels.train %in% ind.classes)
ind2keep.train <- ind2keep.train[seq(1, length(ind2keep.train), by=3)] # subsample

data.train <- t(data.train[, ind2keep.train])
labels.train <- labels.train[ind2keep.train]
Ntrain <- length(labels.train)
nb.per.class <- as.numeric(table(labels.train))

ind2keep.test <- which(labels.test %in% ind.classes)
data.test <- t(data.test[, ind2keep.test])
labels.test <- labels.test[ind2keep.test]
Ntest <- length(labels.test)

# remap labels betweeen 0 and nClass-1
unique.labels <- ind.classes+1 # + 1 in case there are 0s
ind.remap <- c()
ind.remap[unique.labels] <- 0:(length(unique.labels)-1)
labels.train <- ind.remap[labels.train+1]
labels.test <- ind.remap[labels.test+1]
unique.labels <- unique.labels-1


###################################### training xgboost model ##############################################
dtrain <- xgb.DMatrix(data.train, label=labels.train)
dtest <- xgb.DMatrix(data.test, label=labels.test)


depth <- 7 # 3
nTrees <- 100
params.xgb <- list(
  objective = "binary:logistic",
  eval_metric = "logloss", # "auc",
  eta = 0.01,
  gamma=1,
  min_child_weight=10,
  max_depth = depth,
  colsample_bytree = 0.8, 
  scale_pos_weight = nb.per.class[1] / nb.per.class[2]
)

m1 <- xgb.train(data = dtrain, params.xgb, nrounds = nTrees, watchlist = list(mod = dtrain, val = dtest))
feature_names <- paste0('F', 1:fs)
dt_tree.matrix <- dump_model(m1, feature_names)


# compute accuracy and isolate miss-classified data
thresh.dec <- 0.5
pred.test <- predict(m1, dtest)
pred.labels <- as.numeric(pred.test > thresh.dec)
accuracy <- sum(labels.test == pred.labels) / Ntest
print(paste("accuracy =", accuracy*100, "%"))


######################################################################################################################
# demonstrating model prediction (two-class binary classification model) using cmbdtc4app pkg
pred.comp <- predict_model(
  data2predict=t(data.test[1, , drop=FALSE]),
  tree_list=dt_tree.matrix,
  max_depth=depth,
  nb_class=2,
  nb_trees_per_class=nTrees,
  thresh_dec=0.5,
  model_type="binary:logistic"
)
print(paste0("cmbdtc4app prediction:", pred.comp[1], " | xgboost prediction: ", pred.test[1]))


######################################################################################################################
# demonstrating approximate CF example search using a derivable tree ensemble model surrogate
ind.queries <- which((pred.labels == 1) & (labels.test == 0)) # missclassification in class 1: we look for the CF example in class 0 !
ind.query <- ceiling(runif(1)*length(ind.queries)) # between 1 and length(ind.queries)
ind.query.data.test <- ind.queries[ind.query]
query <- as.numeric(data.test[ind.query.data.test, ])

# plot reconstructed query in PCA basis
thresh.binary <- 0.5
query.rec <- obj.PCA$axis%*%query + obj.PCA$centering
query.rec <- pmax(pmin(query.rec, 1), 0) > thresh.binary
par(pty="s", xaxt='n', yaxt='n', cex=0.7) # make axes square
image(matrix(query.rec, 28)[, 28:1])
title(paste0("Initial image: ", unique.labels[1], " missclassifed as a ", unique.labels[2]), line=1)

res.surrogate <- CF_find_surrogate(
  query=query,
  target=0, 
  tree_list=dt_tree.matrix,
  max_depth=depth,
  nb_class=2,
  nb_trees_per_class=nTrees,
  thresh_dec=0.4,
  sigma = 8,
  Niter = 1024, 
  lr = 1e-3, # learning rate
  lambda_distortion = 2e-1, 
  model_type="binary:logistic", 
  optim_crit="binary_CE"
)
print(predict(m1, matrix(res.surrogate$CF_example, 1)))


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

budget <- 4e6
max_dim_width_first <- 15
thresh <- thresh.dec - 5e-2
res <- CF_search(query=query, 
                 predicted_class=1, # predicted class of the query
                 tree_list=dt_tree.matrix, 
                 max_depth=depth, 
                 nb_class=2, 
                 nb_trees_per_class=nTrees, 
                 thresh_dec=thresh, 
                 sup_d2query_dataset=sup.d2query, # use square distance value here !
                 budget=budget, 
                 max_dim_width_first=max_dim_width_first, 
                 check_has_target=FALSE, 
                 update_sup_bound = TRUE, 
                 model_type = "binary:logistic", 
                 nb_rec_iter=2, # 4, 
                 cheap_try=cheap_try
)
if (res$found_solution) {
  print(paste0("Found CF example at a distance ", sqrt(res$dist_CF2query), " with threshold ", thresh))
  comp <- rbind(query, as.numeric(res$CF_example))
  rownames(comp) <- c("query", "CF example")
  print(comp)
  
  # plot reconstruction of CF example found from PCA base
  CF.rec <- obj.PCA$axis%*%as.numeric(res$CF_example) + obj.PCA$centering
  CF.rec <- pmax(pmin(CF.rec, 1), 0) > thresh.binary
  par(pty="s", xaxt='n', yaxt='n', cex=0.7) # make axes square
  image(matrix(CF.rec, 28)[, 28:1])
  title("CF example")
}


