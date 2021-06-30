# /** demo_CF_MNIST_beam_search.R
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


################################################ loading MNIST data ################################################
print("loading MNIST ...")
mnist <- readRDS("./data/MNIST_dataset.rds")
data.train <- t(mnist$train$x) / 255
labels.train <- mnist$train$y
data.test <- t(mnist$test$x) / 255
labels.test <- mnist$test$y
mode(data.train) <- "numeric"
mode(data.test) <- "numeric"

thresh.PCA <- 0.7 # percentage of explained variance
obj.PCA <- RPCA_fun(data.train, NA, thresh.PCA, normalize=FALSE)
data.train <- obj.PCA$coords
fs <- nrow(data.train)
data.test <- t(obj.PCA$axis)%*%(data.test - obj.PCA$centering)

# select two classes for the binary classification scenario
ind.classes <- c(5, 6)

# formulate train and test sets
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

# remap labels in the ensemble {0, 1}
unique.labels <- ind.classes+1 # + 1 in case there are 0s
ind.remap <- c()
ind.remap[unique.labels] <- 0:(length(unique.labels)-1)
labels.train <- ind.remap[labels.train+1]
labels.test <- ind.remap[labels.test+1]
unique.labels <- unique.labels-1


###################################### training xgboost model ##############################################
dtrain <- xgb.DMatrix(data.train, label=labels.train)
dtest <- xgb.DMatrix(data.test, label=labels.test)


depth <- 7
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

ind.missclassified <- which(labels.test != pred.labels)
nb.missclassified <- length(ind.missclassified)
class.missclassified <- ind.classes[labels.test[ind.missclassified]+1] # true class of miss-classified data


################################################## CF example beam search #################################################
# making vary the decision threshold of a binary classification problem to obtain CF examples that are classified in the right class with more and more confidence by the model
ind.queries <- which((pred.labels == 1) & (labels.test == 0)) # missclassification in class 1: we look for the CF example in class 0 !
ind.query <- ceiling(runif(1)*length(ind.queries)) # between 1 and length(ind.queries)
ind.query.data.test <- ind.queries[ind.query]
query <- as.numeric(data.test[ind.query.data.test, ])

# find closest train data belonging to target class (0)
data.train.target <- data.train[labels.train == 0, ]
temp.diff <- t(data.train.target) - query
sup.d2query <- min(colSums(temp.diff^2)) # should be adjusted to take the threshold for the CF query into account

budget <- 4e6
max_dim_width_first <- 15

CONTINUE <- TRUE
thresh <- thresh.dec - 1e-2
step.thresh <- 3e-2

thresh.val <- c()
dCF <- c()
CF_examples <- matrix(NA, fs, 0)
is_approx <- c()

cheap_try <- sup.d2query/10
while (CONTINUE && thresh > 0) {
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
                   check_has_target= TRUE, 
                   update_sup_bound = TRUE, 
                   model_type = "binary:logistic", 
                   nb_rec_iter=2, # 4, 
                   cheap_try=cheap_try
  )
  if (res$found_solution) {
    CONTINUE <- TRUE
    thresh.val <- c(thresh.val, thresh)
    dCF <- c(dCF, sqrt(res$dist_CF2query))
    CF_examples <- cbind(CF_examples, as.numeric(res$CF_example))
    is_approx <- c(is_approx, res$is_approx)
    print(paste0("Found CF example at a distance ", sqrt(res$dist_CF2query), " with threshold ", thresh))
    
    thresh <- thresh-step.thresh
    cheap_try <- res$dist_CF2query + 1
  } else {
    CONTINUE <- FALSE
  }
}


#################################################### plot results ##########################################################
nsteps <- length(thresh.val)
nb.per.row <- 8
par(pty="s", # makes axes square
    mfrow=c(1, nb.per.row), 
    mar=c(0, 0, 0, 1), 
    xaxt='n', yaxt='n', # removes ticks
    cex=2
)

query.rec <- obj.PCA$axis%*%query + obj.PCA$centering
query.rec <- pmax(pmin(query.rec, 1), 0) > thresh.bin
image(matrix(query.rec, 28)[, 28:1])
title("Query", line=1)

to_plot <- c(1, round(seq(3, nsteps, length.out=nb.per.row-2)))
for (n in to_plot) {
  CF.rec <- obj.PCA$axis%*%CF_examples[, n] + obj.PCA$centering
  CF.rec <- pmax(pmin(CF.rec, 1), 0) > thresh.bin
  image(matrix(CF.rec, 28)[, 28:1])
  title(paste0("eps = ", thresh.val[n]), line=1, sub=paste0("dCF = ", round(dCF[n], 3)))
}


