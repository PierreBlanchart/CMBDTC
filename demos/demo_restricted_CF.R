# /** demo_restricted_CF.R
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

# summary of attributes
# "Status of existing checking account", # 1
# "Credit duration in month", # 2
# "Credit history", # 3
# "Purpose", # 4
# "Credit amount", # 5
# "Savings account/bonds", # 6
# "Present employment since", # 7
# "Installment rate in percentage of disposable income", # 8
# "Personal status and sex", # 9
# "Other debtors / guarantors", # 10
# "Present residence since", # 11
# "Property", # 12
# "Age", # 13
# "Other installment plans", # 14, 
# "Housing", # 15
# "Number of existing credits at this bank", # 16
# "Job", # 17
# "Number of people to provide maintenance for", # 18
# "Telephone", # 19
# "Foreign worker" # 20

# load already trained model
obj.model <- readRDS("./data/model_credit.rds")
model <- obj.model$model
dt_tree.matrix <- obj.model$dt_tree.matrix
fs <- obj.model$fs
depth <- obj.model$depth
nTrees <- obj.model$nTrees

# load dataset
obj.data <- readRDS("./data/data_credit_train_test.rds")
data.train <- obj.data$data.train
labels.train <- obj.data$labels.train
data.test <- obj.data$data.test
labels.test <- obj.data$labels.test
Ntest <- length(labels.test)

# predict test data
colnames(data.test) <- NULL
dtest <- xgb.DMatrix(data.test, label=labels.test)
pred.test <- predict(model, dtest)
pred.labels <- as.numeric(pred.test > 0.5)
print(paste0("Accuray = ", (sum(pred.labels == labels.test)/Ntest)*100, "%"))

# find which are the true credit denial cases
ind.queries <- which((pred.labels == 1) & (labels.test == 1)) # true faults (credit was refused)
nb.queries <- length(ind.queries) # total number of CF queries that can be formulated on this dataset

# pick one credit denial case
index.query <- ind.queries[ceiling(runif(1)*nb.queries)]
query <- data.test[index.query, ]

# find closest normal data to query in train dataset
data.train.normal <- data.train[labels.train == 0, ]
temp.diff <- t(data.train.normal) - as.numeric(query)
sup.d2query <- min(colSums(temp.diff^2))

# find closest CF example to query point
budget <- 1e6
max_dim_width_first <- 14
res <- CF_search(query=query,
                 predicted_class=1, # predicted class of the query
                 tree_list=dt_tree.matrix,
                 max_depth=depth,
                 nb_class=2,
                 nb_trees_per_class=nTrees,
                 thresh_dec=0.5,
                 sup_d2query_dataset=sup.d2query, # use square distance value here !
                 budget=budget,
                 max_dim_width_first=max_dim_width_first,
                 check_has_target= TRUE,
                 update_sup_bound = TRUE,
                 model_type = "binary:logistic",
                 nb_rec_iter=2, # 4,
                 cheap_try=sup.d2query/10
)
if (res$found_solution) {
  print(paste0("Found CF example at a distance ", sqrt(res$dist_CF2query)))
  comp <- rbind(query, as.numeric(res$CF_example))
  rownames(comp) <- c("query", "CF example")
  print(comp)
  
  print(predict(model, matrix(res$CF_example, 1)))
}


# restricted CF query on the same query point than before
attr.names <- colnames(data.train)
ind.fixed <- rep(FALSE, fs)
ind.fixed[c(3, 6, 7, 8, 9, 10, 11, 13)] <- TRUE
res.fixed <- CF_search_with_mask(query=query, 
                                 predicted_class=1, # predicted class of the query
                                 mask_fixed_features = ind.fixed, 
                                 tree_list=dt_tree.matrix, 
                                 max_depth=depth, 
                                 nb_class=2, 
                                 nb_trees_per_class=nTrees, 
                                 thresh_dec=0.5, 
                                 masked_sup_d2query_dataset=sup.d2query, # use square distance value here !
                                 budget=budget, 
                                 max_dim_width_first=max_dim_width_first, 
                                 check_has_target= TRUE, 
                                 update_sup_bound = TRUE, 
                                 model_type = "binary:logistic", 
                                 nb_rec_iter=2, # 4, 
                                 cheap_try= res$dist_CF2query+1e-2
)
if (res.fixed$found_solution) {
  print(paste0("Found CF example at a distance ", sqrt(res.fixed$dist_CF2query)))
  # comp <- rbind(query, as.numeric(res.fixed$CF_example))
  # rownames(comp) <- c("query", "CF example")
  # print(comp)
  # print(predict(model, matrix(res.fixed$CF_example, 1)))
}

# formulate recommendations
res.comp <- cbind(query, res$CF_example, res.fixed$CF_example)
res.comp <- apply(res.comp, MARGIN = 2, FUN=function(x) ((x-floor(x)) >= 0.5)*round(x+0.5, 0) + ((x-floor(x)) < 0.5)*round(x, 0) ) # revert to categorical variables
colnames(res.comp) <- c("query", "unrestricted CF", "restricted CF")
rownames(res.comp) <- attr.names
print(res.comp)

