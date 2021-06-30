# /** demo_restricted_CF_regression.R
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

obj.data <- readRDS("./data/data_house_prices_train_test.rds")
data.train <- obj.data$data.train
labels.train <- obj.data$labels.train
data.test <- obj.data$data.test
labels.test <- obj.data$labels.test
Ntest <- length(labels.test)
lst.factors <- obj.data$lst.factors
is.factor <- obj.data$is.factor
mult.factor <- obj.data$mult.factor
fs <- ncol(data.train)


# if we'd like to make the sale price higher, which variables can be changed, and which ones cannot
fixed.variables <- c("MSSubClass", "MSZoning", "LotArea", "LotFrontage", "Street", "LotShape", "LandContour", 
                     "LotConfig", "LandSlope", "Neighborhood", "Condition1", "Condition2", 
                     "BldgType", "HouseStyle", "YearBuilt", "YearRemodAdd", "RoofStyle", 
                     "1stFlrSF", "2ndFlrSF", "GrLivArea", "GarageYrBlt", "GarageCars", "GarageArea", 
                     "PoolArea", "MiscFeature", "MiscVal", "TotalBsmtSF", "YrSold", 
                     "OverallQual", "TotRmsAbvGrd")


###################################### training xgboost model ##############################################
ind2modif <- setdiff(colnames(data.train), fixed.variables)
find.non_fixed <- match(ind2modif, colnames(data.train))
ind.non_fixed <- rep(FALSE, fs)
ind.non_fixed[find.non_fixed] <- TRUE

dtrain <- xgb.DMatrix(data.train, label=labels.train)
dtest <- xgb.DMatrix(data.test, label=labels.test)

depth <- 4
nTrees <- 300

model.type <- "reg:squarederror" # "reg:logistic"
params.xgb <- list(
  objective = model.type, 
  eval_metric = "rmse",
  eta = 0.02,
  gamma=1e-4,
  min_child_weight=10,
  max_depth = depth,
  colsample_bytree = 0.9, 
  base_score = 0 # should be set to 0 for the reg:squarederror loss !
)

m1 <- xgb.train(data = dtrain, params.xgb, nrounds = nTrees, watchlist = list(mod = dtrain, val = dtest))

predictions <- predict(m1, dtest)
RMSE <- sum((labels.test - predictions)^2) # /Ntest
print(paste0("RMSE = ", RMSE))


#############################################################################################################
feature_names <- paste0('F', 1:fs)
dt_tree.matrix <- dump_model(m1, feature_names)


#############################################################################################################
data2predict <- t(data.test[1:100, ])
pred.comp <- predict_model(
  data2predict=data2predict,
  tree_list=dt_tree.matrix,
  max_depth=depth,
  nb_class=1,
  nb_trees_per_class=nTrees,
  thresh_dec=-1,
  model_type=model.type
)
print(rbind(predictions[1:100], pred.comp))


#############################################################################################################
ind.query <- 17
query <- as.numeric(data.test[ind.query, ])
print(predictions[ind.query])

target_interval <- c(predictions[ind.query]+0.02, 1)
pred.train <- predict(m1, dtrain)
ind.inside <- pred.train >= target_interval[1] & pred.train <= target_interval[2]
if (sum(ind.inside) > 0) {
  data.inside <- data.train[ind.inside, ]
  temp.diff <- t(data.inside) - query
  sup.d2query <- min(colSums(temp.diff^2))
} else {
  sup.d2query <- Inf
}

res <- CF_search_with_mask_regression(
  query=query,
  target_interval=target_interval,
  mask_fixed_features=!ind.non_fixed,
  tree_list=dt_tree.matrix,
  max_depth=depth,
  nb_trees=nTrees,
  masked_sup_d2query_dataset=sup.d2query,
  budget=1e6,
  max_dim_width_first = 28,
  check_has_target = TRUE,
  update_sup_bound = TRUE, 
  nb_rec_iter=3,
  cheap_try=sup.d2query/20
)
pred.CF <- predict(m1, matrix(res$CF_example, 1))
print(pred.CF)

# display changes to perform
comp <- rbind(query[ind.non_fixed], res$CF_example[ind.non_fixed])
comp <- comp*matrix(rep(mult.factor[colnames(data.train)[ind.non_fixed]], each=2), 2)
colnames(comp) <- colnames(data.train)[ind.non_fixed]

is.factor.non_fixed <- match(is.factor, colnames(data.train)[ind.non_fixed])
is.factor.non_fixed <- is.factor.non_fixed[!is.na(is.factor.non_fixed)]
comp.rounded <- comp
comp.rounded[, is.factor.non_fixed] <- round(comp[, is.factor.non_fixed]+1e-8)
is.factor.fixed <- setdiff(1:ncol(comp), is.factor.non_fixed)
comp.rounded[, is.factor.fixed] <- round(comp[, is.factor.fixed], 2)

comp.rounded.char <- matrix(as.character(comp.rounded), 2, ncol(comp.rounded))
for (i in is.factor.non_fixed) {
  comp.rounded.char[, i] <- names(lst.factors[[colnames(comp)[i]]][comp.rounded[, i]+1])
}
colnames(comp.rounded.char) <- colnames(comp)
rownames(comp.rounded.char) <- c('initial', 'CF')

ind.diff <- comp.rounded.char[1, ] != comp.rounded.char[2, ]
comp.diff <- comp.rounded.char[, ind.diff, drop=FALSE]

print(paste0("House with initial price ", round(predictions[ind.query]* mult.factor['SalePrice'], 0)))
print(paste0("Minimal changes to perform to sell it within a target interval [", round(target_interval[1]* mult.factor['SalePrice'], 0), ", ", round(target_interval[2]* mult.factor['SalePrice'], 0), "]"))
print(comp.diff)
print(paste0("New sale price estimation: ", round(res$prediction* mult.factor['SalePrice'], 0)))


