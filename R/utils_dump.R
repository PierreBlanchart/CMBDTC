# /** utils_dump.R
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


#' @export dump_model
#' dump xgboost model into a list of array
#' @import data.table
#' @importFrom xgboost xgb.dump
#' @importFrom xgboost xgb.model.dt.tree
dump_model <- function(model, feature_names) {

  # dump model
  txt <- xgb.dump(model, with_stats = TRUE)
  dt <- xgb.model.dt.tree(feature_names=feature_names, model=model, text=txt)
  features.name <- unique(dt$Feature)
  # dt[, nb_nodes_per_tree:=length(Node), by=Tree]
  # nb_nodes_per_tree <- dt$nb_nodes_per_tree[sapply(1:nb_trees, function(x, set) match(TRUE, set==x), set=dt$Tree)]

  feature.names <- list()
  for (x in 1:length(feature_names)) {
    feature.names[[feature_names[x]]] <- x
  }
  feature.names[['Leaf']] <- as.numeric(NA)

  dt[, Node := as.numeric(Node)]
  dt[, yes.dest := as.numeric(gsub("[0-9]+-([0-9]+)", "\\1", Yes))]
  dt[, no.dest := as.numeric(gsub("[0-9]+-([0-9]+)", "\\1", No))]
  dt[, missing.dest := as.numeric(gsub("[0-9]+-([0-9]+)", "\\1", Missing))]
  # dt[, is.leaf := as.numeric(Feature == 'Leaf')]
  dt[, nFeature := as.numeric(as.vector(sapply(Feature, function(x) as.numeric(feature.names[[x]]))))]
  setorder(dt, Tree, Node)

  nb_trees <- length(unique(dt$Tree))
  dt_tree.matrix <- list()
  col2keep <- c('Node', 'yes.dest', 'no.dest', 'missing.dest', 'Split', 'Quality', 'nFeature') # c('yes.dest', 'no.dest', 'missing.dest', 'Split', 'Quality', 'nFeature', 'is.leaf')
  for (t in 1:nb_trees) {
    dt_tree <- data.matrix(dt[Tree==t-1, col2keep, with=F])
    ind.nodes <- dt_tree[, "Node"]
    if (sum(diff(ind.nodes) != 1) > 0 || ind.nodes[1] != 0) {
      max.node <- max(ind.nodes)
      new_dt_tree <- matrix(NA, max.node+1, ncol(dt_tree))
      colnames(new_dt_tree) <- colnames(dt_tree)
      new_dt_tree[ind.nodes+1, ] <- dt_tree
      dt_tree.matrix[[t]] <- new_dt_tree[, 2:ncol(dt_tree)]
    } else {
      dt_tree.matrix[[t]] <- dt_tree[, 2:ncol(dt_tree)]
    }
    dt_tree.matrix[[t]][is.na(dt_tree.matrix[[t]])] <- -1
  }

  return(dt_tree.matrix)

}



#' @export conf_mat
#' computes confusion matrix
conf_mat <- function(labels.pred, labels.gt, normalize=FALSE) {
  unique.labels <- unique(c(labels.pred, labels.gt))
  sz <- max(unique.labels)
  if (min(unique.labels) == 0) {
    classes.names <- 0:sz
    sz <- sz+1
  } else {
    classes.names <- 1:sz
  }

  conf.mat <- matrix(NA, sz, sz)
  colnames(conf.mat) <- classes.names
  rownames(conf.mat) <- classes.names
  for (i in unique.labels) {
    ind.i <- labels.gt==i
    nb.i <- sum(ind.i)
    row.i <- paste0(i)
    if (nb.i > 0) {
      for (j in unique.labels) {
        conf.mat[row.i, paste0(j)] <- sum(labels.pred[ind.i] == j)
      }
      if (normalize) conf.mat[row.i, ] <- conf.mat[row.i, ] / nb.i
    } else {
      for (j in unique.labels) {
        conf.mat[row.i, paste0(j)] <- 0
      }
    }
  }

  return(conf.mat)

}

