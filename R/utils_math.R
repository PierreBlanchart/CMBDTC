# /** utils_math.R
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


#' @export RPCA_fun
#' X: fs x N
RPCA_fun <- function(X, kmax=NA, thresh.inertia=NA, normalize=FALSE) {
  
  X.mean <- rowMeans(X)
  
  X.c <- X - X.mean
  cov.mat <- (1/ncol(X))*(X.c%*%t(X.c))
  
  if (!is.na(thresh.inertia)) {
    if (is.na(kmax)) {
      obj.eigen <- svd(cov.mat, nu=nrow(X), nv=0)
    } else {
      obj.eigen <- svd(cov.mat, nu=min(kmax, nrow(X)), nv=0)
    }
    eig.values <- obj.eigen$d
    cum.inertia <- c(0, cumsum(eig.values)); cum.inertia <- cum.inertia/sum(diag(cov.mat))
    
    if (is.na(kmax)) {
      ind.cut <- match(TRUE, cum.inertia >= thresh.inertia)-1
    } else {
      ind.cut <- min(kmax, match(TRUE, cum.inertia >= thresh.inertia)-1)
    }
    if (is.na(ind.cut)) ind.cut <- as.numeric((!is.na(kmax)))*kmax + as.numeric((is.na(kmax)))*nrow(d)
    
  } else {
    obj.eigen <- svd(cov.mat, nu=kmax, nv=0)
    eig.values <- obj.eigen$d
    ind.cut <- kmax
  }
  # print(ind.cut)
  
  axis_pca <- obj.eigen$u[, 1:ind.cut] # eigenvectors
  nf <- rep(1, ind.cut)
  if (!normalize) {
    proj.X <- t(axis_pca)%*%X.c
  } else {
    nf <- sqrt(eig.values[1:ind.cut])
    proj.X <- (t(axis_pca)%*%X.c)*(1/nf)
  }
  
  return(
    list(
      centering=X.mean,
      axis=axis_pca,
      nf=nf,
      coords=proj.X
    )
  )
  
}


