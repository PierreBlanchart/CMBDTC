.onLoad <- function(libname, pkgname) {
  print(paste0("Loading dynamic library ", libname, '/', pkgname, '.so'))
  library.dynam(pkgname, package = pkgname, lib.loc=libname, now = FALSE)
}
