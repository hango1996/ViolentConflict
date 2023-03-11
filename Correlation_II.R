#------------------------------------------------------
# packages
#------------------------------------------------------
library(clustertend)
library(NbClust)
library(factoextra)
library(ClusterR) # Distance matrix computation 
library(fpc)
library(clusterSim)
library(psych)
library(FactoMineR) # MCA 
library(clustMixType)
library(hopkins)
library(dplyr)
library(Rtsne)
library(ggraph)
library(rsetse)
library(proxy)
library(tidyverse)
library(kernlab)
library(dbscan)

#===============================================================
# Auxiliary functions 
#===============================================================
jaccard <- function(a, b) {
  intersection = length(intersect(a, b))
  union = length(a) + length(b) - intersection
  return (1-(intersection/union))
}

#================================================================
sample_data = read.csv("actor&event.csv")
proj_data = read.csv("data.csv")
var_int <- c("region", "country", "latitude", "longitude")
new_data_post  = proj_data[, var_int]
actor_event_geolocation_data <- cbind(sample_data, new_data_post)
time_fatality_data <- read.csv("time&faltatliy.csv")
time_fatality_data2 <- time_fatality_data[, -c(1,2)]
#========================================================================
# Combined data 
#========================================================================
combined_data <- cbind(actor_event_geolocation_data, time_fatality_data2) 
#=======================================================================
new_combined_data <- combined_data[,-c(1,2)]
# correlation vector between MCA Euclidean similarity matrix and 1/2(Jaccard + Euclidean)
p = 10
Cor <- numeric(p)
for (k1 in 1:p){
sample_combined_data <- sample_n(new_combined_data, 3000)
sample_num <- sample_combined_data[, 5:9]
sample_num <- as.data.frame(scale(sample_num))
sample_cat <- sample_combined_data[,c(1:4)]
#
# matrix of categorical variables 
sample_cat_mat <- as.matrix(sample_cat) 
#=============================================================================
# jaccard similarity matrix  
#============================================================================
JM <- matrix(data = NA, nrow = nrow(sample_cat_mat), ncol= nrow(sample_cat_mat))
for (i1 in 1:nrow(sample_cat_mat)){
  for (j1 in 1:nrow(sample_cat_mat)){
    a = sample_cat_mat[i1,]
    b <- sample_cat_mat[j1,]
    JM[i1,j1] <- jaccard(a,b)
  }
}
Jac_mat <- as.matrix(JM)
# Euclidean similarity 
EM <- distance_matrix(sample_num, method = "euclidean", upper = TRUE, diagonal = TRUE)
Euclid_mat <- as.matrix(EM)
sim_matrix <- 0.5*(Euclid_mat + Jac_mat) 
# #======================================================================
# # MCA, Changing categorical variables into numerical variable
# #======================================================================
sample_cat$interaction <-as.factor(sample_cat$interaction)
res.mca1 <- MCA(X=sample_cat, graph = FALSE)
mca1_obs_df = data.frame(res.mca1$ind$coord)
#
# 
# # Creating new data set using 
new_data <- cbind(mca1_obs_df,sample_num)
scaled_new_data <- as.data.frame(scale(new_data))
# 
# new similarity matrix 
sim2 = distance_matrix(scaled_new_data, method = "euclidean", upper = TRUE, diagonal = TRUE)
sim_matrix2 <- as.matrix(sim2)
v1 = as.vector(sim_matrix)
v2 = as.vector(sim_matrix2)
r = cor(v1,v2)
Cor[k1] = r
}


