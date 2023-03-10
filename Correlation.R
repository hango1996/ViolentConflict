#------------------------------------------------------
# packages
#------------------------------------------------------
library(clustertend)
library(NbClust)
library(factoextra)
library(ClusterR)
library(fpc)
library(clusterSim)
library(psych)
library(FactoMineR) # MCA 
library(clustMixType)
library(dplyr)
library(Rtsne)
library(ggraph)
library(rsetse)
library(proxy)
library(tidyverse)
library(kernlab)
library(dbscan)
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
# Data for south Africa
Southern_Africa <- filter(combined_data, region == "Southern Africa")
#=======================================================================
SA = Southern_Africa[,-1]
SA2 = SA[,-1]
SA3 = SA2[,-3]

#=============================================================================
# jaccard matrix
#============================================================================
jaccard <- function(a, b) {
  intersection = length(intersect(a, b))
  union = length(a) + length(b) - intersection
  return (intersection/union)
}
# Numerical part of the data
SA_num <- SA3[,c(4,5,6,7,8)]
scaled_sa_num = scale(SA_num)
#categorical paprt of the data 
SA_Cat <- SA3[, c(1,2,3)]
SA_cat_mat <- as.matrix(SA_Cat[1:5000,])
#
# Jaccard Matrix
JM <- matrix(data = NA, nrow = nrow(SA_cat_mat), ncol= nrow(SA_cat_mat))
for (i in 1:nrow(SA_cat_mat)){
  for (j in 1:nrow(SA_cat_mat)){
    a = SA_cat_mat[i,]
    b <- SA_cat_mat[j,]
    JM[i,j] <- jaccard(a,b)
  }
}
JM[1:5,1:5]
#
# Euclidean similarity 
EM <- daisy(scaled_sa_num[1:5000,], metric = "euclidean")
EM_mat <- as.matrix(EM)
sim_matrix <- 0.5*(EM_mat + JM) 
sim_matrix[1:5,1:5]
# #======================================================================
# # MCA, Changing categorical variables into numerical variable
# #======================================================================
SA_Cat$interaction <-as.factor(SA_Cat$interaction)
res.mca1 <- MCA(X=SA_Cat, graph = FALSE)
mca1_obs_df = data.frame(res.mca1$ind$coord)
# 
# # Creating new data set using 
new_data <- cbind(mca1_obs_df,SA_num)
scaled_new_data <- scale(new_data)
# 
sca2 <- scaled_new_data[1:5000,] 
# new similarity matrix 
sim2 = daisy(sca2, metric = "euclidean")
sim_matrix2 <- as.matrix(sim2)

v1 = as.vector(sim_matrix)
v2 = as.vector(sim_matrix2)
r = cor(v1,v2)
r 

