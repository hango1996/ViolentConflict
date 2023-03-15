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
    intersection =  length(a[a %in% intersect(a, b)])
    #length(intersect(a, b))
  union = length(a) + length(b) - intersection
  return (1-(intersection/union))
}


#=======================================================
# Data combining
#======================================================

# Data from Han 
new_feature_data <- read.csv("data_feat_stage3.csv")

var_exc <- c("iso", "event_id_cnty", "event_id_no_cnty","event_date", "year", 
             "actor1","assoc_actor_1", "actor2", "assoc_actor_2","geo_precision",
              "source","source_scale","notes", "iso3", "time_precision")

clean_data <- new_feature_data[, !names(new_feature_data)%in%var_exc]

sum(is.na(clean_data))

#df <- replace(df, df=='', NA)

cd = replace(clean_data, clean_data=="", NA)
sum(is.na(cd))

# dropping NAs
cd <- cd %>% drop_na() 

#var_exc2 <- c("data_id", "admin1", "admin2", "admin3", "location")
#cd2  <- cd[, !names(cd)%in%var_exc2]
#names(cd2)


# selecting categorical variables 
var_cat <- c("data_id", "event_type", "sub_event_type", "inter1", "inter2",
             "interaction", "region", "country", "admin1",
             "admin2", "admin3", "location")

cd_sample <- sample_n(cd, 3000)
df <- cd_sample 
df_cat <- df[, names(df)%in%var_cat]
df_cat2 <- df_cat[,-1]
# Selecting numerical variables 
df_num <- df[, !names(df)%in%var_cat]

#============================================================================
# Applying MCA 
#=========================================================================
df_cat2$inter1 <- as.factor(df_cat2$inter1)
df_cat2$inter2 <- as.factor(df_cat2$inter2)
df_cat2$interaction <-as.factor(df_cat$interaction)
res.mca_cat <- MCA(X=df_cat2, graph = FALSE, ncp = 25)
mca_obs_df = data.frame(res.mca_cat$ind$coord)

#==============================================================
# Combined numeric data 
#=============================================================
new_df_num <- cbind(df_num, mca_obs_df)
scaled_df_num <- as.data.frame(scale(new_df_num))
sim1 = distance_matrix(scaled_df_num, method = "euclidean", upper = TRUE, diagonal = TRUE)
sim_mat_1 <- as.matrix(sim1)

#=======================================================
# Jaccard matrix 
#=====================================================

#============================================================================
df_cat_mat <- as.matrix(df_cat2)
k1 <- nrow(df_cat_mat)  
JM <- matrix(data = NA, nrow = k1, ncol= k1)
for (i1 in 1:k1){
  for (j1 in 1:k1){
    a = df_cat_mat[i1,]
    b <- df_cat_mat[j1,]
    JM[i1,j1] <- jaccard(a,b)
  }
}



jm_matrix = as.matrix(JM)

scale_df_num <- as.data.frame(scale(df_num))

EM <- distance_matrix(scale_df_num, method = "euclidean", upper = TRUE, diagonal = TRUE)
em_matrix = as.matrix(EM)

sim_mat_2 <- 0.5*(em_matrix+jm_matrix)

v1 <- as.vector(sim_mat_1)
v2 <- as.vector(sim_mat_2)
r = cor(v1,v2)
r





