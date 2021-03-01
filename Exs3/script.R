# Olivier Brahma, Gerwin de Kruijf, Dirren van Vlijmen

# library(AUC)
# load data scores 

# Load scores_english and scores_tagalog with 
# 1. Import dataset
# 2. From text base
# 3. Import the two datasets
merged = merge(x = scores_english, y = scores_tagalog, by="V1", all=TRUE)

# Sort dataset
sorted = merged[order(merged$V1),]





