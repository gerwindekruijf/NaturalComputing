# Olivier Brahma, Gerwin de Kruijf, Dirren van Vlijmen

library(AUC)
# load data scores 

# Load scores_english and scores_tagalog with 
# 1. Import dataset
# 2. From text base
# 3. Import the two datasets
scores_english$string = english$V1
scores_english$val = 0

scores_tagalog$string = tagalog$V1
scores_tagalog$val = 1

merged = merge(x = scores_english, y = scores_tagalog, all=TRUE)
sorted = merged[order(merged$V1),]

roc_x = roc(merged$V1, factor(merged$val))
x = auc(roc_x)

# R = 1
scores_eng_r1$string = english$V1
scores_eng_r1$val = 0

scores_tag_r1$string = tagalog$V1
scores_tag_r1$val = 1

merged = merge(x = scores_eng_r1, y = scores_tag_r1, all=TRUE)
sorted = merged[order(merged$V1),]

roc_x = roc(merged$V1, factor(merged$val))
x = auc(roc_x)

# R = 9
scores_eng_r9$string = english$V1
scores_eng_r9$val = 0

scores_tag_r9$string = tagalog$V1
scores_tag_r9$val = 1

merged = merge(x = scores_eng_r9, y = scores_tag_r9, all=TRUE)
sorted = merged[order(merged$V1),]

roc_x = roc(merged$V1, factor(merged$val))
x = auc(roc_x)

# Scores hiligaynon
scores_hiligaynon$string = "NA"
scores_hiligaynon$val = 1

merged = merge(x = scores_english, y = scores_hiligaynon, all=TRUE)
sorted = merged[order(merged$V1),]

roc_x = roc(merged$V1, factor(merged$val))
x = auc(roc_x)
print(x)

# Scores mid english
scores_mid_eng$string = "NA"
scores_mid_eng$val = 1

merged = merge(x = scores_english, y = scores_mid_eng, all=TRUE)
sorted = merged[order(merged$V1),]

roc_x = roc(merged$V1, factor(merged$val))
x = auc(roc_x)
print(x)

# Scores plaut
scores_plaut$string = "NA"
scores_plaut$val = 1

merged = merge(x = scores_english, y = scores_plaut, all=TRUE)
sorted = merged[order(merged$V1),]

roc_x = roc(merged$V1, factor(merged$val))
x = auc(roc_x)
print(x)

# Scores xhosa
scores_xhosa$string = "NA"
scores_xhosa$val = 1

merged = merge(x = scores_english, y = scores_xhosa, all=TRUE)
sorted = merged[order(merged$V1),]

roc_x = roc(merged$V1, factor(merged$val))
x = auc(roc_x)
print(x)


