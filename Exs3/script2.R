# Olivier Brahma, Gerwin de Kruijf, Dirren van Vlijmen

# starting path = user

#Path for Dirren
#path = "D:/Documents/GitHub/NaturalComputing/Exs3/"
#Path for Gerwin & Olivier
path = "Documents/GitHub/NaturalComputing/Exs3/"

# CERT ----------------------------------------------------
path_cert = paste0(path, "syscalls/snd-cert/")

# reading the data

# cert 1
cert_1 = read.delim(paste0(path_cert, "snd-cert.1.test"), header = FALSE)
cert_1$Labels = read.delim(paste0(path_cert, "snd-cert.1.labels"), header = FALSE)$V1
cert_1_split = read.delim(paste0(path_cert, "snd-cert.1.test_split"), header = FALSE)
cert_1_split$Refs = read.delim(paste0(path_cert, "snd-cert.1.test_ref"), header = FALSE)$V1
cert_1_split$Output = read.delim(paste0(path_cert, "snd-cert.1.output_format"), header = FALSE)$V1
cert_1$Score = aggregate(cert_1_split$Output, list(cert_1_split$Refs), mean)$x

# cert 2
cert_2 = read.delim(paste0(path_cert, "snd-cert.2.test"), header = FALSE)
cert_2$Labels = read.delim(paste0(path_cert, "snd-cert.2.labels"), header = FALSE)$V1
cert_2_split = read.delim(paste0(path_cert, "snd-cert.2.test_split"), header = FALSE)
cert_2_split$Refs = read.delim(paste0(path_cert, "snd-cert.2.test_ref"), header = FALSE)$V1
cert_2_split$Output = read.delim(paste0(path_cert, "snd-cert.2.output_format"), header = FALSE)$V1
cert_2$Score = aggregate(cert_2_split$Output, list(cert_2_split$Refs), mean)$x

# cert 3
cert_3 = read.delim(paste0(path_cert, "snd-cert.3.test"), header = FALSE)
cert_3$Labels = read.delim(paste0(path_cert, "snd-cert.3.labels"), header = FALSE)$V1
cert_3_split = read.delim(paste0(path_cert, "snd-cert.3.test_split"), header = FALSE)
cert_3_split$Refs = read.delim(paste0(path_cert, "snd-cert.3.test_ref"), header = FALSE)$V1
cert_3_split$Output = read.delim(paste0(path_cert, "snd-cert.3.output_format"), header = FALSE)$V1
cert_3$Score = aggregate(cert_3_split$Output, list(cert_3_split$Refs), mean)$x

# processing results
library(AUC)

# cert 1
cert_roc1 = roc(cert_1$Score, factor(cert_1$Labels))
cert_auc1 = auc(cert_roc1)

# cert 2
cert_roc2 = roc(cert_2$Score, factor(cert_2$Labels))
cert_auc2 = auc(cert_roc2)

# cert 3
cert_roc3 = roc(cert_3$Score, factor(cert_3$Labels))
cert_auc3 = auc(cert_roc3)

# UNM ----------------------------------------------------
path_unm = paste0(path, "syscalls/snd-unm/")

# reading the data

# unm 1
unm_1 = read.delim(paste0(path_unm, "snd-unm.1.test"), header = FALSE)
unm_1$Labels = read.delim(paste0(path_unm, "snd-unm.1.labels"), header = FALSE)$V1
unm_1_split = read.delim(paste0(path_unm, "snd-unm.1.test_split"), header = FALSE)
unm_1_split$Refs = read.delim(paste0(path_unm, "snd-unm.1.test_ref"), header = FALSE)$V1
unm_1_split$Output = read.delim(paste0(path_unm, "snd-unm.1.output_format"), header = FALSE)$V1
unm_1$Score = aggregate(unm_1_split$Output, list(unm_1_split$Refs), mean)$x

# unm 2
unm_2 = read.delim(paste0(path_unm, "snd-unm.2.test"), header = FALSE)
unm_2$Labels = read.delim(paste0(path_unm, "snd-unm.2.labels"), header = FALSE)$V1
unm_2_split = read.delim(paste0(path_unm, "snd-unm.2.test_split"), header = FALSE)
unm_2_split$Refs = read.delim(paste0(path_unm, "snd-unm.2.test_ref"), header = FALSE)$V1
unm_2_split$Output = read.delim(paste0(path_unm, "snd-unm.2.output_format"), header = FALSE)$V1
unm_2$Score = aggregate(unm_2_split$Output, list(unm_2_split$Refs), mean)$x

# unm 3
unm_3 = read.delim(paste0(path_unm, "snd-unm.3.test"), header = FALSE)
unm_3$Labels = read.delim(paste0(path_unm, "snd-unm.3.labels"), header = FALSE)$V1
unm_3_split = read.delim(paste0(path_unm, "snd-unm.3.test_split"), header = FALSE)
unm_3_split$Refs = read.delim(paste0(path_unm, "snd-unm.3.test_ref"), header = FALSE)$V1
unm_3_split$Output = read.delim(paste0(path_unm, "snd-unm.3.output_format"), header = FALSE)$V1
unm_3$Score = aggregate(unm_3_split$Output, list(unm_3_split$Refs), mean)$x

# processing results
library(AUC)

# unm 1
unm_roc1 = roc(unm_1$Score, factor(unm_1$Labels))
unm_auc1 = auc(unm_roc1)

# unm 2
unm_roc2 = roc(unm_2$Score, factor(unm_2$Labels))
unm_auc2 = auc(unm_roc2)

# unm 3
unm_roc3 = roc(unm_3$Score, factor(unm_3$Labels))
unm_auc3 = auc(unm_roc3)
