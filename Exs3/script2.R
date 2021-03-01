# Olivier Brahma, Gerwin de Kruijf, Dirren van Vlijmen

# starting path = user


path = "Documents/GitHub/NaturalComputing/Exs3/"
path_cert = paste0(path, "syscalls/snd-cert/")
path_unm = paste0(path, "syscalls/snd-unm/")

args_ = paste0("-jar ", path, "negsel2.jar -self ", path, "english.train", 
                 " -n 10 -r 4 -c -l < ", path, "english.test")

# system2("java", args_)
cert_1 = read.delim(paste0(path_cert, "snd-cert.1.test"), header = FALSE)
cert_1_l = read.delim(paste0(path_cert, "snd-cert.1.labels"), header = FALSE)

cert_2 = read.delim(paste0(path_cert, "snd-cert.2.test"), header = FALSE)
cert_2_l = read.delim(paste0(path_cert, "snd-cert.2.labels"), header = FALSE)

cert_3 = read.delim(paste0(path_cert, "snd-cert.3.test"), header = FALSE)
cert_3_l = read.delim(paste0(path_cert, "snd-cert.3.labels"), header = FALSE)

cert_alpha = read.delim(paste0(path_cert, "snd-cert.alpha"), header = FALSE)
cert_train = read.delim(paste0(path_cert, "snd-cert.train"), header = FALSE)

split_train_set <- function(n, dataset, name){
  result = c()
  
  for(s in dataset$V1){
    splitted = strsplit(s,"(?<=.{100})", perl = TRUE)[[1]]
    for(t in splitted){
      result = append(result, t)
    }
  }
  print(result)
  write.table(result, paste0(path, "/", name))
}

# split_train_set(100, cert_train, "syscalls/snd-cert/cert_train_split")


# Test sets
split_test_set <- function(n, testset, labels_, name){
  result = data.frame(Chunk=character(), String_Number=numeric(), Label=numeric())
  testset$labels <- labels_$V1
  
  for(s in 1:nrow(testset)){
    # print(testset[s,"labels"])
    splitted = strsplit(testset[s,"V1"],"(?<=.{100})", perl = TRUE)[[1]]
    for(t in splitted){
      result[nrow(result) + 1,] = list(t, s, testset[s, "labels"])
    }
  }
  return(result)
}

# CERT
x <- split_test_set(100, cert_1, cert_1_l, "syscalls/snd-cert/cert_1_split")
write.table(x$Chunk, paste0(path, "/syscalls/snd-cert/cert_1_split"), row.names = FALSE, 
            col.names = FALSE, quote = FALSE)

x <- split_test_set(100, cert_2, cert_2_l, "syscalls/snd-cert/cert_2_split")
write.table(x$Chunk, paste0(path, "/syscalls/snd-cert/cert_2_split"), row.names = FALSE, 
            col.names = FALSE, quote = FALSE)

x <- split_test_set(100, cert_3, cert_3_l, "syscalls/snd-cert/cert_3_split")
write.table(x$Chunk, paste0(path, "/syscalls/snd-cert/cert_3_split"), row.names = FALSE, 
            col.names = FALSE, quote = FALSE)

# UNM
unm_1 = read.delim(paste0(path_unm, "snd-unm.1.test"), header = FALSE)
unm_1_l = read.delim(paste0(path_unm, "snd-unm.1.labels"), header = FALSE)

unm_2 = read.delim(paste0(path_unm, "snd-unm.2.test"), header = FALSE)
unm_2_l = read.delim(paste0(path_unm, "snd-unm.2.labels"), header = FALSE)

unm_3 = read.delim(paste0(path_unm, "snd-unm.3.test"), header = FALSE)
unm_3_l = read.delim(paste0(path_unm, "snd-unm.3.labels"), header = FALSE)

x <- split_test_set(100, unm_1, unm_1_l, "syscalls/snd-unm/unm_1_split")
write.table(x$Chunk, paste0(path, "/syscalls/snd-unm/unm_1_split"), row.names = FALSE, 
            col.names = FALSE, quote = FALSE)

x <- split_test_set(100, unm_2, unm_2_l, "syscalls/snd-unm/unm_2_split")
write.table(x$Chunk, paste0(path, "/syscalls/snd-unm/unm_2_split"), row.names = FALSE, 
            col.names = FALSE, quote = FALSE)

x <- split_test_set(100, unm_3, unm_3_l, "syscalls/snd-unm/unm_3_split")
write.table(x$Chunk, paste0(path, "/syscalls/snd-unm/unm_3_split"), row.names = FALSE, 
            col.names = FALSE, quote = FALSE)


# paste0(path, "syscalls/snd-unm/")






