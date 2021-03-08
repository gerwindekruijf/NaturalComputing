
# Task 2
Depending on which file _i_, number of characters _n_ and _r_

Default n = 6, r = 3

syscall = {unm, cert}

### Splitting the testfile

```python3 split.py "syscalls/snd-{syscall}/snd-{syscall}.{i}.test" {n}```

### Running the algorithm (DON'T FORGET TO SPLIT THE .TRAIN FILE)

```java -jar negsel2.jar -alphabet file://syscalls/snd-{syscall}/snd-{syscall}.alpha -self syscalls/snd-{syscall}/snd-{syscall}.train_split -n {n} -r {r} -c -l < syscalls/snd-{syscall}/snd-{syscall}.{i}.test_split > syscalls/snd-{syscall}/snd-{syscall}.{i}.output```


### Formatting output (NOT NEEDED, BUT JUST IN CASE)

```python3 format_output.py "syscalls/snd-{syscall}/snd-{syscall}.{i}.output"```

### Computing ROC and AUC
After splitting the testfiles and running the algorithm using those testfiles, run _script2.R_ 

The AUC scores are computed for each testfile


