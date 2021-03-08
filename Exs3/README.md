
Depending on which file i, number of characters n and r
default n = 6, r = 3

syscall = unm or cert

Splitting the testfile

```python3 split.py "syscalls/snd-{syscall}/snd-{syscall}.{i}.test" {n}```

Running the algorithm (DONT FORGET TO SPLIT TRAIN FILE)

```java -jar negsel2.jar -alphabet file://syscalls/snd-{syscall}/snd-{syscall}.alpha -self syscalls/snd-{syscall}/snd-{syscall}.train_split -n {n} -r {r} -c -l < syscalls/snd-{syscall}/snd-{syscall}.{i}.test_split > syscalls/snd-{syscall}/snd-{syscall}.{i}.output```


Formatting output (NOT NEEDED, BUT JUST IN CASE)

```python3 format_output.py "syscalls/snd-{syscall}/snd-{syscall}.{i}.output"```
