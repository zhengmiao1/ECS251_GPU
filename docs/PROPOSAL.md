## An Efficient GPU Scheduling Method for Task Workloads

Authors: Zheng Miao, Zaishuo Xia, Qiyao Ma

### Problem and Importance
Large-scale LLM and VLM workloads place heavy demand on shared GPUs. The main
issue is not single jobs exceeding memory, but poor coordination among
concurrent submissions. Without memory-aware admission and ordering, tasks can
be admitted into saturated GPUs, causing avoidable OOM failures and long queues.

### Existing Solutions and Gaps
Most schedulers allocate GPUs exclusively, which ignores dynamic memory
pressure and task heterogeneity. GPU sharing techniques improve utilization,
but without admission control and ordering they can increase contention and
instability.

### Proposed Solution
The scheduler enforces memory-feasible admission decisions and orders tasks
using lightweight indicators (estimated memory and duration class). Short
feasible tasks can proceed without being blocked by long tasks, while large
jobs are admitted when resources are sufficient.

### Anticipated Results
We expect fewer OOM failures, reduced waiting time, and higher utilization in
mixed LLM/VLM workloads.

### Project Timeline (Quarter 1)
- Weeks 2-3: problem formalization and literature review
- Weeks 4-5: policy design (admission + ordering rules)
- Weeks 6-7: prototype implementation
- Week 8: experimental setup and evaluation
- Week 9: analysis and report

### Evaluation Plan
Compare against FIFO and exclusive allocation. Measure OOM frequency, average
wait time, utilization, and completion time variance across users. Include
microbenchmarks for admission and ordering behavior under memory pressure.
