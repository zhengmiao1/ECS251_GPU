## ECS251 GPU Scheduling Project

This repository contains a lightweight prototype for a memory-aware GPU scheduler
that coordinates admission and execution ordering for mixed LLM/VLM workloads.

### Goal (Quarter 1)
- Formalize the scheduling problem and build a minimal prototype.
- Provide essential functions for memory-feasible admission and ordering.
- Include a small simulator to exercise the policy.

### Project Structure
- `src/models.py`: core data structures (Task, GPUState, results)
- `src/scheduler.py`: memory-aware admission + ordering policy
- `src/metrics.py`: basic evaluation metrics
- `src/simulate.py`: synthetic workload generator and runner
- `docs/PROPOSAL.md`: project context and timeline

### Quick Start
```
python -m src.simulate --tasks 100 --gpus 2 --gpu_mem 24
```

### Notes
This is an initial milestone focusing on policy logic. The simulator is
deliberately simple and can be replaced with real workload traces later.
