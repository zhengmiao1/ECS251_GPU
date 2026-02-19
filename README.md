## ECS251 GPU Scheduling Project

This repository contains a lightweight prototype for a memory-aware GPU scheduler
that coordinates admission and execution ordering for mixed LLM/VLM workloads.

### Goal (Quarter 1)
- Formalize the scheduling problem and build a minimal prototype.
- Provide essential functions for memory-feasible admission and ordering.
- Include a small simulator to exercise the policy.

### Project Structure
- `scripts/models.py`: core data structures (Task, GPUState, results)
- `scripts/scheduler.py`: memory-aware policy + FIFO baseline
- `scripts/metrics.py`: evaluation metrics
- `scripts/simulate.py`: synthetic workload generator and runner
- `scripts/experiment.py`: multi-seed baseline experiments
- `scripts/event_logger.py`: JSONL event logger for admission/dispatch traces
- `docs/PROPOSAL.md`: project context and timeline

### Quick Start
```
python -m scripts.simulate --tasks 120 --gpus 2 --gpu_mem 24 --policy both --workload mixed
```

Run multi-seed baseline experiments:
```
python -m scripts.experiment --tasks 200 --gpus 2 --gpu_mem 24 --workload mixed
```

Enable per-decision logs for report evidence:
```
python -m scripts.simulate --policy both --log_dir logs
```

### Notes
This is an initial milestone focusing on policy logic. The simulator is
deliberately simple and can be replaced with real workload traces later.
