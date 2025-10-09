# Agents Guide

This reference focuses on the reusable search agents and helper abstractions that power the GRS stack. Each entry highlights when to use the agent, the data it needs, and how it interacts with the rest of the system.

## Core agents

| Agent | Location | When to use it | Key inputs | Produces |
| --- | --- | --- | --- | --- |
| `GeneticRandomSearch` | `src/grs/core.h` | Baseline CPU exploration for compact models or when CUDA is unavailable | `Model*`, `stairs` | In-place weight buffers, reward history, optimizer telemetry |
| `SmartGeneticRandomSearch` | `src/grs/smart.h` | Multi-population CPU search with adaptive learning rate and anti-local-minima logic | `Model*`, `stairs`, `grs_amount` | Updated weight pools, tuned learning rates, `BatchEnvironment` rewards |
| `SmartGeneticRandomSearchGPU` | `src/grs/smart_gpu.h` | Large-scale experiments that need CUDA-managed rollouts and per-population streams | `Model*`, `stairs`, `grs_amount` | Device-resident weights, GPU rewards, adjustable streams |
| `LearnableOptimizer` | `src/grs_optimizers/core.h` & GPU variant | Replaces the default iterative optimizer with a learnable policy | GRS directions, pretrained weights (optional) | Updated optimizer weights, per-direction reward deltas |
| `BatchEnvironment*` | `src/environment/core.h` & GPU variant | Shared batching/utilities for running instructions and caching rewards | `PipelineBuilder*`, `batch_size` | `RunnerInfo` iterators, reward arrays, datastream/memory slabs |

## Selecting an agent
- Use **classic GRS** when you need deterministic, debuggable CPU training loops or want to unit test instruction builders.
- Use **Smart GRS** when performance plateau avoidance matters; anti-local minima logic and stair replication make it easier to escape flat reward landscapes.
- Use **Smart GRS GPU** to scale out to thousands of rollouts per iteration. It mirrors the CPU API so you can port existing loops by switching constructors.

## Integration tips
1. Always create a `Model` and call `BuildFastExecutionGraph(true)` before wiring it into any agent so builders can cache accurate instruction layouts.
2. Keep `stairs` modest during iteration (4–8) and grow it once reward trends stabilize—each stair duplicates populations.
3. When switching between CPU and GPU runs, save or copy weights through `saveParams`/`loadParams` helpers in `src/helper_functions/core.h`.
4. `BatchEnvironment` iterators (`initIterator`, `getNext`) drive both CPU and GPU loops; tests in `tests/batch_environment_test.cpp` show minimal usage patterns.
5. For GPU deployments, guard CUDA calls and stream counts—`SmartGeneticRandomSearchGPU` now throws descriptive errors when allocations fail, which helps isolate driver issues early.

## Adding new agents
- Implement the CPU version first so it can leverage the existing test harness and Catch2 suites.
- Derive GPU variants from the CPU counterpart only when the API stabilizes; prefer composition if you only need kernels for a subset of operations.
- Update this document with the new agent’s name, header, and quick-start tips to keep discoverability high.
