# Genetic Random Search (GRS)

GRS is a low-level, CUDA-aware experimentation sandbox for training neural-network graphs with evolutionary exploration. The project mixes high-throughput GPU search, CPU simulations, and reusable builders so you can move quickly between research prototypes and production-quality agents.

## Why use GRS?

- **Instruction-level execution.** Networks are flattened into compact instruction buffers (`PipelineBuilder`) for fast CPU/GPU replay and easy serialization.
- **Hierarchical search.** `GeneticRandomSearch` and its "smart" variants run stair-based populations, leaderboard optimizers, and adaptive noise to explore difficult games.
- **GPU-first batching.** CUDA kernels and managed memory let you evaluate thousands of rollouts in parallel without hand-writing kernels for every model.
- **Composable environments.** Batch environments abstract reward accumulation for classic GuessGame variants or your own custom tasks.
- **Comprehensive tests.** Catch2 suites cover models, pipelines, optimizers, instructions, and environment helpers so regressions are caught early.

## Project layout

- `src/` – headers for models, instructions, helpers, optimizers, and search agents (`src/grs` for classic GRS, `src/grs/smart*.h` for smart variants).
- `gpu_tests/` – CUDA benchmarks and stress tests (`gpu_tests/grs_performance_test.cu`, `gpu_tests/sgrs_performance_test.cu`).
- `tests/` – Catch2-based unit tests covering CPU behavior.
- `keras_utils/` – optional Python helpers for hybrid workflows.
- `Dockerfile` – CUDA 12.6 image for reproducible builds.

## Quick start

```bash
# Clone or sync sources, then from the repository root:
make build_dir
make -j$(nproc) test          # sequential CPU test execution + sample CUDA build
# or for a faster local loop (CPU + GPU tests run in parallel)
make fast_test
```

### Only CPU tests

```bash
make build_dir
# Build every Catch2 binary
for test_src in tests/*.cpp; do \
  exe="$(basename "${test_src}" .cpp)"; \
  make "build/${exe}" || exit 1; \
  ./build/"${exe}" || exit 1; \
done
```

### Dockerized workflow

```bash
docker build . -t lowlevelrl
docker run --gpus all lowlevelrl make fast_test
```

## Using SmartGeneticRandomSearch

The "smart" agent layers adaptive learning-rate schedules and multi-population handoffs on top of classic GRS. A minimal end-to-end training loop looks like this:

```cpp
#include "grs/smart.h"
#include "game_examples.h"

void train_guess_game()
{
    Input input_layer(5);
    Dense action_head(&input_layer, 2);
    Model nn(&input_layer, &action_head);

    SmartGeneticRandomSearch trainer(&nn, /*stairs=*/6, /*grs_amount=*/4);
    GuessGame game(/*seed=*/0);
    trainer.train(&game, /*epochs=*/200, /*num_games=*/32);

    saveParams("guess_game_actor", trainer.weights, trainer.weights_size * trainer.directions);
}
```

`SmartGeneticRandomSearchGPU` shares the same API but executes rollouts on-device. After instantiating it you can reuse `initIterator`, `getNext`, and `updateWeights` in exactly the same way while rewards arrive from `BatchEnvironmentGPU`.

## Advanced GPU orchestration

- Launch full GPU loops with `index.cu` or CUDA performance tests in `gpu_tests/`.
- `SmartGeneticRandomSearchGPU` allocates managed memory, randomized kernels, and per-population CUDA streams for massive throughput.
- Helper utilities in `src/helper_functions/` expose serialization helpers, weight dumping, and heap-based sorting shared by CPU and GPU agents.

## Contributing

1. Keep headers under 1k LOC and prefer translation units for heavy implementations.
2. Extend or add Catch2 suites alongside new source files.
3. Run `make test` (CPU focus) or `make fast_test` (CPU+GPU) before opening a PR.
4. Document new agent behaviors in `AGENTS.md` (see below).

## Support & discussions

- Use `tests/*` as living documentation for public APIs.
- GPU-specific defects: open issues with your CUDA version and `nvidia-smi` output.
- For Python helpers (`keras_utils`), ensure TensorFlow is available (installed automatically inside the Docker image).
