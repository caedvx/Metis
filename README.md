# Metis

Where I prototype simulation engines. Pieces that hold up here move into Hyperion, a
magnetic-confinement fusion simulation. Write-up: [caedvx.com/hyperion](https://caedvx.com/hyperion/).

## PhysXEngine

GPU N-body particle engine: CUDA direct-sum gravity, leapfrog KDK integration on the
device, and real-time rendering with SDL3 + OpenGL 4.6 through CUDA-GL interop. An FP64
CPU path is the reference the tests check against; the FP32 GPU path drives the renderer.

### Accuracy

Semi-implicit Euler against the analytic Kepler orbit (`PhysXEngine/tests/test_kepler.cpp`):

| dt | position error at t=3600 s | ratio |
|---|---|---|
| 1.0 | 4.853756 m | — |
| 0.1 | 0.485360 m | 10.00 |
| 0.01 | 0.048514 m | 10.00 |
| 0.001 | 0.004836 m | 10.03 |

First order, as expected. Over a full lunar orbit (2.36e9 steps at dt=0.001) all
timesteps agree within 0.014 m and the orbital radius holds to under 1.3 mm.

Leapfrog KDK (`PhysXEngine/tests/test_leapfrog.cpp`) is second order: halving dt cuts
the error about 4x until it hits the FP64 round-off floor, between 1e-5 and 2e-5 m. At
dt=0.1 that's ~26,000x smaller than Euler. Full table in
[docs/design/01-gpu-engine.md](docs/design/01-gpu-engine.md#convergence-measured).

### Performance

RTX 2060, 1280x720, vsync off, native Wayland. The gate is average >= 144 fps and
p99.9 >= 120 fps. Measured with `metis_bench_noio`, first 600 frames discarded.

| particles | avg fps | p99.9 fps | gate |
|---|---|---|---|
| 8192 | 1411.6 | 518.3 | pass |
| 16384 | 556.9 | 343.8 | pass |
| 24576 | 254.3 | 211.0 | pass |
| 26624 | 185.9-212.0 | 133.8-161.1 | pass |
| 27136 | 185.1 | 110.3 | fail |
| 32768 | 169.3 | 114.6 | fail |

The ceiling sits between 26880 and 27136 particles. At 32768 the average still passes
but p99.9 doesn't.

### Build

```sh
cmake -S . -B build -G Ninja
cmake --build build -j6
ctest --test-dir build --output-on-failure
```

### Run

```sh
./build/PhysXEngine/metis_app [frame-count] [particle-count]
```

With no arguments it runs until Escape or the window closes, with 32768 particles. Left
drag orbits, right drag pans, scroll zooms. A frame count renders that many frames and
exits; a second argument sets the particle count.

## Design

- [GPU particle engine](docs/design/01-gpu-engine.md)
- [Electrostatic particle-in-cell](docs/design/02-electrostatic-pic.md), next

## License

AGPL-3.0. See [LICENSE](LICENSE).
