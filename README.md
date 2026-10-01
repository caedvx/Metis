# Metis

Learning and prototyping ground for simulation engines and base ideas. Components
that prove out here get lifted into the projects that need them — primarily
[Hyperion](https://github.com/caedvx/Hyperion), a magnetic-confinement fusion
simulation.

## PhysXEngine

GPU N-body particle engine: CUDA direct-sum gravity, leapfrog KDK integration
entirely on device, and real-time rendering through SDL3 + OpenGL 4.6 via
CUDA-GL interop. A CPU reference path (FP64) backs the test suite and oracle
checks; the device path (FP32) drives the interactive renderer.

### Current state

Semi-implicit Euler, verified against the analytic Kepler solution
(`tests/test_kepler.cpp`):

| dt | position error at t=3600 s | ratio |
|---|---|---|
| 1.0 | 4.853756 m | — |
| 0.1 | 0.485360 m | 10.00 |
| 0.01 | 0.048514 m | 10.00 |
| 0.001 | 0.004836 m | 10.03 |

Clean first-order convergence. Over a full lunar orbit (2.36e9 steps at dt=0.001) all
timestep choices agree within 0.014 m and orbital radius holds to under 1.3 mm.

Leapfrog KDK (`tests/test_leapfrog.cpp`), against the same analytic solution: cutting
dt by 4x (400 -> 100) cuts the error by ~19x, consistent with second-order O(dt^2)
convergence against Euler's 10x (first order). At dt=0.1, leapfrog's error is already
down to 0.0000187 m — the FP64 round-off floor for this problem, ~26,000x smaller
than Euler's 0.485360 m at the same timestep.

### Performance

Measured on an RTX 2060 (sm_75, driver 610.57.04, CUDA 13.3), native Wayland,
vsync disabled, at 1280x720. Gate: average >= 144 fps **and p99.9 >= 120 fps**.
(An earlier draft of this gate used an absolute worst-frame floor; that is not
a convergent statistic — a long enough run always finds a worse frame, so it
is unpassable by construction rather than a real requirement. p99.9 is the
reproducible version of the same intent.) Frame times were sampled with no
per-frame I/O in the measured loop, discarding the first 600 of each run as
warm-up (shader compilation, allocation, GPU clock ramp-up).

| particles | avg fps | p99 fps | p99.9 fps | max fps | gate |
|---|---|---|---|---|---|
| 8192 | 1411.6 | 601.4 | 518.3 | 465.1 | pass |
| 16384 | 556.9 | 365.4 | 343.8 | 318.5 | pass |
| 20480 | 302.2 | 247.6 | 227.5 | 213.6 | pass |
| 24576 | 254.3 | 218.0 | 211.0 | 201.1 | pass |
| 25600 | 250.0 | 212.4 | 204.0 | 182.8 | pass |
| 26624 | 185.9-212.0 | 165.1-172.3 | 133.8-161.1 | 130.5-153.9 | **pass** |
| 26880 | ~199.6 | 135.8 | 123.3-125.9 | 114.5-118.5 | **pass** (margin ~3-5%) |
| 27136 | 185.1 | 143.2 | 110.3 | 89.1 | fail |
| 27392 | 197.2 | 132.3 | 118.5 | 114.4 | fail (just under) |
| 27648 | 175.2 | 120.9 | 107.9 | 105.3 | fail |
| 28672 | 167.7 | 109.9 | 101.0 | 93.4 | fail |
| **32768** | 169.3 | 120.5 | **114.6** | 111.7 | **fail** |
| 35000 | ~117-125 | — | — | ~13-89 | fail (avg alone) |
| 40000 | ~102-109 | — | — | ~70-79 | fail (avg alone) |
| 49152 | ~74-75 | — | — | ~64-66 | fail (avg alone) |

Ranges are across repeated independent runs (2-3 repeats for the values
nearest the boundary). p50 is not in this table: the instrumentation to
record it was added after this sweep was run, and a same-session attempt to
back-fill it was confounded by another GPU-intensive application sharing the
card at the time (confirmed via `nvidia-smi`: 0-95% utilization and 45-150 W
draw from a process other than the benchmark) — those numbers are discarded
rather than reported as clean.

At the target 32768, average clears the gate (169.3 fps) but p99.9 does not
(114.6 fps). 26624 held the gate in every repeat tried, including a
6000-frame sustained check, with the worst single run's p99.9 at 133.8 fps
(11% margin). 26880 is the largest tested count that still passes, but only
two runs were done there and the margin is thin (p99.9 123.3 and 125.9 fps,
3-5% above the floor); 27136 fails clearly (p99.9 110.3 fps). The true
ceiling sits somewhere in (26880, 27136] — not pinned further, since the GPU
became shared with another application before a finer sweep could run
cleanly. The cause of the worst-frame dips themselves was not identified (not
explained by diagnostic I/O, kernel scaling, or thermal throttling — each was
checked and ruled out); the 26624/26880 boundary is therefore not known to be
compute-bound, and could shift on a different run of the same hardware.

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

With no arguments, runs interactively until Escape or window close, default
32768 particles. Left drag orbits, right drag pans, scroll zooms. An optional
frame count renders exactly that many frames and exits (0 renders nothing);
an optional second argument overrides the particle count.

### Where it is going

See [the design spec](docs/design/01-gpu-engine.md)
for the full architecture and the measured outcome against its original goals.
