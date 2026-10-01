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

Leapfrog KDK integration, verified against the analytic Kepler solution:

| dt | position error at t=3600 s | ratio |
|---|---|---|
| 1.0 | 4.853756 m | — |
| 0.1 | 0.485360 m | 10.00 |
| 0.01 | 0.048514 m | 10.00 |
| 0.001 | 0.004836 m | 10.03 |

Clean first-order convergence. Over a full lunar orbit (2.36e9 steps at dt=0.001) all
timestep choices agree within 0.014 m and orbital radius holds to under 1.3 mm.

### Performance

Measured on an RTX 2060 (sm_75, driver 610.57.04, CUDA 13.3), native Wayland,
vsync disabled. Figures exclude the first ~600-900 frames of each run (shader
compilation, allocation, and GPU clock/thermal ramp-up); frame count is per
`FrameTimer`'s 300-sample rolling window, so "worst" is the worst frame within
that window, sampled every 120 frames for the duration of a multi-thousand-frame
run.

| particles | avg fps | worst fps | gate (avg>=144, worst>=120) |
|---|---|---|---|
| 8192 | ~1400 | ~465 | pass |
| 16384 | ~557 | ~318 | pass |
| 20480 | ~302 | ~214 | pass |
| 24576 | ~254 | ~201 | pass |
| 25600 | ~250 | ~183 | pass |
| 26624 | ~198-216 | ~124-185 | **pass** |
| 26880 | ~200 | ~114-118 | fail (marginal) |
| 27648 | ~175 | ~105 | fail |
| 28672 | ~168 | ~93 | fail |
| 32768 | ~165-213 | ~109-129 | **fail** (worst floor) |
| 35000 | ~117-125 | ~13-89 | fail |
| 40000 | ~102-109 | ~70-79 | fail |
| 49152 | ~74-75 | ~64-66 | fail |

At the target 32768, the average comfortably clears 144 fps (sustained
160-215 fps), but the worst-frame floor does not reliably hold 120 fps: true
worst-case (measured over 5100 post-warmup frames with no per-frame I/O) was
111.7 fps. The largest particle count that held the worst-frame floor
reliably across repeated runs, including a 6000-frame sustained check, was
**26624**.

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
