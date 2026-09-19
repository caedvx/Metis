# Metis

Learning and prototyping ground for simulation engines and base ideas. Components
that prove out here get lifted into the projects that need them — primarily
[Hyperion](https://github.com/caedvx/Hyperion), a magnetic-confinement fusion
simulation.

## PhysXEngine

Gravitational N-body integrator, currently CPU and two-body, being extended into a
GPU particle engine with real-time rendering.

### Current state

Semi-implicit Euler integration with Plummer softening, verified against the analytic
Kepler solution:

| dt | position error at t=3600 s | ratio |
|---|---|---|
| 1.0 | 4.853756 m | — |
| 0.1 | 0.485360 m | 10.00 |
| 0.01 | 0.048514 m | 10.00 |
| 0.001 | 0.004836 m | 10.03 |

Clean first-order convergence. Over a full lunar orbit (2.36e9 steps at dt=0.001) all
timestep choices agree within 0.014 m and orbital radius holds to under 1.3 mm.

### Build

```sh
cd PhysXEngine
g++ -std=c++17 -O2 -Wall -Wextra main.cpp Particle.cpp -o simulation
./simulation
```

Simulation and print intervals are constants at the top of `main()`. Default
integrates at 1 ms and prints every 1 s of simulated time.

### Where it is going

See [the design spec](docs/design/01-gpu-engine.md)
for the full architecture: CUDA physics, SDL3 + OpenGL rendering, and the path to
Particle-In-Cell for plasma.
