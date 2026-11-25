# GPU particle engine

PhysXEngine started as a CPU two-body gravity integrator, checked against the analytic
Kepler solution. These are my notes for turning it into a GPU particle engine with
real-time rendering.

Metis is where I prototype. What works here moves into Hyperion, a magnetic-confinement
fusion simulation (tokamak and stellarator). That end goal drives most of the choices
below, and it's the reason I'm not building a Barnes-Hut tree.

## Goals

- Real-time rendering at a sustained 144 Hz, never dipping below 120 Hz
  (frame budget 6.94 ms, hard ceiling 8.33 ms)
- 100k+ particles
- Physics on GPU
- An architecture that carries into Hyperion without a rewrite
- Keep the verified CPU accuracy path

## Measured outcome (Stage 1)

Measured after Stage 1, 1280x720, same RTX 2060. I changed the "never below 120 Hz"
goal to a p99.9 frame-time floor. The single worst frame isn't a stable statistic: run
long enough and you always find a worse one. So the gate is average >= 144 fps and
p99.9 >= 120 fps.

- Average alone hits 144 fps somewhere between 32768 and 35000 particles
  (32768 averages 169.3 fps; 35000 averages ~117-125 fps).
- The p99.9 floor holds through 26624 particles in every repeat, including a
  6000-frame run (worst p99.9: 133.8 fps, 11% margin). 26880 passed too, more
  thinly (123-126 fps), in two runs. 27136 fails (110.3 fps).
- 32768, the original target, fails p99.9 at 114.6 fps even though its average
  (169.3 fps) passes.
- Not 100k: direct-sum FP32 is compute-bound long before memory or interop matter.

Full sweep in `README.md`.

Not in scope: Barnes-Hut or FMM tree codes, inertial confinement physics
(radiation-hydrodynamics), multi-GPU, and anything other than Linux.

## Hardware

Measured on my machine, 2026-09-20.

| | |
|---|---|
| OS | Arch Linux, kernel 7.2.3, Wayland (KDE) |
| CPU | Intel i5-10600K, 6C/12T, AVX2 |
| RAM | 15 GB |
| GPU | RTX 2060 12 GB, compute capability 7.5 (Turing) |
| Driver | NVIDIA 610.57.04 |
| APIs | OpenGL 4.6, Vulkan 1.4.341, CUDA 13.3 |

### Direct sum, O(N^2)

```
        N      FP32 ms   FP32 fps      FP64 ms   FP64 fps   penalty
    16384        1.158      863.6       46.387       21.6      40.1x
    32768        4.618      216.5      187.244        5.3      40.5x
    65536       18.513       54.0      760.169        1.3      41.1x
   100000       43.427       23.0     1748.303        0.6      40.3x
```

Two things follow:

1. **FP64 is out on this GPU.** The penalty is 40x, worse than Turing's 32x spec
   ratio. At 144 Hz FP64 tops out around 3000 particles.
2. **Direct-sum FP32 tops out near 37,000 particles** at 144 Hz with one physics
   step per frame. Not 100k.
   *(That estimate was the physics kernel only: no rendering, no interop, no
   frame variance. Measured with rendering it's ~32768-35000 for average fps and
   26624-26880 with the p99.9 floor. See "Measured outcome" above.)*

### Barnes-Hut force-phase proxy (FP32)

```
        N      K           ms        fps
   100000    128        1.776      562.9
   100000    256        4.023      248.6
   100000    512        8.365      119.5
   250000    128       11.987       83.4
```

K is interactions per particle. These use fully random gathers, so they're a
pessimistic floor. A real Morton-sorted tree has better locality.

## Why not Barnes-Hut

Gravity and Coulomb are both 1/r^2, so a tree looks like it should carry over to
plasma. It doesn't, because plasma screens:

```
phi(r) ~ (q/r) * exp(-r / lambda_D)
```

Past a few Debye lengths the interaction is negligible. Barnes-Hut exists to compute
unscreened long-range far fields cheaply. In a plasma that far field is carried by
the fields, not by summed pair forces.

For tokamak parameters (n ~ 1e20 m^-3, T ~ 10 keV), lambda_D ~ 74 um with ~1.7e8
particles per Debye sphere. Weakly coupled, collective rather than collisional.
That's the regime particle-in-cell was made for, and it's why production fusion codes
(GENE, XGC, GTC, OSIRIS) are PIC or gyrokinetic, not tree codes.

PIC also scales better:

```
deposit charge to grid   O(N)
solve field on grid      O(M log M)    FFT / multigrid
gather field to particle O(N)
push particles           O(N)
```

Linear in particle count, so particle count stops being the limit.

## Architecture

Nothing below the model boundary knows which physics is running.

```
Renderer      SDL3 + OpenGL 4.6 + CUDA interop
Diagnostics   energy, momentum, field energy, stability assertions
──────────────────────────────────────────────────
IIntegrator   Euler | LeapfrogKDK | Boris
IForceModel   GravityCPU_FP64 | GravityGPU_FP32 | ElectrostaticPIC | EM-PIC
──────────────────────────────────────────────────
SimState      SoA particle buffers + optional grid
Core          math types, buffers
```

Interfaces take whole buffers, never single particles:

```cpp
void IForceModel::computeAccelerations(SimState&);
void IIntegrator::step(SimState&, IForceModel&, double dt);
```

One virtual call per step instead of per body. A per-particle virtual interface would
kill GPU throughput.

A different physics model later (MHD, hydro) is a new `IForceModel`, not a rewrite,
and can't slow down the tokamak path.

### Windowing and graphics

SDL3 (native Wayland) + OpenGL 4.6 core + CUDA-GL interop.

CUDA-GL buffer interop (`cudaGraphicsGLRegisterBuffer`) is zero-copy and a lot simpler
than CUDA-Vulkan external memory. The physics is already CUDA, so this is the short
path.

The only new vendored dependency is a `glad` loader, one file.

Rendering cost doesn't matter here: 100k point sprites is well under 1 ms on this GPU,
and 100k particles is 3.2 MB out of 12 GB.

## Data layout and precision

SoA throughout. `float4` packing coalesces well (benchmarked):

```
float4 pos_w[]    // x, y, z, mass-or-weight
float4 vel_q[]    // vx, vy, vz, charge
```

| Path | Precision | Why |
|---|---|---|
| GPU particles | FP32 | 40x FP64 penalty on Turing |
| CPU reference | FP64 | It's the validation oracle |
| Diagnostics accumulation | FP64 | Summing 100k FP32 energies loses the signal |

**PIC deposition:** charge deposition uses atomic adds, and FP32 atomics pick up real
error with many particles per cell. Fix with per-cell ordering or fixed-point
accumulation. Cheap to plan for now, painful to retrofit.

The old `struct Particle` is AoS and needs porting to SoA. The test for that port
already exists, see below.

## Validation

Every stage gets checked against an analytic result or the CPU oracle.

| Test | Checks | Expected |
|---|---|---|
| Kepler two-body | gravity, integrator | 4.836 mm error at dt=0.001 (measured) |
| GPU vs CPU at N=1024 | FP32 kernel correctness | bounded relative force error |
| Single particle in uniform B | Boris pusher | gyrofrequency, Larmor radius, exact |
| Langmuir oscillation | field solve | frequency equals plasma frequency |
| Two-stream instability | full PIC loop | growth rate vs linear theory |
| Energy / momentum | integrator | bounded, no drift |

The SoA port is proven by re-running Kepler: if the error moves off 4.836 mm at
dt=0.001, the port broke something.

Two-stream is the standard PIC acceptance test. When the loop is wrong it fails
loudly and specifically.

### Runtime stability checks

The classic PIC failure modes, asserted at runtime:

- `dx <= lambda_D`, otherwise numerical heating
- `omega_p * dt < 2`, plasma frequency resolution
- `c * dt < dx`, CFL, electromagnetic stage only

## Integrator

The current integrator is semi-implicit Euler, first order. Error converges as O(dt).

Stage 1 moves to **leapfrog KDK**: second-order symplectic, same memory, one extra
half-kick per step. Same bounded energy, error O(dt^2). At the large dt a 100k
simulation needs, that's the biggest accuracy gain for the cost.

Stage 3 swaps leapfrog for the **Boris pusher** once magnetic fields come in. Boris is
the standard magnetized-particle integrator and preserves gyro-radius exactly in a
uniform field.

## Stages

1. **Stage 1:** SDL3 + OpenGL + CUDA direct sum, leapfrog KDK, ~35k particles,
   checked against the CPU FP64 oracle. Proves the whole pipeline.
2. **Stage 2:** electrostatic PIC. Charge deposition, Poisson solve via cuFFT, field
   gather. O(N).
3. **Stage 3:** Boris pusher, magnetized particles. Tokamak physics starts here.
4. **Stage 4:** electromagnetic PIC on a Yee grid.

Gyrokinetics comes after Stage 4.

## Risks

| Risk | Plan |
|---|---|
| NVIDIA + Wayland + OpenGL on driver 610 | Try it first; fall back to `SDL_VIDEODRIVER=x11` via XWayland |
| No cmake installed | Install cmake, use ninja |
| Little free RAM during nvcc builds | Limit parallel compile jobs |
| FP32 not enough for some diagnostics | Accumulate diagnostics in FP64 |

## Still open

- Grid resolution and domain geometry for Stage 2, once Stage 1 is done
- Whether Hyperion's renderer needs toroidal geometry or Cartesian is enough for now

## Convergence, measured

Earth-Moon two-body, FP64 CPU path, error in the Moon's y position against the analytic
orbit at t=3600 s, softening 1e4 m.

| dt (s) | semi-implicit Euler (m) | leapfrog KDK (m) |
|---|---|---|
| 400 | 1941.511 | 0.001636505 |
| 200 | 970.7552 | 0.0003957748 |
| 100 | 485.3775 | 0.00008559227 |
| 50 | 242.6887 | 0.000008165836 |
| 25 | 121.3444 | 0.00001138449 |
| 10 | 48.53773 | 0.00001686811 |
| 1 | 4.853756 | 0.00001871586 |
| 0.1 | 0.48536 | 0.00001865625 |
| 0.01 | 0.04851437 | 0.00001168251 |

Euler halves its error every time dt halves: first order. Leapfrog drops by 4.1x and 4.6x
over the first two halvings, which is second order, then flattens between 1e-5 and 2e-5 m.
That's the FP64 round-off floor for this problem, not truncation error.
