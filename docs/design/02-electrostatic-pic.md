# Electrostatic particle-in-cell

Stage 2. Follows [the GPU particle engine](01-gpu-engine.md), which is merged.

Stage 1 gave me a validated GPU particle engine: structure-of-arrays host state behind
whole-buffer interfaces, a second-order symplectic leapfrog, an FP64 CPU oracle, a CUDA
FP32 direct-sum force model, and an SDL3 + OpenGL 4.6 renderer with zero-copy CUDA-GL
interop.

Stage 2 replaces the O(N^2) direct sum with particle-in-cell. It's the first stage that's
actually plasma physics, so it's the first real step toward Hyperion.

## Goals

- A correct 3D electrostatic PIC loop: deposit, solve, gather, push
- Two species, mobile ions
- Pass three analytic tests: Langmuir oscillation, two-stream instability,
  ion-acoustic dispersion
- Target: 128^3 grid, ~20 particles per cell per species (~84M particles)
- O(N) scaling, so particle count stops being the limit

Not in this stage: magnetic fields and the Boris pusher (Stage 3), electromagnetic PIC on
a Yee grid (Stage 4), gyrokinetics, non-periodic boundaries (conducting walls, toroidal
geometry), implicit or sub-cycled time stepping, and interactive framerate.

## Not interactive

Stage 1 chased a 144 Hz display budget. Stage 2 doesn't, and the arithmetic says why. At
84M particles the particle memory traffic per step alone is:

```
gather E    read pos          1.34 GB
push        read+write vel    2.69 GB
            read+write pos    2.69 GB
deposit     read pos          1.34 GB
                              --------
                              ~8.1 GB/step
```

At the RTX 2060's ~270 GB/s realistic bandwidth that's a ~30 ms/step floor before
deposition atomics or the FFT, so about 20-30 steps/sec. Validation runs of a few hundred
to a few thousand steps take tens of seconds to minutes.

So Stage 2 is a batch simulation with visualization, not an interactive viewer.

## Decisions

### 3D

Matches the engine's existing 3D state and renderer, and it's what a tokamak code needs in
the end. The cost: grid memory scales as N^3, and the validation tests prove the same
physics 1D or 2D would have proven for less work.

### Normalized units, not SI

I don't have a choice here. SI electrostatics in FP32 breaks:

```
e^2 = (1.602e-19)^2 = 2.566e-38
FP32 smallest normal = 1.18e-38
```

Charge squared sits only a factor of two above FP32's smallest normal number. Any SI
expression that multiplies it by something small, like a Debye-scale length, drops into
denormals or to zero on the device. Stage 1 got away with SI because gravitational masses
(~1e24) and distances (~1e8) are large. Plasma is the opposite.

Lengths in electron Debye lengths `lambda_De`, time in `1/omega_pe`, charge in `e`, mass in
`m_e`, velocity in the electron thermal speed `v_te = lambda_De * omega_pe`:

```
lambda_De^2 = eps0 * k_B * T_e / (n0 * e^2)
omega_pe^2  = n0 * e^2 / (eps0 * m_e)
=> lambda_De^2 * omega_pe^2 = k_B * T_e / m_e = v_te^2
```

Poisson and the equation of motion collapse to:

```
grad^2 phi = -rho              rho = net charge density in units of (e * n0)
dv/dt      = (q_s / m_s) E     q in units of e, m in units of m_e
```

Electrons have `q/m = -1`, ions `q/m = +1/massRatio`. Everything is O(1).

The FP64 CPU reference can convert to SI for sanity checks. The device path is normalized
throughout.

### Electrons and mobile ions

Mobile ions are more complete physics, and without them there's no ion-acoustic test.

The cost is the timescale gap between the species. I use one shared timestep for both, with
`massRatio` as a runtime parameter. No sub-cycling and no implicit methods in this stage.
Sub-cycling breaks leapfrog's time-centring, which is what makes energy conservation
provable, and conservation checks are how Stage 1 caught its real bugs. Implicit PIC is a
research project, not a stage.

The penalty is smaller than the raw ratio suggests:

```
omega_pi / omega_pe = 1 / sqrt(massRatio)   = 1/42.8 at the true proton ratio
```

Ion physics needs ~43x longer runs, not 1836x. Reduced mass ratios (100-400) are standard
published PIC practice and keep the qualitative physics. Tests default to a reduced ratio;
`massRatio = 1836` works, it just takes longer.

### Triply periodic boundaries

The FFT Poisson solve assumes periodicity, and all three validation tests are periodic
problems anyway. Other boundaries need a different solver (multigrid or a radial basis) and
come later.

### 128^3 grid, ~20 ppc per species

Field noise goes as `1/sqrt(ppc)`. At 20 ppc the two-stream growth-rate fit is clean enough
that a real bug doesn't hide in the noise.

```
Grid 128^3                            = 2,097,152 cells
Particles (84M x float4 pos + vel)    = 2.69 GB
Grid rho, phi, Ex, Ey, Ez (FP32)      = 42 MB
R2C spectrum (65 x 128 x 128 complex) = 8.5 MB
cuFFT workspace                       = ~17 MB
Morton sort buffers (Phase 2)         = ~1.0 GB
                                        -------
                                        ~3.8 GB of 12 GB
```

Memory isn't the constraint. Bandwidth and deposition atomics are.

## Architecture

### Per-species arrays

```cpp
struct SpeciesData
{
    float4* pos;      // xyz position, w unused
    float4* vel;      // xyz velocity, w unused
    int     count;
    float   charge;   // units of e:   -1 electrons, +1 ions
    float   mass;     // units of m_e:  1 electrons, massRatio ions
};
```

Two instances. Deposition runs once per species into one shared `rho` grid.

Better than a per-particle species id: the kernel never branches on charge or mass (they're
kernel arguments), and the two species can have different counts.

`PlasmaState` owns the species and the grid metadata. It sits next to Stage 1's `SimState`
rather than changing it, so the gravity path keeps working.

### The loop

```
deposit   CIC, 8 neighbours per particle, atomic adds -> rho       O(N)
solve     R2C FFT -> phi_k = rho_k / |k|^2, k=0 -> 0
                  -> E_k = -i * k * phi_k
                  -> batched C2R -> Ex, Ey, Ez                     O(M log M)
gather    same CIC weights -> E at each particle                   O(N)
push      leapfrog KDK, shared dt, per-species q/m                 O(N)
```

### Spectral E instead of differencing phi

`E_k = -i * k * phi_k` is an exact derivative with no stencil error. It also keeps scatter
and gather adjoint, which is what makes the self-force vanish and total momentum conserve.
The validation tests check both.

Cost: three inverse transforms instead of one, batched into a single `cufftExecC2R` call.

### Zero the k=0 mode

It's the undefined mean potential, and net charge in a periodic box is unphysical. Get it
wrong and you get a slow global drift that looks like a bug somewhere else. Test 4 checks it.

### Deposition

Phase 1 uses plain global atomics. Phase 2 adds Morton-ordered particles with shared-memory
accumulation, because at 128^3:

```
84M particles x 8 CIC neighbours = 672M atomic adds per component per step
```

spread over 2.1M cells, which serialises badly. With sorted particles a block can accumulate
in shared memory and issue one global atomic per cell.

Particles move well under a cell per step, so re-sort every ~10-20 steps. Sorting 2.69 GB of
particle data is expensive itself and can't happen every step.

## Validation

Most of Stage 1's gates were single-number comparisons, and three of them turned out not to
test anything: a `std::max` that swallowed NaN, a comparison target pre-filled with the
oracle's own answer, and 1-pixel particles that made a fill-rate benchmark meaningless. PIC
is better here, because nearly every piece has a closed-form check that needs no reference
implementation.

| # | Test | Basis | Catches |
|---|---|---|---|
| 1 | `sum(rho * dV) == sum(q)` | exact conservation | dropped neighbours, wrong weights, atomic races |
| 2 | One particle at a known offset -> 8 known CIC weights | closed form | weight algebra |
| 3 | `rho = A sin(k.x)` -> `phi = A sin(k.x)/|k|^2` | analytic | FFT layout, k-vector indexing, normalization |
| 4 | Uniform `rho` -> `phi == 0` | k=0 handling | global drift |
| 5 | Single particle alone -> zero force | adjoint deposit/gather | asymmetric weights |
| 6 | Total momentum change per step == 0 | adjointness | gather != scatter |
| 7 | Langmuir: `omega^2 = 1 + 3k^2` (Bohm-Gross, normalized) | analytic | the whole loop |
| 8 | Two-stream: growth rate vs linear dispersion | analytic | the whole loop, loudly |
| 9 | Ion-acoustic: `omega = k * c_s`, `c_s = 1/sqrt(massRatio)` | analytic, needs mobile ions | ion dynamics |
| 10 | Total energy (kinetic + field) bounded | symplectic | integrator |

1-6 are millisecond unit tests and have to pass before 7-9 are worth running. Test 3 matters
most: a single Fourier mode has a closed-form answer, so the Poisson solver gets checked
against maths instead of another implementation that could share its bugs.

### Two-stream test

The growth rate gets computed, not looked up. For two cold counter-streaming electron beams of
equal density at `+/- v0`:

```
1 = (omega_pe^2 / 2) * [ 1/(omega - k*v0)^2 + 1/(omega + k*v0)^2 ]
```

The test solves this numerically for the `k` the grid resolves and compares against the
measured exponential growth of field energy.

### Runtime checks

In normalized units these are simple:

- `dx <= 1` (grid resolves the Debye length, otherwise numerical heating)
- `dt < 2` (plasma-frequency stability, 0.1-0.2 in practice)

The ion timescale isn't an extra stability limit: `omega_pi < omega_pe`, so any `dt` stable
for electrons is stable for ions. The ions constrain run length, not step size. Ion-acoustic
physics needs `sqrt(massRatio)` times more steps, so that test asserts it ran at least one
full period, or it's fitting noise.

### Precision (same as Stage 1)

- Device particle and field arrays: FP32
- CPU reference and all diagnostic accumulation: FP64
- Field energy over 2.1M cells loses the signal in FP32, so accumulate in FP64

## Phases

1. **Correctness at small scale.** 64^3 grid, ~10M particles, plain global atomics. Build
   the full loop and pass tests 1-10. All the debugging happens here, where a run takes
   seconds.
2. **Scale.** 128^3, ~84M particles. Morton sorting and shared-memory deposition as a
   measured optimisation, with tests 1-10 still passing.
3. **Visualization.** Field slice planes (rho or phi, colour-mapped) with a subsampled
   particle overlay. 84M point sprites isn't drawable, and the field is the better view
   anyway.

Why not go straight to full scale: Stage 1 took five rounds of fixes on a 150-line CUDA
kernel and had three tests passing while measuring nothing. Debugging sorted, shared-memory,
two-species deposition at 84M particles, where the pass condition is a growth-rate fit, is a
lot harder. Phase 1 gets the physics right where a wrong answer is cheap to find.

## What carries over

| Component | Reuse |
|---|---|
| SDL3 window, GL 4.6 context, input | as is |
| CUDA-GL zero-copy interop | as is (Phase 3) |
| `LeapfrogKdk` KDK sequence | ports directly, only q/m differs |
| Diagnostics, FP64 accumulation | as is |
| `SimState`, `IForceModel`, `IIntegrator` | untouched, gravity path keeps working |
| Checking against an oracle | extended with analytic tests |

## Risks

| Risk | Plan |
|---|---|
| Deposition atomics serialise at 128^3 | That's what Phase 2 is for; Phase 1 gets correctness first |
| Growth-rate fits too noisy to show a bug | 20 ppc; tests 1-6 gate 7-9 |
| cuFFT R2C index layout misread | test 3 checks it against a closed form |
| FP32 not enough for field energy | FP64 accumulation |
| massRatio 1836 makes ion tests too slow | it's a parameter; tests use a reduced ratio |
| Morton sort buffers use too much memory | sort indices and gather, or sort in chunks; measure first |

## Still open

- Phase 3: slice planes or volume rendering. Decide once Phase 1 has something to look at.
- Phase 2's sort: reorder particles directly (better locality, 2.69 GB moved) or sort an
  index array and gather (cheaper sort, worse locality). Measure it.

### From Stage 1

- The p99.9 frame dips at 32768 particles were never root-caused. My guess is the per-frame
  `cudaGraphicsMapResources` / `UnmapResources` sync. A double-buffered interop VBO would
  test that in ~60 lines and show what Vulkan's explicit semaphores could get back. Worth
  doing before Phase 3.
- The 26624-particle gate boundary needs re-checking on an idle GPU. That's what
  `metis_bench_noio` is for.
