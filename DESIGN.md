# Jineta — Design Document

> Living document. Update it at the end of each phase. It lives in the repo and in
> the claude.ai Project "Jineta", so every new chat there starts with full context.
> Section 8 says how Claude and the author work together.

## 1. What Jineta is

A **multi-agent air-defence simulator**: incoming aerial threats (one-way attack
drones, slow cruise missiles, hostile drones) versus a group of interceptor drones
that coordinate to neutralise them, over **real terrain**, with **realistic 6-DOF
physics**.

The goal is to **compare coordination strategies** (target assignment, centralised
vs. distributed command, robustness to comms loss or losing an interceptor) through
Monte Carlo analysis.

Name: *jineta* = historical Spanish light cavalry — small, agile units that
coordinated hit-and-run attacks. Team brand: **Ibero**.

Author: 3rd-year aerospace engineering student. First large software project, so
code should be clear and every concept explained before it is used.

## 2. Environment

- OS: Windows (two machines: laptop + desktop, synced via GitHub)
- Python **3.13** in a `.venv` virtual environment
- Editor: VS Code (extensions: Python, Pylance, Ruff, GitLens)
- Dependencies: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `pytest`, `ruff`
  (pinned in `requirements.txt`)
- Run tests from repo root: `pytest -v`
- Run examples from repo root: `python -m examples.<name>`
- Save all open files (Ctrl+S) before running tests: pytest runs what is on disk.

## 3. Conventions

- **Code in English** (names, comments, docstrings). Module names are themed.
- **Units: SI everywhere** (m, s, kg, rad, N). Degrees only at input/output edges.
- **Reference frame: NED** (North-East-Down): x North, y East, z Down. z points
  DOWN → altitude = −z, gravity is +g along z.
- **State vectors are NumPy arrays.**
  - Point mass: `[x, y, z, vx, vy, vz]`
  - Rotation only: `[qw, qx, qy, qz, p, q, r]`
  - Rigid body: `[x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]` (13 elements) —
    position and velocity in NED, quaternion BODY → NED, angular rate in body axes.
- **Quaternions:** `[w, x, y, z]` (scalar first), Hamilton product, unit norm.
  `q` rotates vectors BODY → NED: `v_ned = to_matrix(q) @ v_body`.
- **Body axes:** x forward (nose), y right wing, z down. Angular velocity
  `omega = [p, q, r]` is expressed in body axes: p = roll rate (about x), q = pitch
  rate (about y), r = yaw rate (about z). With a diagonal inertia tensor the body
  axes are the principal axes, and "axis 1, 2, 3" means x, y, z.
- **Reference point:** the body's position is its centre of mass, and the inertia
  tensor is expressed about the centre of mass in body axes.
- Euler angles (roll φ, pitch θ, yaw ψ, ZYX order) only for display/input —
  never as internal state (gimbal lock at θ = ±90°).
- Dynamics expose `derivative(t, state) -> d(state)/dt`; integrators know nothing
  about physics.
- Type hints and docstrings on every public function.
- Every physics module ships with tests against known analytic results.
- Git: pull when starting, commit + push when finishing. Small commits, clear
  English messages.

## 4. Architecture

```
jineta/
├── jineta/
│   ├── core/       # physics engine: integrators, point mass, rigid body
│   ├── campo/      # environment: ISA atmosphere, wind, gravity, real terrain
│   ├── jinete/     # vehicle models (multirotor, fixed wing, missile) + control
│   ├── atalaya/    # sensors (radar, EO, IMU) + Kalman filtering
│   ├── capitan/    # target assignment / threat prioritisation
│   └── escuadron/  # inter-unit coordination and communications
├── tests/
├── scenarios/      # mission definitions (YAML)
├── examples/       # runnable demo scripts
└── DESIGN.md
```

Simulation flow: `atalaya` detects → `capitan` decides → `escuadron` coordinates →
each `jinete` flies using `core` inside the `campo`.

**The simulator never draws.** It produces data (states + events over time).
Visualisation is a separate layer that reads that data:

1. Now: matplotlib, for debugging and analysis plots.
2. Multi-drone scenarios: export to **CesiumJS** (CZML) for a web 3D globe with real
   terrain and satellite imagery — the showcase/demo view.
3. Optional: Godot real-time viewer.

## 5. Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Integrators (Euler, RK4) | ✅ Done |
| 2 | Point mass: gravity + drag, simulation loop | ✅ Done |
| 3 | Rotation: quaternions, Euler's equations | ✅ Done |
| 4 | Full 6-DOF rigid body | ✅ Done |
| 5 | Environment: ISA atmosphere, wind | 🔄 Next |
| 6 | Vehicle models with real parameters + attitude control | ⏳ |
| 7 | Scenarios in YAML, multiple entities | ⏳ |
| 8 | Sensors + Kalman filter | ⏳ |
| 9 | Guidance (proportional navigation) | ⏳ |
| 10 | Coordination strategies (assignment, comms) | ⏳ |
| 11 | Real terrain + Cesium visualisation | ⏳ |
| 12 | Monte Carlo analysis, README, demo video | ⏳ |

## 6. Decision log

- Python + NumPy for the engine; performance work only when profiling shows a need.
- RK4 fixed-step as the default integrator.
- NED frame from day one, to avoid a later refactor.
- Simulation decoupled from visualisation.
- Gravity is constant (9.80665 m/s²): its change with altitude is ~0.3 % at 10 km,
  far below the uncertainty in vehicle aero parameters.
- Air density varies with altitude (~10 %/km) and matters. Dynamics receive a
  density model `altitude -> rho` from outside; the full ISA model lives in `campo`
  (phase 5). Until then, constant sea-level density is the default.
- Aerodynamics are NOT computed from 3D geometry at runtime (that would be CFD).
  3D models are visual only. Each vehicle carries precomputed coefficient models
  (CD, CL, CY, Cl, Cm, Cn as functions of alpha, beta, Mach, control deflections),
  from published data, DATCOM, XFLR5/AVL or offline CFD. Runtime: airspeed in body
  axes (via quaternion) → alpha, beta → interpolate → forces/moments. Multirotors
  use per-axis body drag + linear rotor drag.
- Interceptor drones vs. ballistic projectiles is physically unrealistic; threats
  are drones and slow cruise missiles. Unwinnable engagements should be reported as
  such.
- Rigid-body state mixes frames on purpose: position and velocity in NED, angular
  rate in body axes. With NED velocity the translational equation has no Coriolis
  term (−ω × v_b), and each equation is written in the frame where it is natural.
  Airspeed in body axes (for alpha, beta) is obtained later as `Rᵀ(v − wind)`
  (phase 6).
- `RigidBody` takes a `force_moment(t, state) -> (F_body, M_body)` callback. Gravity
  is NOT part of it: the body adds it in NED. Any coupling between attitude and
  translation (thrust direction, aerodynamics) lives in that callback, never in
  `derivative()`. The callback receives the full state, so forces can depend on
  attitude and angular rate.
- RK4 does not preserve the quaternion norm. `simulation.run` has an optional
  `post_step(y) -> y` hook applied after every step; `rigid_body.normalize_attitude`
  renormalises the quaternion. Without it the norm drifts (~3e-3 after 100 s at
  dt = 0.1 and ω ≈ 5 rad/s; ~3e-8 at dt = 0.01).
- Decoupling is tested in vacuum only (zero force model): rotation alone does not
  change the centre-of-mass motion. In air it does (attitude changes alpha/beta and
  hence drag and lift); that effect belongs to the aerodynamic force model and gets
  its own test in phase 6.
- `RotatingBody` (attitude only) is kept next to `RigidBody` for now although both
  use the same Euler equations. Revisit in phase 6 (possible shared function).

## 7. Current state

- `jineta/core/integrators.py` — `euler_step`, `rk4_step`, `Derivative` type alias.
  Tested against free fall.
- `jineta/core/point_mass.py` — `PointMass` dataclass (mass, drag_area, density
  model) with `derivative()`: constant gravity + quadratic drag
  `D = -½·rho·CdA·|v|·v`. Also `GRAVITY`, `DensityModel`, `sea_level_density`.
- `jineta/core/simulation.py` — `run(f, y0, dt, t_end, stop=None, post_step=None)`:
  fixed-step RK4 loop with optional early-stop condition and optional post-step hook
  (applied to the state after every step, before `stop` is evaluated); returns
  `(times, states)`.
- `jineta/core/quaternion.py` — `multiply`, `from_axis_angle`, `normalize`,
  `to_matrix`, `derivative(q, omega_body) = ½ q ⊗ [0, ω]`.
- `jineta/core/rotating_body.py` — `RotatingBody` dataclass (inertia 3x3, torque
  model) with Euler's equations `ω̇ = I⁻¹(M − ω × Iω)`; inverse inertia computed once
  in `__post_init__`. Also `TorqueModel`, `zero_torque`.
- `jineta/core/rigid_body.py` — `RigidBody` dataclass (mass, inertia 3x3,
  `force_moment` model) with the 13-element `derivative()`: body force rotated to NED
  with `R(q)` + gravity, plus quaternion kinematics and Euler's equations. Also
  `ForceMomentModel`, `zero_force_moment`, `normalize_attitude` (post-step hook).
- Tests (19 passing): integrators (2); point mass (4); rotation (5: 90° about z maps
  N→E, matrix is proper orthogonal, steady spin matches axis-angle, constant torque
  spins up linearly, torque-free motion conserves energy + NED angular momentum +
  quaternion norm); rigid body (8: no forces = exact free fall; quaternion norm
  drifts without the hook; the hook keeps the norm at 1; `post_step=None` changes
  nothing; thrust along +x_body follows yaw at 0°/90°/180° — 3 cases; free spin in
  vacuum does not change the centre-of-mass fall).
- Examples: `projectile_drag.py`, `density_comparison.py`, `dzhanibekov.py`
  (intermediate-axis instability reproduced), `tumbling_projectile.py` (a tumbling
  body in vacuum: centre of mass follows the exact parabola, deviation ~1e-11 m,
  while angular rates show the flip).
- Physics note (intermediate-axis instability, torque-free): spin Ω about the
  intermediate axis grows any tiny perturbation at rate
  `λ = Ω·sqrt((I2−I1)(I3−I2)/(I1·I3))`. The example perturbation (0.01 rad/s on p
  and r) is an arbitrary stand-in for real imperfections, not a physical constant.
  The flip time grows only with the logarithm of the perturbation size
  (~ln(10)/λ ≈ 2 s per 10× smaller for I = diag(1,2,3), Ω = 2); it shrinks with
  faster spin and larger λ. Exactly zero perturbation never flips. Spin about the
  smallest or largest axis is stable.
- Known loose ends (none block phase 5):
  1. Style: missing blank lines before top-level defs in `rigid_body.py` and
     `simulation.py`; import order/spacing in `tests/test_rigid_body.py`
     (`RigidBody , normalize_attitude`, `import pytest` after local imports);
     trailing whitespace (`test_point_mass.py:51`, `test_rigid_body.py:44`). Fix in
     one separate commit with `ruff format .` and `ruff check .`.
  2. The docstring of `run` does not describe `post_step` yet.
  3. After `stop` fires, the last sample overshoots the event (e.g. ~v·dt below
     ground). Interpolating the event time matters once miss distance is measured
     (phases 9–10).
  4. No validation of mass > 0 or of the inertia tensor (symmetric, positive
     definite). `inertia_inv` is computed once, so treat `inertia` as immutable.
  5. Optional: `pyproject.toml` with pytest and ruff configuration.
- Next: phase 5 — environment in `campo`:
  - ISA atmosphere: temperature, pressure, density and speed of sound vs. altitude
    (troposphere with −6.5 K/km lapse rate, isothermal layer above 11 km). Tests
    against standard values: sea level 288.15 K, 101 325 Pa, 1.225 kg/m³, 340.29 m/s;
    11 km 216.65 K, 22 632 Pa, 0.3639 kg/m³.
  - Wind: constant and altitude-profile models, returning a NED velocity.
  - Plug into the existing interfaces: `PointMass(density=isa_density)`, and airspeed
    `v − wind` for the force models of phase 6.

## 8. Working with Claude

- **Language:** replies in Spanish; code, comments and commits in English.
- **Teaching first:** explain every new concept (physics, maths or Python) before
  using it. The author writes or pastes the code, runs it locally and reports back.
  Visual explanations are welcome when a concept is hard to picture.
- **Small verified steps:** each step ends with tests passing (`pytest -v`) and a
  commit + push.
- **Accuracy:** Claude checks what it states against the real repo and by running
  the code (test counts, labels, numbers, behaviour) before handing it over. When
  something could not be verified, it says so.
- **End of each phase:** update this document (roadmap, conventions, decision log,
  current state) and hand it over to be committed.
- **Session hygiene:** Claude must tell the author when to start a new chat —
  at the end of a phase block, or when the conversation is getting long enough that
  older details risk being lost. Before switching, make sure this document is up to
  date. A new chat starts inside the "Jineta" Project with:
  "Seguimos con Jineta, fase N: <topic>".
