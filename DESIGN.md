# Jineta — Design Document

> Living document. Update it at the end of each phase. Paste it at the start of
> any new chat with Claude to restore full project context.

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

## 3. Conventions

- **Code in English** (names, comments, docstrings). Module names are themed.
- **Units: SI everywhere** (m, s, kg, rad, N). Degrees only at input/output edges.
- **Reference frame: NED** (North-East-Down). z points DOWN → altitude = −z,
  gravity is +g along z.
- **State vectors are NumPy arrays.**
  - Point mass: `[x, y, z, vx, vy, vz]`
  - Rigid body (planned): position, velocity, attitude quaternion, angular rate
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
| 3 | Rotation: quaternions, Euler's equations | 🔄 Next |
| 4 | Full 6-DOF rigid body | ⏳ |
| 5 | Environment: ISA atmosphere, wind | ⏳ |
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
- Interceptor drones vs. ballistic projectiles is physically unrealistic; threats
  are drones and slow cruise missiles. Unwinnable engagements should be reported as
  such.

## 7. Current state

- `jineta/core/integrators.py` — `euler_step`, `rk4_step`, `Derivative` type alias.
  Tested against free fall.
- `jineta/core/point_mass.py` — `PointMass` dataclass (mass, drag_area, density
  model) with `derivative()`: constant gravity + quadratic drag
  `D = -½·rho·CdA·|v|·v`. Also `GRAVITY`, `DensityModel`, `sea_level_density`.
- `jineta/core/simulation.py` — `run(f, y0, dt, t_end, stop=None)`: fixed-step RK4
  loop with optional early-stop condition; returns `(times, states)`.
- Tests (6 passing): integrators (RK4 exact on free fall, Euler 1st-order
  convergence); point mass (vacuum parabola, terminal velocity, drag reduces range,
  thinner air at altitude increases range).
- Examples: `projectile_drag.py` (vacuum vs drag), `density_comparison.py`
  (constant vs exponential atmosphere).
- Next: phase 3 — attitude with quaternions, Euler's rotation equations, tests
  (torque-free rotation, conservation of angular momentum).