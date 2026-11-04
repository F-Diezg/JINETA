# Jineta — Design Document

> Living document. Update it at the end of each phase. It lives in the repo and in
> the claude.ai Project "Jineta", so every new chat there starts with full context.
> Section 8 says how Claude and the author work together.

## 1. What Jineta is

A **realistic simulator to train drones for indoor flight**: a 6-DOF physics engine,
indoor environments (rooms, gates, obstacles) and a training loop in which a drone
learns a task with artificial intelligence (reinforcement learning), so that what it
learns can later be flown by the real vehicle.

**MVP:** a quadrotor (the Iris profile) learns to fly an **indoor obstacle course in
the shortest possible time**: take off from a start point, cross a sequence of gates
in order without touching a wall, an obstacle or the floor, and finish at the last
gate. The time is measured against a classical baseline (a hand-written controller
on the same course), so "the AI flies it well" is a number, not an impression. The
policy is designed from the start to be deployable: its output is what a real
PX4 autopilot accepts from a companion computer (collective thrust and body rates,
see section 6), not a command only the simulator understands.

**Final goal:** any user builds the environment and the drone profile they need for
their purpose (a course, a warehouse, an inspection route; a racing quadrotor, a
heavier camera drone) and trains a drone for it, with the confidence that the
simulated vehicle behaves like the real one.

**Where it stands (end of phase 7):** the physics engine is built and validated
(phases 1–6c): integrators, rigid body, atmosphere and wind, a quadrotor with its
cascaded autopilot, a fixed-wing aircraft with its own autopilot, and a Kalman
filter. There is no indoor world, no training loop and no viewer yet. The first
measured limit is speed: the Iris with its autopilot simulates at ~10 times real
time on one core, far too slow to train (section 5, phase 9).

Name: *Jineta* is a working name until the project has its own.

Author: 3rd-year aerospace engineering student. First large software project, so
code should be clear and every concept explained before it is used.

## 2. Environment

- OS: Windows (two machines: laptop + desktop, synced via GitHub)
- Python **3.13** in a `.venv` virtual environment
- Editor: VS Code (extensions: Python, Pylance, Ruff, GitLens)
- Dependencies: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `pytest`, `pytest-xdist`,
  `ruff` (pinned in `requirements.txt`). After pulling on the other computer, run
  `pip install -r requirements.txt` when `requirements.txt` changed: `git pull`
  brings the list, not the packages. `pyyaml` is not used by the code today; the
  course files of phase 8 will use it.
- Run tests from repo root: `pytest -n auto` (pytest-xdist spreads the tests over
  every core) or `pytest -v` (one by one, every name listed); one file:
  `pytest tests\test_trim.py`; by name: `pytest -k orbit`; the last failures:
  `pytest --lf`.
- `pyproject.toml` configures pytest (`testpaths`) and ruff (line length 88, rules
  E, F, I, UP, B, RUF). VS Code's Ruff extension reads it, so the editor shows the
  same warnings as `ruff check .`
- Run examples from repo root: `python -m examples.<name>`
- Save all open files (Ctrl+S) before running tests: pytest runs what is on disk.
- Before every commit run `ruff format .` (not only `--check`): it adds the final
  newline and fixes blank lines that are easy to lose when pasting code. In VS
  Code, "Files: Insert Final Newline" and format-on-save with the Ruff formatter
  do the same automatically.
- The full test run, in one process, takes ~47 s (485 tests, in Claude's workspace
  at phase 8, step 4); the closed-loop fixed-wing tests are the slowest (the
  orbit hold, 5 s). `pytest -n auto` divides that by the number of cores.
- `scipy` is used for the trim (`scipy.optimize.root`) and in the Kalman tests
  (`scipy.stats.chi2`).
- Regenerate dependencies with `pip freeze | Set-Content -Encoding ascii
  requirements.txt`. PowerShell's `>` writes UTF-16, and `Out-File -Encoding utf8`
  in Windows PowerShell 5 adds a BOM (an invisible mark at the start of the file;
  pip tolerates it, but it does not belong there). Package names are ASCII.
- A block of commands pasted into PowerShell does not stop when one line fails:
  the next lines still run. So the checks come first, and the commit only once
  they passed.
- `git rm` removes tracked files but leaves the folders that only held Python's
  `__pycache__` (untracked); remove those by hand, or a deleted package keeps
  appearing in the file explorer.

## 3. Conventions

- **Code in English** (names, comments, docstrings). Module names are themed.
- **Units: SI everywhere** (m, s, kg, rad, N). Degrees only at input/output edges.
- **Reference frame: NED** (North-East-Down): x North, y East, z Down. z points
  DOWN → altitude = −z, gravity is +g along z. Indoors "North" is just the x axis
  of the building (the room's own reference), not geographic north.
- **State vectors are NumPy arrays.**
  - Point mass: `[x, y, z, vx, vy, vz]`
  - Rotation only: `[qw, qx, qy, qz, p, q, r]`
  - Rigid body: `[x, y, z, vx, vy, vz, qw, qx, qy, qz, p, q, r]` (13 elements) —
    position and velocity in NED, quaternion BODY → NED, angular rate in body axes.
- **Quaternions:** `[w, x, y, z]` (scalar first), Hamilton product, unit norm.
  `q` rotates vectors BODY → NED: `v_ned = to_matrix(q) @ v_body`.
- **Body axes:** x forward (nose), y right, z down. Angular velocity
  `omega = [p, q, r]` is expressed in body axes: p = roll rate (about x), q = pitch
  rate (about y), r = yaw rate (about z). With a diagonal inertia tensor the body
  axes are the principal axes, and "axis 1, 2, 3" means x, y, z. These are FRD axes
  (forward-right-down). Sources in FLU (Gazebo, ROS: x forward, y left, z up) are
  converted with `jinete.profiles.flu_to_frd` (flip y and z).
- **Reference point:** the body's position is its centre of mass, and the inertia
  tensor is expressed about the centre of mass in body axes.
- Euler angles (roll φ, pitch θ, yaw ψ, ZYX order) only for display, input and the
  controllers' angle errors — never as internal state (gimbal lock at θ = ±90°):
  `quaternion.from_euler(roll, pitch, yaw)` and `quaternion.to_euler(q)`.
- **Environment models are callables** (plain functions or dataclasses with
  `__call__`), so any of them can be swapped or combined:
  - density: `altitude [m] -> rho [kg/m³]` (`DensityModel`)
  - wind: `(t [s], position NED [m]) -> wind velocity NED [m/s]` (`WindModel`)
  - these aliases, `GRAVITY` and the neutral defaults live in
    `core/environment.py`. `core` never imports `campo`; `campo` and `jinete` may
    import `core`.
  - models that only depend on height read the altitude as `-position[2]`.
- **Wind vector convention:** the NED wind vector points where the air goes TO.
  Weather reports give where it comes FROM, clockwise from north;
  `meteorological_wind(speed, direction_from_deg)` converts. An updraft has a
  negative down component. Indoors the default is still air (`no_wind`).
- **Airspeed** is the velocity relative to the air, `v − wind`. Aerodynamic forces
  use airspeed; the state keeps the ground velocity. In body axes:
  `jinete.aero.body_airspeed(t, state, wind) = R(q)ᵀ (v − wind)`, components u
  (nose), v (right), w (down); w > 0 means positive angle of attack.
- **Vehicle profiles are data, not code.** Each vehicle is a frozen (immutable)
  parameter dataclass (e.g. `MultirotorParams`), one constant per vehicle in
  `jinete/profiles.py`, with a comment citing where every number comes from. Models
  never know which vehicle they fly. The controller gains live next to the
  physical parameters of each profile, not inside them. Files that describe a
  flight name a profile (`MULTIROTOR_PROFILES`, `FIXED_WING_PROFILES`); the
  numbers stay in code next to their sources. A user-built profile (phase 15) is
  the same thing written by someone else.
- **Vehicle interface:** a vehicle contains a `RigidBody` and hands it its own
  `force_moment` (vehicles do not subclass `RigidBody`). It exposes
  `derivative(t, state)` and is commanded through attributes that a controller sets
  in `pre_step` (multirotor: `rotor_speed_command`, one value per rotor [rad/s]).
- **Rotor spin:** `+1` = counter-clockwise seen from above (PX4/Gazebo "ccw"); the
  body reacts the other way, so a `+1` rotor yaws the body right (positive r).
- **Collisions:** a vehicle's collision shape is its `Hitbox` (spheres in body
  axes, profile data like the rotors); walls, ceiling and obstacles are checked
  with it, the floor with the landing gear. `CollisionMonitor` is a `stop` for
  `simulation.run`; it reports the first contact (`Collision`: kind "room",
  "obstacle" or "floor", obstacle index, interpolated time and position). It takes
  the hitbox and the gear as arrays, so `campo` still only imports `core`.
- **Simulation step order:** `pre_step(t, y)` (controllers) → one RK4 step →
  `post_step(y)` (e.g. renormalise the quaternion) → `stop(t, y)`.
- **Controllers** are dataclasses with `update(...)`, run once per step from
  `pre_step`; the ones with memory (integrals) have `reset()`. Per-axis gains are
  3-vectors: `[roll, pitch, yaw]` for the rate and attitude loops, `[north, east,
  down]` for the velocity loop. Gains are expressed so that they do not depend on
  the vehicle (angular acceleration per rate error, rate per attitude error,
  acceleration per velocity error). All the gains of one vehicle live in a
  `MultirotorGains` next to its profile (`IRIS_GAINS`).
- **Multirotor autopilot interface:** `Autopilot(vehicle, gains, dt)` is a
  callable; pass it as `pre_step`. Whoever flies the vehicle sets
  `autopilot.velocity_command` (NED [m/s]) and `autopilot.yaw_command` ([rad]).
  The rate loop alone (`RateController.update(rate_command, thrust_command,
  state)`) is the entry point for a policy that commands thrust and body rates.
- **Fixed-wing vehicle interface:** `FixedWing(params, propulsion=None, density,
  wind)`, built like `Multirotor` (holds a `RigidBody`, `derivative(t, state)`).
  Commanded through `surfaces` (`ControlSurfaces`, frozen: assign a new one) and
  `throttle` in [0, 1], both starting at neutral / 0. `propulsion=None` is a
  glider, i.e. "engine off" (throttle 0 with a propeller is not: it windmills and
  brakes). Signs as in the source: elevator > 0 trailing edge down (nose down),
  aileron > 0 rolls right, rudder > 0 yaws the nose left.
- **Trim:** `trim_level_flight(aircraft, speed, altitude, course)` returns a
  `Trim(state, surfaces, throttle, alpha, beta)` for wings-level flight (φ = 0) at
  that airspeed, and leaves the aircraft commanded with it.
- **Fixed-wing autopilot interface:** `FixedWingAutopilot(vehicle, trim, design,
  dt)` is a callable; pass it as `pre_step`. Whoever flies the aircraft sets
  `course_command` (direction of the ground velocity, clockwise from North [rad]),
  `altitude_command` [m] and `airspeed_command` [m/s]; they start at the trim. The
  design choices (limits, bandwidths, damping) are profile data
  (`SMALL_FIXED_WING_AUTOPILOT`, an `AutopilotDesign`); the gains are computed from
  them at the trim by `design_gains`. It runs at 100 Hz like the Iris.
- **Estimation:** the Kalman filter is a set of plain functions
  (`atalaya/kalman.py`) that take matrices and return new arrays; whoever uses it
  holds the state and the time. Every random process takes a seeded
  `numpy.random.default_rng`, so a run repeats exactly on any computer.
- Parameter classes validate their inputs in `__post_init__` and raise
  `ValueError` with a clear message. Inertia tensors go through
  `core.inertia.inertia_tensor` (symmetric, positive definite, triangle inequality)
  and are stored read-only.
- Use `zip(..., strict=True)` so sequences of different length fail loudly instead
  of being silently truncated (ruff B905).
- Dynamics expose `derivative(t, state) -> d(state)/dt`; integrators know nothing
  about physics.
- **The simulator never draws.** It produces data (states and events over time);
  visualisation is a separate layer that reads that data.
- Type hints and docstrings on every public function.
- Every physics module ships with tests against known analytic results.
- Git: pull when starting, commit + push when finishing. Small commits, clear
  English messages.

## 4. Architecture

Today:

```
jineta/
├── jineta/
│   ├── core/       # physics engine: integrators, point mass, rotating and rigid
│   │               #   body, quaternions, inertia, environment interfaces, run loop
│   ├── campo/      # environment: atmosphere.py (ISA), wind.py, indoor.py (room
│   │               #   and obstacles by signed distance), collision.py (collisions
│   │               #   in flight)
│   ├── jinete/     # vehicles: aero, multirotor, control (multirotor autopilot),
│   │               #   contact (landing gear on the floor), hitbox (collision
│   │               #   spheres), fixed_wing, propulsion, trim, stability,
│   │               #   fixed_wing_control, profiles
│   └── atalaya/    # estimation: kalman.py (later: onboard sensors)
├── tests/
├── examples/       # runnable scripts (python -m examples.<name>)
├── pyproject.toml  # pytest + ruff configuration
├── DESIGN.md
└── README.md
```

Planned (each package arrives with its phase; names are proposals, settled when the
phase starts):

- `campo/` grows the **indoor world** (phase 8): room geometry, gates, obstacles,
  collision checks, the course file format.
- a **fast batch engine** (phase 9): the same physics for N drones at once on
  arrays, checked against the reference engine in `core` + `jinete`, which stays as
  the truth.
- a **training layer** (phase 10–11): the environment interface (observation,
  action, reward, end of episode), the classical baseline and the training scripts.
- `atalaya/` grows **onboard sensors** (IMU, position from a motion-capture system
  or a camera) for the realism phase (12).
- a **visualisation layer** outside `jineta/` (phase 13), which reads recorded
  flights and never runs the physics.

Dependency rule: `core` imports nothing of the project; `campo` and `jinete` import
`core`; layers that compose others (world, training) sit on top and are never
imported by the engine.

## 5. Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Integrators (Euler, RK4) | ✅ Done |
| 2 | Point mass: gravity + drag, simulation loop | ✅ Done |
| 3 | Rotation: quaternions, Euler's equations | ✅ Done |
| 4 | Full 6-DOF rigid body | ✅ Done |
| 5 | Environment: ISA atmosphere, wind | ✅ Done |
| 6a | Quadrotor (Iris profile) + cascaded attitude/velocity control | ✅ Done |
| 6b | Fixed-wing model (aero coefficients, propeller, trim, modes, autopilot) | ✅ Done |
| 6c | Estimation groundwork: linear Kalman filter | ✅ Done |
| 7 | Project goal, repo reduced to the engine, new README and DESIGN | ✅ Done |
| 8 | Indoor world: room, floor, gates, obstacles, collisions; course files | ⏳ |
| 9 | Fast batch engine (many drones at once), checked against the reference | ⏳ |
| 10 | Training environment (observation, action, reward) + classical baseline | ⏳ |
| 11 | **MVP:** the Iris learns the course (reinforcement learning) | ⏳ |
| 12 | Realism for the real drone: motor lag, ground effect, sensor noise, latency, randomised parameters | ⏳ |
| 13 | 3D viewer of recorded flights | ⏳ |
| 14 | Towards the real drone: its profile, PX4 software-in-the-loop, offboard interface | ⏳ |
| 15 | Builder: any user defines the environment and the drone profile | ⏳ |
| 16 | Final README, demo video | ⏳ |

Each phase's step plan is drawn up with the author at its start and written here.
What each phase is meant to deliver:

- **8 — Indoor world.** A room is a box (floor, walls, ceiling); a gate is a frame
  the drone must pass through, in order; an obstacle is a simple solid (box,
  cylinder). Collisions are checked against the drone as a sphere (its rotor
  diagonal) first. A gate is crossed when the path between two samples crosses the
  gate's plane inside its opening, interpolated inside the step. The course is a
  YAML file (start, gates, obstacles, room), strict like every input file. A
  matplotlib figure shows the course and a flight. Tests: crossings and collisions
  against hand-computed geometry.
- **9 — Fast batch engine.** Training needs millions of steps: the reference
  engine does ~1 000 steps/s (Iris + autopilot at 100 Hz, ~10× real time on one
  core, measured at the end of phase 7); Swift, the drone that beat human racing
  champions, trained on the equivalent of a month of flight in under an hour
  (over 700× real time). The batch engine flies N drones as one array operation
  (NumPy first), and must agree with the reference engine to rounding (1e-9) for
  every drone. The speed-up is measured, not assumed.
- **10 — Training environment.** The standard reinforcement-learning interface
  (Gymnasium: `reset`, `step`): observation (first the true state, relative
  position of the next gates), action (collective thrust + body rates, executed by
  the existing rate loop), reward (progress along the course, a penalty for a
  crash, a bonus per gate), end of episode (crash, last gate, time limit). The
  classical baseline flies the same course with the existing velocity autopilot
  through the gate centres, and gives the time to beat.
- **11 — MVP.** Train a policy (PPO is the usual first choice) on one course, then
  on courses with randomised gates; compare lap time and success rate with the
  baseline. The library (e.g. Stable-Baselines3, which brings PyTorch) is chosen
  at the start of the phase.
- **12 — Realism.** What separates a policy that works in simulation from one
  that works in a real room: motor time constants, ground and ceiling effect,
  battery sag, sensor noise and delay, and the vehicle's parameters randomised
  during training so the policy does not depend on numbers that are only
  approximately right.
- **14 — Towards the real drone.** The 3DR Iris is long discontinued and too big
  and slow for a racing course (1.5 kg, thrust-to-weight 1.9), so the real vehicle
  will be a current PX4 quadrotor (or a smaller research platform), added as a
  profile from its own measured data. The policy is flown first in PX4
  software-in-the-loop, then on the vehicle. A real drone indoors has no GPS: it
  needs its position from a motion-capture system or onboard vision.
- **15 — Builder.** Environments and profiles written by any user, validated with
  clear errors, from files first; a graphical editor only if it earns its place.

**Phase 7 — done.** The project's goal was set (indoor training simulator, MVP
above); the repository was reduced to the physics engine (core, environment,
vehicles, Kalman filter) and its examples; README, roadmap and this document were
rewritten. Tests: 346, ruff clean.

**Phase 8 — in progress.** Plan agreed with the author at the start of the phase
(each step: tests green, ruff clean, commit + push). Decided with the author: the
drone starts each lap landed on the floor and takes off (so the phase brings a
ground contact model), and gates are rectangular or circular from the start.
1. ✅ **Geometry** (`jineta/campo/indoor.py`): the room (a box: floor, walls,
   ceiling), obstacles (boxes with a yaw, vertical cylinders), all through their
   signed distance, for one point or many at once. Tests: distances by hand (3-4-5
   and 2-3-6-7 triangles); the yaw is a heading (checked at 30°, where its sense
   shows); a distance never changes faster than the point moves; outside a convex
   shape its gradient has length 1.
2. ✅ **Ground contact** (`jineta/jinete/contact.py`, `IRIS_LANDING_GEAR`): a
   spring-damper at the landing-gear points, with smoothed Coulomb friction
   (penalty method); `Multirotor(..., contact=GroundContact(gear))`. Tests: at rest
   the Iris sinks exactly mg/(4k) = 1 mm and the state is an equilibrium (1e-12); a
   drop bounces as the exact solution of the model says (rebound within 0.2 %),
   with a restitution above the textbook e = exp(−ζπ/√(1−ζ²)) (changed from the
   plan, see the decision log); a slide brakes in the closed-form time (≈ μg); every
   mode at rest is stable with RK4 at 100 Hz, and a little more damping or less slip
   speed is not; a rough landing ends at rest and level.
3. ✅ **Take-off and landing with the autopilot** (`LandDetection`,
   `Autopilot(..., landed=True)`): on the ground the motors are stopped and the
   integrators held at zero; asked to climb faster than 0.2 m/s it takes off; in
   flight it detects a landing from its speed and its own thrust. Tests: waiting 1 s
   or 5 s on the ground gives the same take-off to the last bit; the take-off is
   the climb it would fly in the air (within 0.005 m/s after 1 s), straight up; a
   0.5 m/s landing is detected 1.3 s after touching and ends exactly at rest; never
   a landing in the air (hover, braking, a 5 m/s descent).
4. ✅ **Collisions in flight** (`jineta/campo/collision.py`, `jinete/hitbox.py`,
   `Room.wall_clearance`, `IRIS_HITBOX`): the drone's spheres against walls,
   ceiling and obstacles; the floor by the landing gear, a crash once the drone has
   left it; the moment of contact interpolated inside the step. Tests: straight at a
   wall, the ceiling, a crate or a pillar, the moment exact (1e-9); spheres placed
   with the attitude; the earliest contact inside a step wins; a free fall onto the
   floor within g dt²/(8v); standing still and taking off are not crashes, coming
   back down is; the tunnelling limit measured (a thin wall needs a step longer
   than 2R + w, 79 m/s for the Iris; a graze misses at most the depth of a chord as
   long as the step, 1.2 cm at 20 m/s).
4b. ⏳ **Shape of the drone with several spheres** (added during step 4): one sphere
   per rotor disc and one for the body instead of one around everything, for the
   Iris; the sphere is 0.77 m high for a 0.11 m high vehicle. Tests: every rotor
   disc and the collision box inside the spheres; how much room it gives back near
   the ceiling and beside a wall.
5. ⏳ **Gates, rectangular and circular**: crossing between samples, through the
   opening and in the right direction, with the time interpolated inside the step;
   touching the frame is a crash (four bars, or a torus).
6. ⏳ **Course and referee**: gate order, lap time, end by crash, finish or time
   limit.
7. ⏳ **Course files in YAML**: a strict format with clear errors, and a first
   course.
8. ⏳ **See a flight**: a figure of an Iris that takes off, flies through the gate
   centres with its autopilot and finishes.
9. ⏳ **Close the phase**: this document and the README.

## 6. Decision log

Phase 8 (indoor world):
- **The drone collides as a set of spheres fixed to its body** (`Hitbox`): a sphere
  touches a shape when the signed distance of its centre is below its radius, so any
  shape with a signed distance is checked the same way, for one drone or many. The
  monitor takes a list of spheres from the start; step 4 gives the Iris one sphere
  around everything (the plan's "sphere first"), step 4b several.
- **The Iris sphere is 0.384 m**, the reach of its front rotor discs: hub
  0.2555 m out and 0.023 m up, propeller radius 0.128 m (`rotor_N_collision` of
  `iris.sdf.jinja`): sqrt((0.2555 + 0.128)² + 0.023²). The exact reach of a disc,
  not |hub| + radius (the triangle inequality, a little more). The corners of its
  collision box (0.337 m) are inside.
- **A better shape is a step, not a loose end** (decided with the author, step 4b
  added to the plan): the hitbox sets where the drone fits, so an oversized one
  makes the policy keep away from room it has (slower laps, the MVP's measure) and
  makes the simulated drone unlike the real one (phase 14). It comes before the
  gates (whose frames are checked the same way) and before the batch engine (which
  must reproduce it).
- **The floor is judged by the landing gear, not by the spheres** (changed from the
  plan, which said "after take-off, touching the floor is a crash" without saying
  with what): the drone starts standing on the floor, its centre of mass 5.4 cm up,
  inside its own 0.384 m sphere. A crash is a gear point touching the floor after
  the first sample with every gear point above it. So "touching the floor" means
  "the floor pushes on the drone", and flying 20 cm up is not a crash. Price: a
  drone banked hard or upside down near the floor touches with a rotor first, and
  that is seen late (loose end 25). The walls and ceiling use
  `Room.wall_clearance`, the room without its floor.
- **Collisions are checked at the samples, and the moment interpolated inside the
  step** (linear in the margin between the last free sample and the first touching
  one): exact flying straight at a flat face, error ~g dt²/(8v) in a free fall
  (3e-5 s at 100 Hz). When several checks start touching in the same step the
  earliest one is reported. Rejected for now: checking the segment between samples
  (conservative advancement: a distance never changes faster than the point moves,
  so steps of distance − r along the segment can never jump a contact). With the
  Iris it is not needed (see the tunnelling limit); a small fast profile may need
  it (loose end 26).
- **Tunnelling limit, measured** (straight sampled paths, any phase of the samples):
  a slab of thickness w crossed head on is always seen while the step is shorter
  than 2R + w and can be jumped beyond it: (2R + w)/dt = 79 m/s for the Iris at
  100 Hz with w = 2 cm, four times its top speed. Grazing a rounded edge of an
  inflated shape (radius ρ: R at a box edge, R + radius at a column), the deepest
  contact that can go unseen is ρ − sqrt(ρ² − s²/4) ≈ s²/(8ρ): 1.3 cm at a box edge
  and 1.2 cm at a 5 cm pillar, at 20 m/s.
- **The monitor stops the run at the first contact**; nothing pushes back from
  walls or obstacles (no contact force), because a collision ends a lap. Cost:
  ~57 µs per check with two obstacles in Claude's workspace, against ~860 µs for a
  step of the Iris with its autopilot (+7 %).
- **The tests were checked by breaking the code**: spheres or feet placed with R^T,
  the floor counted before take-off, airborne from the first sample, no
  interpolation, the first check in the list instead of the earliest, the floor
  height with the wrong sign, the room with its floor, the position not
  interpolated, the ceiling overhang with the wrong sign. The feet with R^T passed
  unnoticed at first (every floor test was level or only yawed, and the gear is
  symmetric): the floor margin is now checked nose up 20° on five headings.
- **The autopilot has a landed state, as PX4 does** (measured first: without it,
  an Iris on the floor with a zero command spun its motors at hover speed and
  floated 3.5 mm up; 3 s pressed down charged the vertical integrator with 3 m/s²
  and the next climb was still 15 % slow 3 s later; after a landing the motors kept
  spinning at 491 rad/s). Landed: motors stopped, every integrator held at zero, so
  waiting on the ground leaves nothing behind (an integrator charged against the
  floor makes a take-off late, or crooked if the rate loop is involved). Take-off:
  a climb command faster than `takeoff_speed` (0.2 m/s).
- **The landing is detected from what a real autopilot knows**, not from the
  contact force: speed below 0.1 m/s while the velocity loop asks for less than 70 %
  of the weight, for 0.5 s. In the air that is impossible for long (at 0.7 g of
  thrust it falls past 0.1 m/s within a few hundredths of a second); only the floor
  holding it up allows it. 50 % was the first choice and took 3.3 s to notice a
  landing (the integrator has to unwind further); 70 % takes 1.3 s. The speed
  condition is what keeps a fast descent (5 m/s asked, thrust below 70 % for most of
  a second while it speeds up) from looking like a landing. A "not asked to climb"
  condition was written first and removed: a climb command always asks for more
  than the weight, so it could never decide anything (a deliberate mistake there
  changed no test).
- **The default is `landed=False`**, a vehicle already in the air: every earlier
  test and example flies exactly as before. A vehicle that starts on the floor is
  built with `landed=True`; one built airborne on the floor finds out by itself
  when asked to descend.
- **Ground contact by the penalty method**: the floor lets a contact point sink and
  pushes back with `N = k·depth + c·(sinking speed)`, never pulling (N ≥ 0).
  Friction `−μN v_t / √(|v_t|² + v_s²)`: Coulomb when sliding, fading to zero below
  the slip speed v_s, because exact Coulomb friction jumps when the velocity changes
  sign and a fixed-step integrator cannot follow it. Force and moment in body axes,
  added by the vehicle to its own (`Multirotor.contact`, any `ForceMomentModel`).
- **The landing gear is vehicle data** (`jinete/contact.py`, not `campo`): contact
  points in body axes, like the rotors. `campo` and `jinete` still only import
  `core`. The floor is the plane z = `floor_z` (0 for the room).
- **The Iris gear from its PX4 file**: the four bottom corners of its collision box
  (0.47 × 0.47 × 0.11 m, centred on the CoM); stiffness so that it sinks the file's
  `min_depth`, 1 mm, at rest (k = 3727 N/m per corner; Gazebo's own contact is
  nearly rigid, kp = 1e12, which a penalty spring cannot be); μ = 1, the SDF default
  (the file's friction block is empty).
- **The damping ratio (0.5) and the slip speed (0.2 m/s) are set by stability at
  100 Hz, not by the source.** A prototype computed every mode of the resting Iris
  (Jacobian eigenvalues) and RK4's amplification `|R(λ dt)|` for each: two modes set
  the limit, the overdamped tilt on the springs (ζ = 0.7 already gives 1.85) and the
  yaw braked by the friction of the four feet (v_s = 0.1 m/s gives 1.34: its rate is
  ~ μ N r² / (v_s I_zz)). A test keeps the gear inside the limit and another shows
  that the neighbouring values are outside it. Price: below 0.2 m/s friction is soft,
  so a foot under a sideways push creeps slowly instead of sticking.
- **The rebound is tested against the exact solution, not the textbook
  restitution** (changed from the plan): e = exp(−ζπ/√(1−ζ²)) assumes a damper that
  also pulls the body back on the way out; this floor never pulls, so the body
  leaves a little earlier and faster (e = 0.617 against 0.605 for ζ = 0.16). While
  touching, the motion is the closed-form damped oscillator around m g / k; it leaves
  where k x + c x' = 0 (found with `brentq`), and then flies freely. At dt = 1e-4 the
  rebound matches to 0.1 % (70.18 mm against 70.11 mm). At 100 Hz the same bounce
  rebounds 68.8 mm (−1.9 %): a hard impact lasts ~5 steps, which is stable but not
  precise; for landing it does not matter.
- **The tests were checked by breaking the code** again: five mistakes (pulling
  floor, friction sign, moment from the NED force, point velocity not rotated,
  damping not shared between the points). Two passed unnoticed at first, both
  because a level vehicle heading North has R = I: the moment is now checked exactly
  with the vehicle banked, and the friction with the vehicle spinning on a heading of
  30°.
- **Shapes are described by their signed distance** (positive outside, negative
  inside, the true Euclidean distance): one number tells both whether a sphere of
  radius r touches the shape (distance < r) and how much room is left, and the
  same function serves collisions, the referee and later the observation of a
  policy. Formulas: box `|max(q, 0)| + min(max qᵢ, 0)` with `q = |p| − half size`
  in the box's axes; vertical cylinder the same with two components (radial,
  vertical).
- **The room is the inside of a box with one floor corner at the origin**:
  `0 ≤ x ≤ length`, `0 ≤ y ≤ width`, `−height ≤ z ≤ 0`, so the floor is z = 0 and
  every position in a room is positive in x and y. Its `clearance` is minus the
  box's signed distance.
- **Obstacles turn only about the vertical** (a yaw, as a heading: positive turns
  North towards East). Tilted obstacles are not needed for a course.
- **Every shape takes one point (3,) or many (n, 3)** and returns a float or an
  array: the batch engine of phase 9 can ask for many drones at once.
- **Tests are checked by breaking the code**: four deliberate mistakes (yaw sign,
  inside part missing, cylinder upside down, room sign) each made tests fail. The
  yaw sign was caught by only one test at first (90° and 45° look the same turned
  either way), so a 30° test was added.

Phase 7 (project goal):
- **The MVP is a quadrotor learning a timed indoor course**, measured against a
  classical baseline on the same course: without a reference, "fastest possible"
  cannot be judged.
- **The policy commands collective thrust and body rates**, the setpoint a real
  PX4 autopilot accepts in offboard mode from a companion computer. In the
  simulator the existing rate loop executes it. Rejected: rotor speeds (no real
  autopilot takes them from outside; the policy would have to learn the mixer) and
  velocity commands (too slow and conservative for a racing course).
- **The reference engine stays the truth.** Speed comes from a separate batch
  engine checked against it, not from simplifying the validated models.
- **Simplest first, one change per phase**: perfect state before sensors, an ideal
  room before randomisation, so each phase's effect can be measured.
- **Real flight is the final goal, not the MVP.** The MVP must be deployable in
  design (interface, rates, limits); the hardware, its profile and indoor
  positioning come in phase 14.
- **The fixed-wing model and the Kalman filter are kept**: both are validated
  engine parts; "any drone profile" includes fixed wings, and indoor flight needs
  state estimation.
- **Git history is kept** (no rewriting): the cleanup is a commit on top.

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
- **ISA scope:** two layers, troposphere (0–11 km, −6.5 K/km) and isothermal layer
  (11–20 km). Altitude is treated as geopotential without conversion (difference
  ~0.17 % at 11 km). Above 20 km results are not valid and this is documented, not
  enforced: drones fly far below. Below sea level the
  troposphere formulas simply continue.
- ISA keeps its own `STANDARD_GRAVITY` (same value as `GRAVITY`) on purpose: g0 is
  part of the ISA definition, not the gravity model of the simulator.
- Speed of sound is available (`isa_speed_of_sound`) so Mach can drive aero
  coefficient tables in phase 6.
- **Wind only acts through airspeed:** drag uses `v − wind`, the state keeps the
  ground velocity. A body released in wind drifts towards the wind velocity with a
  lag of the order of `v_terminal / g` (~1.8 s for the example body).
- `ProfileWind` interpolates NED components, not speed and direction, so a wind
  turning through north does not swing through south.
- `PowerLawWind` clamps the height to `[0, gradient_height]`: zero wind at and below
  the ground, constant above the gradient height. Exponent 0 gives a uniform wind
  (`0**0 = 1` in Python, covered by a test).
- `DiscreteGust` is the deterministic "1-cosine" gust, uniform in space (no gust
  front travelling across the field). `CombinedWind` sums any models.
- **Stochastic turbulence (Dryden/von Kármán) is deferred:** it needs a seeded
  random generator to keep runs reproducible for Monte Carlo, and it only matters
  once vehicles have control loops (phase 6+).
- **Vehicle parameters come from open sources:** published research platforms and
  airframes (e.g. open quadrotor datasets, PX4 SITL models, XFLR5/AVL analyses).
  A profile that stands for a class of vehicle rather than a specific one (e.g.
  "small fixed-wing drone") says so in its comment.
- Shared interfaces (`GRAVITY`, `DensityModel`, `WindModel`, `sea_level_density`,
  `no_wind`) moved from `point_mass.py` to `core/environment.py`, so `RigidBody`
  and `campo` no longer depend on the point-mass model. `point_mass` still imports
  them, so old imports keep working, but new code imports from `environment`.
- Physical validity is checked once, at construction: mass > 0, drag area ≥ 0,
  inertia symmetric + positive definite + triangle inequality (flat-plate equality
  allowed). The validated inertia is read-only because its inverse is cached; a
  silent in-place change would leave a stale inverse.

Phase 6a (quadrotor):
- **Phase 6 is split**: 6a quadrotor + control, 6b fixed wing.
- **Discrete controllers with zero-order hold.** A PID has memory (its integral).
  `force_moment` is called 4 times per RK4 step at trial states, so a controller
  inside it would update its memory 4 times with states that are not real. Instead
  `run` calls `pre_step(t, y)` once per step; the controller sets the vehicle
  commands there and they stay constant for the whole step, as in a real autopilot.
  The rigid body stays pure and the state stays 13 elements (rejected alternative:
  appending the controller states to the state vector). For now the control rate is
  the simulation rate (1/dt); a wrapper can run a controller every N steps if needed.
- **Profiles separate from code** (see conventions). The first profile is the 3DR
  Iris of PX4 SITL (`iris.sdf.jinja`): open and well documented, used to validate
  the model. Its thrust-to-weight ratio is only 1.896 (max 7.07 N per rotor; hover
  at 3.73 N per rotor = 799 rad/s, 73 % of the 1100 rad/s maximum): enough to
  validate the models and fly a first course, low for racing (racing quadrotors are
  around 5–10). A more agile profile will be added later as data, with its own
  sources.
- Iris mass = base_link 1.5 kg + 4 × 0.005 kg rotors = 1.52 kg. Inertia = base_link
  only: the rotors would add 1–3 % (parallel-axis terms), below the uncertainty of
  the published values.
- **Mixer** built from the rotor positions and spins, not a hand-written X matrix
  (the Iris is not square: arms 0.13/0.22 m front, 0.13/0.20 m back). Allocation
  matrix rows: total thrust, roll `−y_i`, pitch `x_i`, yaw `spin_i · moment_constant`.
  Inverse with the pseudo-inverse (exact for 4 rotors, minimum effort for 6 or 8).
  Layouts with rank < 4 are rejected (e.g. all rotors spinning the same way cannot
  yaw). Rotor thrusts are clipped to `[0, T_max]`; when that happens the produced
  wrench differs from the requested one.
- **The vehicle input is the rotor speed** [rad/s], like an ESC command, clipped to
  `[0, Ω_max]`. Motors respond instantly: the Iris motor time constants (12.5 ms up,
  25 ms down) are not modelled.
- **Thrust, reaction torque and rotor drag scale with ρ/ρ_ref** (ρ_ref = 1.225 kg/m³):
  propeller theory gives `T = C_T ρ n² D⁴`, and the PX4 constants are sea-level
  values. At 3000 m the Iris has ~74 % of its sea-level thrust. With that factor
  the rotor drag grows like √ρ at constant thrust, the same law the PX4 EKF2 drag
  model uses.
- **Rotor drag** (Gazebo motor model form): at each hub,
  `−(ρ/ρ_ref) · c_rd · Ω_i · v_perp,i`, linear in airspeed, with v_perp the hub
  airspeed in the rotor plane including `ω × r_i`. Applied at the hub, so it pitches
  the nose up in forward flight (hubs 0.023 m above the CoM) and damps yaw rotation.
  `c_rd = 0.000175` from `iris.sdf`.
- **Body drag** (PX4 EKF2 drag model form): `−½ ρ CdA_i u_i |u|` per body axis
  (isotropic when the CdA are equal). The Gazebo Iris has no body drag; CdA =
  m / 100 kg/m² = 0.0152 m² from the default PX4 ballistic coefficient
  (`EKF2_BCOEF_X/Y`), a generic value not measured on the Iris. No open value for
  the vertical axis: same CdA as a placeholder.
- **Not modelled, on purpose:** the Gazebo thrust loss with axial airspeed
  (`1 − v/25`, it reduces thrust equally when climbing and descending, which is not
  physical), the Gazebo rolling moment (coefficient 1e-6, negligible) and the
  gyroscopic moments of the rotors.
- The in-air decoupling test pending since phase 4 is done: with the same ground
  velocity, a level quad feels rotor drag and a nose-up one (air along the rotor
  axis) does not.

Phase 6a (control):
- **Cascade** velocity → attitude → body rate → mixer → rotor speeds, each loop a
  discrete controller called from `pre_step`, each one slower than the loop it
  commands.
- **Rate loop: PI, no D.** It asks for an angular acceleration
  `α = kp·e + ki·∫e` and turns it into a moment `M = I·α + ω × Iω`; the second term
  cancels the gyroscopic coupling, so each axis is exactly `ω̇ = α`. With
  instantaneous motors the plant is first order and needs no D (PX4 uses one; it
  matters with motor lag and sensor noise). Anti-windup: the integral term is
  clamped. kp chosen here (PX4's are in its normalised mixer units and do not
  transfer): roll/pitch 20 1/s (τ = 50 ms), yaw 5 1/s (little yaw authority).
- **Rate-loop integral from the PX4 I/P ratios** (0.2/0.15 roll and pitch, 0.1/0.2
  yaw), which are unit-free and do transfer. The first choice, critical damping of
  the rate loop alone (`ki = kp²/4`), left a slow tail in the attitude response
  once cascaded: tuning a loop alone is not tuning it in the cascade.
- **Attitude error** `q_err = conj(q_cmd) ⊗ q`: the rotation from the commanded to
  the actual attitude, in body axes. Turned into a rotation vector the short way
  (flip q if w < 0: q and −q are the same attitude), and `ω_cmd = −kp·θ`, clipped to
  the rate limits. `kp_att = kp_rate / 4` makes the cascade critically damped when
  the rate loop is P only (natural frequency kp_rate/2). Values 5, 5, 1.25 1/s,
  same order as PX4 (`MC_ROLL_P` = 4.0, `MC_YAW_P` = 2.8, same units). Rate limits
  3.5 rad/s roll/pitch (PX4 220 deg/s), 1.5 rad/s yaw.
- **Velocity loop** with the PX4 default gains, which are in the same units
  (acceleration per velocity error): kp 1.8 / 1.8 / 4.0, ki 0.4 / 0.4 / 2.0, tilt
  limit 45 deg; PX4's D term is left out. Thrust vector `F = m(a − g)`; limits with
  the vertical part first (thrust range 10–90 % of the maximum, the rest of the
  horizontal part up to the tilt limit); attitude with body z against F and the
  nose towards the commanded heading (`quaternion.from_matrix`, Shepperd's method);
  total thrust = F projected on the *current* thrust axis (Lee/Mellinger), so a
  vehicle still turning does not push the wrong way and gives no thrust upside
  down.
- **Velocity integral limit 5 m/s²** (chosen): with 3 m/s² the integral saturated in
  a 10 m/s cruise (drag needs ~3.6 m/s²) and the speed stuck at 9.7 m/s. It also
  covers the ~3.4 m/s² lost at 3000 m.
- **The controllers do not know the air density.** They convert thrust to rotor
  speed at the reference density; the vertical integral absorbs the difference
  (tested at 3000 m).
- **Closed-loop tests compare with the exact discrete solution** where possible
  (P-only rate loop: `e_(k+1) = (1 − kp·dt) e_k`; P-only rate + attitude: a
  two-line recursion), not with the continuous one: with a zero-order hold the
  discrete solution is what the simulation must reproduce (to 1e-9 / 1e-6), and
  the continuous one differs by about 1 %.
- **Top speed of the model: 20.7 m/s (74 km/h)** at the 45 deg tilt limit (example).
  No published top speed of the Iris was found, so it is plausible but not
  validated. The PX4 SITL Iris weighs 1.52 kg; the real Iris+ is listed at
  1.282 kg (iFixit): the model is the SITL vehicle, not the retail one.

Phase 6b (fixed wing):
- **Air data** (`aero.air_data`): V, α = atan2(w, u) (keeps the quadrant: tail first
  is 180°), β = asin(v/V) in [−90°, 90°]; α > 0 air from below the nose, β > 0 air
  from the right. Below `min_speed` the angles are 0, never NaN. The inverse
  `airspeed_from_air_data` builds [u, v, w] from V, α, β (for trim).
- **Aerodynamic model** (`jinete/fixed_wing.py`) as in Beard & McLain, *Small
  Unmanned Aircraft*, ch. 4: coefficients = sums of stability and control
  derivatives; rates made nondimensional (p b/2V, q c/2V, r b/2V) so one derivative
  holds at any speed; lift and drag rotated into body axes by α (stability axes),
  side force and moments with β directly (good for small β, as the source). Static
  lift: the linear curve blended into a flat plate (2 sin²α cos α) past the stall by
  a sigmoid σ(α; M, α₀). Drag: parabolic polar C_D_p + C_L_lin²/(π e AR).
- **Generic small fixed-wing profile** (`SMALL_FIXED_WING`): Aerosonde-class numbers
  from the open-source simulator of the same book (`mavsim_public`,
  `aerosonde_parameters.py`): 11 kg, b = 2.90 m, S = 0.55 m², AR = 15.2. The source
  file is a student template with the force equations left blank, so only its
  parameters are used; its linear drag model (C_D_0, C_D_alpha) is not.
- **Control signs as in the source:** elevator > 0 trailing edge down, pitches the
  nose down; aileron > 0 rolls right; rudder > 0 yaws the nose left.
- c_ell is the rolling-moment coefficient (C_l in most books), renamed so it cannot
  be read as lift.
- **Propulsion: the source's motor + propeller model**, chosen over a simple throttle
  → thrust law (the source no longer lists the constants of the simple law, and that
  law gives ~790 N of static thrust, 7 times the weight). The throttle sets the motor
  voltage; the rotor speed is where motor torque K(i − i0) equals propeller torque
  ρn²D⁵C_Q(J): a quadratic in Ω. Thrust ρn²D⁴C_T(J). Written multiplied out (no
  division by n), so it also holds for a stopped propeller. Density enters through the
  physics, with no extra factor. Static thrust at full throttle ~82 N (thrust-to-weight
  0.76). Checked against the reference values of the source's chapter 4 check script
  (T = −12.4307 N at 25 m/s and throttle 0.5) to 1e-12.
- **The motor is stiff** (K²/R large): Ω ≈ throttle · V_max / K almost whatever the
  load. The throttle sets the rotor speed; the airspeed decides what the propeller
  does with it (+30 N at rest, −5 N at 25 m/s, both at throttle 0.6).
- **Throttle 0 is not "engine off"**: with no voltage the motor is a shorted
  generator and the windmilling propeller brakes (−21.9 N at 25 m/s, twice the
  airframe's drag). A glider is `FixedWing(params)` with no propulsion.
- **The propeller sees the axial airspeed u**, not V (the advance ratio is defined
  with the axial inflow; the source uses V). Reverse flow (u < 0) is taken as 0: the
  fits have no data there. Ω is clipped at 0 (below the no-load voltage the motor
  cannot turn the propeller; the root would spin it backwards).
- **Thrust along body x through the CoM; the propeller torque rolls the airframe
  with −Q** (sign confirmed by the source's case 1). Not modelled: P-factor,
  slipstream on the tail, gyroscopic moment of the propeller, motor lag.
- **Vehicle checked against the source** (`chap4_check.py`): case 1 (level, 25 m/s)
  exact to 1e-9 in forces, moments and angular accelerations; case 3 (any attitude,
  rotating, gust) to 1e-6 once the u-versus-V difference of the propeller is undone,
  except F_y, 0.0015 N off (3e-5), not explained (loose end 18). The source's case 2
  is not used: its script computes it with the airspeed of case 1.
- **Glide test**: for a fixed elevator, C_m = 0 gives α, tan γ = −C_D/C_L and
  L = W cos γ give γ and V (elevator −0.1: α 2.35°, L/D 9.6, γ −5.94°, V 26.68 m/s).
  The analytic state is an equilibrium to 1e-12, and a glider released level and
  20 % slow settles on it after the phugoid dies out.
- **Trim as a 6×6 root problem**: three forces and three moments zero; unknowns α, β,
  elevator, aileron, rudder, throttle; φ = 0 fixed (wings level; seven unknowns for
  six equations). `scipy.optimize.root` (Powell hybrid). Lateral trim is not zero: the
  aileron holds the propeller torque (+0.36° at 25 m/s, β 0.02°).
- **The trim starts from a textbook estimate** (lift = weight and C_m = 0, linear in α
  and elevator; then the throttle with thrust = drag). From a fixed guess the solver
  jumped past full throttle, where the clipped thrust has no slope, and failed at some
  speeds that do trim (32.05 m/s); with the estimate it trims at every speed from
  12 to 32.45 m/s at 100 m. Top level speed ~32.5 m/s.
- **Linearisation by central differences** of the full 13-state derivative at the
  trim; eigenvalues give the modes; the zero ones (position, heading, quaternion
  norm) are left out; a mode is lateral or longitudinal by its eigenvector. At
  25 m/s: short period ωn 10.80 rad/s ζ 0.438; phugoid ωn 0.500 rad/s (13.1 s)
  ζ 0.276; roll −21.7 1/s; Dutch roll ωn 4.71 ζ 0.236; spiral +0.090 1/s.
- **Lanchester overestimates the phugoid frequency by ~10 %**: it holds α constant.
  Letting the short period settle quasi-statically, ωn = √2 g/V · √(−M_α/ω_sp²),
  matches to < 1 %. **The phugoid is damped mostly by the propeller**: its thrust
  falls with speed, so ζ is 3–4 times the constant-thrust value.
- **The profile is spirally unstable** (C_lβ C_nr < C_lr C_nβ: −0.0059): a bank of
  0.5° grows to ~7° in 30 s (doubling ~7.7 s at 25 m/s). Common in small UAVs; the
  autopilot holds the roll.
- **Autopilot by successive loop closure** (Beard & McLain ch. 6): course → roll →
  aileron; altitude → pitch → elevator; airspeed → throttle; sideslip → rudder (PI to
  β = 0). Gains from second-order models of each axis at the trim.
- **Roll and pitch gains from the surface authority** (the source's rule: kp = surface
  limit / error that saturates it), not from a chosen natural frequency. With a
  chosen ω the pitch gain at 30 m/s came out almost zero (the static stability alone
  already gave ~11.8 rad/s), the aircraft barely followed the pitch command and a
  descending reversal ended 117 m lower (regression test). A slow inner pitch loop
  (6 rad/s, below the short period) was unstable: Beard's pitch model ignores the α
  dynamics.
- **Trim feedforward** in the autopilot: the elevator, throttle and pitch of the trim
  are added to the loop outputs. Without it the slow altitude integrator had to
  build the trim elevator alone and the pitch oscillated for tens of seconds.
- **Design choices** (`SMALL_FIXED_WING_AUTOPILOT`): surfaces ±30° (only the
  autopilot clips, loose end 17); bank ≤ 35° (at 18 m/s and 45° the elevator
  saturated and a 90° turn lost 12 m); pitch ≤ 20°; course loop 8 and altitude loop
  15 times slower than their inner loops (12 left a slow oscillation after a 20 m
  step); airspeed 0.8 rad/s; sideslip 2 rad/s; anti-windup by conditional
  integration. 100 Hz: at 50 Hz the roll loop (11–19 rad/s) degrades.
- **Orbit guidance**: the vector field of Beard & McLain ch. 10,
  χ = φ + λ(π/2 + atan(k(d − R)/R)), gain 2. The autopilot holds a 150 m orbit at
  18 m/s within 0.5 m, at the bank of a steady turn, atan(V²/gR).

Phase 6c (Kalman filter):
- **The Kalman filter is a set of plain functions** (`atalaya/kalman.py`): `predict`
  and `update` take matrices and return new arrays, and know nothing about vehicles
  or sensors; whoever uses them holds the state and the time.
- **Covariance update in Joseph form**, `(I − KH) P (I − KH)ᵀ + K R Kᵀ`: the same as
  `(I − KH) P` with the optimal gain, but always symmetric and positive
  semi-definite, also with rounding errors. The gain is solved with
  `np.linalg.solve`, not by inverting S.
- **Continuous white-noise acceleration** (`white_noise_acceleration`, spectral
  density q [m²/s³]): predicting an interval in one go or in pieces gives exactly
  the same covariance, so a filter predicted at the simulation step between
  measurements does not depend on the step. The piecewise model (one random
  acceleration per interval) does depend on it; it is kept because the alpha-beta
  filter has its closed-form steady state (Kalata) with it, which tests the filter.
- **Consistency is tested with NEES**, `eᵀ P⁻¹ e`: over N seeded runs its average is
  a χ² with 6N degrees of freedom divided by N when the filter's model matches the
  truth (5.67–6.05 against the 99 % interval 5.50–6.53 over 300 runs); an
  overconfident filter gives 49–304.

## 7. Current state

- `jineta/core/integrators.py` — `euler_step`, `rk4_step`, `Derivative` type alias.
  Tested against free fall.
- `jineta/core/environment.py` — interfaces between dynamics and environment:
  `GRAVITY`, `DensityModel`, `WindModel`, and the defaults `sea_level_density` and
  `no_wind`.
- `jineta/core/inertia.py` — `inertia_tensor(value)`: validates a 3x3 inertia tensor
  and returns a read-only copy.
- `jineta/core/point_mass.py` — `PointMass` dataclass (mass, drag_area, density
  model, wind model; validated) with `derivative()`: constant gravity + quadratic
  drag on the airspeed, `D = -½·rho·CdA·|v − w|·(v − w)`.
- `jineta/core/simulation.py` — `run(f, y0, dt, t_end, stop=None, post_step=None,
  pre_step=None)`: fixed-step RK4 loop. Per step: `pre_step(t, y)` (called with a
  copy of the state at the start of the step; controllers act by side effect and
  their commands are held for the step), RK4, `post_step(y) -> y`, `stop(t, y)`.
  Returns `(times, states)`. The docstring documents every argument.
- `jineta/core/quaternion.py` — `multiply`, `from_axis_angle` (accepts any
  array-like axis), `conjugate`, `to_rotation_vector` (short way, angle in
  [0, π]), `normalize`, `to_matrix`, `from_matrix` (w ≥ 0, Shepperd's method),
  `from_euler(roll, pitch, yaw)` and `to_euler(q)` (ZYX, display and controllers
  only), `derivative(q, omega_body) = ½ q ⊗ [0, ω]`.
- `jineta/core/rotating_body.py` — `RotatingBody` dataclass (validated inertia 3x3,
  torque model) with Euler's equations `ω̇ = I⁻¹(M − ω × Iω)`; inverse inertia
  computed once in `__post_init__`. Also `TorqueModel`, `zero_torque`.
- `jineta/core/rigid_body.py` — `RigidBody` dataclass (validated mass and inertia
  3x3, `force_moment` model) with the 13-element `derivative()`: body force rotated
  to NED with `R(q)` + gravity, plus quaternion kinematics and Euler's equations.
  Also `ForceMomentModel`, `zero_force_moment`, `normalize_attitude` (post-step
  hook).
- `jineta/campo/atmosphere.py` — ISA 0–20 km: `isa_temperature`, `isa_pressure`,
  `isa_density`, `isa_speed_of_sound` (scalar functions of altitude), plus the ISA
  constants (`SEA_LEVEL_*`, `GAS_CONSTANT_AIR`, `LAPSE_RATE`, `TROPOPAUSE_*`, ...).
  `isa_density` plugs directly into `PointMass(density=...)`.
- `jineta/campo/wind.py` — `meteorological_wind` (FROM-direction → NED vector),
  `ConstantWind`, `ProfileWind` (table vs altitude, linear interpolation, end values
  held outside), `PowerLawWind` (boundary layer, gradient height), `DiscreteGust`
  (1-cosine in time), `CombinedWind` (sum of models). `ConstantWind`,
  `ProfileWind`, `PowerLawWind` and `DiscreteGust` validate their inputs.
- `jineta/jinete/aero.py` — `body_airspeed(t, state, wind=no_wind)`: airspeed in
  body axes, `R(q)ᵀ (v − wind(t, position))`. `AirData(speed, alpha, beta)`,
  `air_data(airspeed, min_speed=1e-6)` (V, α = atan2(w, u), β = asin(v/V); angles 0
  below `min_speed`) and its inverse `airspeed_from_air_data(speed, alpha, beta)`.
- `jineta/jinete/multirotor.py` —
  - `Rotor(position, spin)`: position in FRD body axes relative to the CoM [m],
    spin ±1; frozen, position read-only.
  - `allocation_matrix(rotors, moment_constant)`.
  - `MultirotorParams` (frozen, validated): `mass`, `inertia`, `rotors`,
    `thrust_coefficient` k_T, `moment_constant` (Q/T [m]), `max_rotor_speed`,
    `rotor_drag_coefficient`, `body_drag_area` [CdA_x, CdA_y, CdA_z],
    `reference_density` (1.225). Cached read-only `allocation` / `allocation_inv`.
    Methods: `max_rotor_thrust`, `wrench(thrusts)`, `rotor_thrusts(wrench)`
    (clipped), `rotor_speeds_for(thrusts)` (at the reference density).
  - `Multirotor(params, density=sea_level_density, wind=no_wind, contact=None)`:
    holds `rotor_speed_command` (starts at zero) and a `RigidBody`;
    `derivative(t, state)` and `force_moment(t, state)` (thrust + reaction torques
    via the mixer, rotor drag, body drag, plus the contact model if any; all in
    body axes).
- `jineta/jinete/control.py` —
  - `RateGains`, `RateController(params, gains, dt)`: `update(rate_command,
    thrust_command, state) -> rotor speeds`, `reset()`.
  - `AttitudeGains`, `AttitudeController(gains)`: `update(attitude_command, state)
    -> rate command`.
  - `VelocityGains` (kp, ki, integral_limit, max_tilt, thrust_range),
    `VelocityController(params, gains, dt)`: `update(velocity_command, yaw_command,
    state) -> (attitude command, total thrust)`, `max_thrust`, `reset()`.
  - `LandDetection(takeoff_speed=0.2, max_speed=0.1, thrust_fraction=0.7,
    hold_time=0.5)` (validated).
  - `MultirotorGains(rate, attitude, velocity, land=LandDetection())`.
  - `Autopilot(vehicle, gains, dt, landed=False)`: callable `pre_step` hook with
    `velocity_command`, `yaw_command` and `landed`; runs the three loops and writes
    the vehicle's `rotor_speed_command` (zero while landed); `reset()` clears the
    integrators and the landing timer.
- `jineta/jinete/profiles.py` — `flu_to_frd(vector)`, `IRIS` (PX4 SITL Iris, sources
  in the comments), `IRIS_RATE_GAINS`, `IRIS_ATTITUDE_GAINS`, `IRIS_VELOCITY_GAINS`,
  `IRIS_GAINS`, `IRIS_LANDING_GEAR` (the corners of its PX4 collision box),
  `SMALL_FIXED_WING` (generic small fixed-wing UAV, Aerosonde-class
  open data), `SMALL_FIXED_WING_PROPULSION` (its motor and propeller),
  `SMALL_FIXED_WING_AUTOPILOT` (design choices of its autopilot). The profiles by
  name: `MULTIROTOR_PROFILES` (`"iris"` → `MultirotorProfile(params, gains,
  gear)`) and
  `FIXED_WING_PROFILES` (`"small_fixed_wing"` → `FixedWingProfile(params,
  propulsion, autopilot)`).
- `jineta/jinete/fixed_wing.py` — `ControlSurfaces(elevator, aileron, rudder)`,
  `FixedWingParams` (frozen, validated; mass, inertia, S, b, c, e and the
  derivatives; `aspect_ratio`), `stall_blend`, `lift_coefficient`,
  `drag_coefficient`, `aero_force_moment(params, airspeed, rates, surfaces,
  density)` → (force, moment) in body axes; `FixedWing(params, propulsion=None,
  density, wind)` the vehicle (commands `surfaces`, `throttle`; `force_moment` =
  aerodynamics + thrust along x + rolling moment −Q; `derivative`).
- `jineta/jinete/propulsion.py` — `PropulsionParams` (D, K, R, i0, V_max, C_T and C_Q
  fits; frozen, validated), `PropellerOutput(thrust, torque, rotor_speed)`,
  `rotor_speed(params, axial_speed, throttle, density)` (torque balance, Ω ≥ 0) and
  `propeller_thrust_torque(...)`.
- `jineta/jinete/trim.py` — `Trim(state, surfaces, throttle, alpha, beta)` (state
  read-only), `level_flight_state(aircraft, speed, alpha, beta, altitude, course)`,
  `longitudinal_estimate(aircraft, speed, altitude)` (textbook α, elevator,
  throttle) and `trim_level_flight(aircraft, speed, altitude=100, course=0,
  tolerance=1e-9)`.
- `jineta/jinete/stability.py` — `jacobian(func, x)` (central differences),
  `Mode(eigenvalue, lateral)` (`natural_frequency`, `damping_ratio`, `oscillates`,
  `period`), `dynamic_modes(matrix)`, `AircraftModes` and `classify_modes(modes)`
  (short period, phugoid, roll, Dutch roll, spiral).
- `jineta/jinete/fixed_wing_control.py` — `AutopilotDesign` (limits, bandwidths,
  damping; validated), `AutopilotGains`, `design_gains(aircraft, trim, design)`,
  `wrap_angle`, `FixedWingAutopilot(vehicle, trim, design, dt)` (callable
  `pre_step`; `course_command`, `altitude_command`, `airspeed_command`; `reset()`)
  and `orbit_course(position, center, radius, clockwise=True, gain=2)`.
- `jineta/jinete/contact.py` — `LandingGear(points, stiffness, damping, friction,
  slip_speed)` (frozen, validated, points read-only; `from_damping_ratio(points,
  mass, sink, damping_ratio, friction, slip_speed)`), `contact_forces(gear, state,
  floor_z=0)` → (force, moment) in body axes, `GroundContact(gear, floor_z=0)`
  (callable force model), `box_bottom_corners(size)`.
- `jineta/campo/indoor.py` — `Box(center, size, yaw=0)` (`to_local`,
  `signed_distance`), `Cylinder(base, radius, height)` (`signed_distance`),
  `Room(size)` (`center`, `clearance`, `wall_clearance` (no floor), `contains`),
  `box_distance(local_points, half_size)`. Frozen, validated, arrays read-only; one
  point or many.
- `jineta/jinete/hitbox.py` — `Hitbox(centers, radii)` (frozen, validated,
  read-only), `disc_reach(center, radius)`, `box_corners(size)`,
  `bounding_sphere(rotor_hubs, propeller_radius, body_points)`. The Iris's is
  `IRIS_HITBOX` (with `IRIS_PROPELLER_RADIUS`); `MultirotorProfile` has a `hitbox`.
- `jineta/campo/collision.py` — `Collision(kind, index, time, position)`,
  `CollisionMonitor(room, obstacles, sphere_centers, sphere_radii, floor_points)`:
  `margins(state)` ([room, obstacles..., floor], negative = touching), callable
  `stop` (`collision`, `airborne`), `reset()`. `Obstacle` protocol (anything with
  `signed_distance`).
- `jineta/atalaya/kalman.py` — `predict`, `update` (returns the innovation and its
  covariance too), `nees`, `constant_velocity_transition`,
  `white_noise_acceleration`, `piecewise_white_acceleration`, `alpha_beta_gains`.
- Tests (485 passing, ruff check + ruff format clean):
  - integrators (2): RK4 exact on free fall; Euler's error shrinks with dt.
  - rotation (5): 90° about z maps North to East; proper orthogonal matrix; steady
    spin about a principal axis; constant torque spins up linearly; torque-free
    motion conserves energy and angular momentum.
  - rigid body (10): no forces is free fall; the quaternion norm drifts without
    `post_step` and stays unit with it; `post_step=None` changes nothing; thrust
    follows the attitude (3 yaws); a free spin does not affect the fall of the
    centre of mass; rejects mass ≤ 0 (2 cases).
  - inertia (10): read-only copy, accepts lists and products of inertia, accepts
    the flat-plate limit, rejects non-physical tensors (not 3x3, asymmetric, zero
    or negative moment, triangle inequality broken — 5 cases), both bodies validate,
    body inertia cannot be changed in place.
  - point mass (13): vacuum parabola, terminal velocity, drag reduces range,
    thinner air increases range; rejects invalid mass / drag area (3 cases);
    moving with the wind feels no drag; drag depends on airspeed, not ground speed
    (headwind / still / tailwind — 3 cases); a falling body drifts to the wind
    velocity and keeps the same terminal fall speed; the wind model receives time
    and position.
  - atmosphere (12): standard ISA table at 0 / 5 / 11 / 20 km (T, p, rho, a —
    4 cases); −6.5 K/km lapse rate; constant temperature in the isothermal layer;
    pressure continuous at the tropopause; density decreasing with altitude;
    hydrostatic equilibrium `dp/dh = −rho·g0` by finite differences (3 altitudes);
    ISA density read at altitude = −z inside `PointMass`.
  - wind (37): FROM/TO convention (4 headings) and speed kept; constant wind
    (independence, returns a copy, validation); profile wind (table values, linear
    interpolation, end values, ignores time and horizontal position, components not
    angles, plugs into `PointMass`, validation); power law (reference speed,
    exponent, direction, zero at ground, constant above gradient height, exponent 0,
    4 invalid inputs); gust (zero outside, starts/ends at zero, peak in the middle,
    half at a quarter, symmetric, mean = half the peak, uniform in space,
    validation); combined wind (sum, empty = calm).
  - aero (19): body airspeed for level / yawed 90° / pitched 30° / rolled 90°
    flight (4 cases); flying with the wind feels no air; hovering in a headwind;
    rotation keeps the airspeed magnitude (20 random attitudes, seeded); the wind
    model receives time and position; α and β for known attitudes (5 cases); round
    trip V, α, β → u, v, w → V, α, β (50 random, seeded); tail first is α = 180°;
    no airspeed gives zero angles; rejects wrong sizes (2) and negative speed.
  - multirotor (17): equal thrusts give pure lift (Iris is not square but balanced);
    single-rotor wrench signs; mixer round trip (20 random, seeded); roll command
    pushes harder on the left; yaw command speeds up the ccw rotors; clipping;
    rejects all-same-spin layouts, invalid rotors (2 cases) and non-positive
    constants (4 cases); profiles are immutable; FLU→FRD; Iris front-right rotor is
    at x > 0, y > 0, above the CoM; Iris thrust-to-weight 1.896.
  - multirotor dynamics (20): hover is an equilibrium (also over 10 s); motors off
    is free fall; 10 % more speed gives 1.21× thrust; speed command clipped (too
    fast, negative); thrust scales with ISA density (3000 m); mixer moments give
    `ω̇ = M/I` (roll, pitch, yaw — 3 cases); body drag quadratic; rotor drag linear;
    rotor drag ignores axial air; rotor drag pitches the nose up (lever arm
    0.023 m); yaw rate damped by rotor drag; wind acts through the airspeed;
    attitude changes the drag; rejects invalid drag parameters (4 cases); Iris body
    drag matches the PX4 ballistic coefficient.
  - simulation (5): `pre_step` called once per step with the start-of-step time and
    state; cannot alter the history; `None` changes nothing; the command is held
    during the step (u = t sampled → left Riemann sum `dt²·n(n−1)/2`, not `T²/2`);
    motors cut at 1 s → exact free-fall parabola afterwards.
  - rate control (10): P-only rate step is the exact discrete first order (3 axes);
    gyroscopic coupling cancelled; P only leaves the steady error `d/(I·kp)` under
    a disturbance moment; the integral removes it (critically damped PI); integral
    clamp and reset; rejects invalid gains (2 cases) and a non-positive period.
  - quaternion (13): conjugate is the inverse; rotation vector round trip (3
    angles); short way (350° → −10°); q and −q give the same vector; tiny angle;
    `from_matrix` round trip (100 random, seeded) and half turns (3 axes); Euler
    angles round trip (100 random, seeded); nose up, right wing down and heading
    East point the body axes where they should.
  - attitude control (7): `ω_cmd = −kp·θ`; error in body axes from the command;
    turns the short way; rate command clipped; rejects invalid gains; roll recovery
    matches the exact discrete cascade (1e-6) and the continuous critically damped
    response (1 %); combined 30°/−20° tilt with the real gains: tilt < 1° after
    0.6 s and < 0.3° after 1 s, heading < 0.5° after 2 s, no overshoot.
  - velocity control (13): hover asks for level and the weight; heading command;
    2 m/s² North pitches the nose down by atan(a/g); tilt limit keeps the vertical
    part; total thrust limit; thrust projected on the current axis (60° roll →
    half); no thrust upside down; rejects invalid gains (3 cases); 5 m/s step
    (behaviour: no overshoot, altitude within 0.5 m); holding still in a 10 m/s wind
    tilts into it by `atan(D/mg)` (1e-5 rad); at 3000 m the rotor speed settles at
    `Ω₀·√(ρ₀/ρ)` (0.1 %).
  - fixed wing (23): no air, no load; reference condition at α = 0 exact; lift
    perpendicular to the air and drag along it (α = −5, 4, 12° — 3 cases); loads
    scale with the dynamic pressure (speed, density); lift-curve slope = C_L_alpha;
    stall blend 0 / ½ at α₀ / 1 and symmetric; lift falls past the stall and is 0
    broadside; drag polar exact and smallest at zero linear lift; static pitch
    stability (exact slope); weathercock, dihedral and side-force signs; roll, pitch
    and yaw damping exact (3 cases); control surface signs and sizes exact; mirror
    symmetry in β; profile (AR, −J_xz, frozen); rejects invalid parameters (5 cases).
  - propulsion (34): the source's reference thrust and torque (2 cases, 1e-12);
    motor torque = propeller torque (12 operating points); thrust = ρn²D⁴C_T(J)
    (2 speeds); no throttle at rest gives nothing; thrust and rotor speed grow with
    the throttle (2 speeds); thrust falls with airspeed; throttle 0 in flight
    windmills and brakes (< −20 N); full-throttle static thrust 80–84 N; throttle
    clipped; reverse flow taken as none; thinner air: less thrust, faster propeller;
    `rotor_speed` consistent; rejects invalid parameters (7 cases); profile frozen.
  - fixed-wing vehicle (16): matches the source in level flight (1e-9) and in a
    general state with wind (1e-6; F_y within 2e-3); commands start neutral; a
    glider feels only the aerodynamics; thrust along the nose and torque as a rolling
    moment; thrust follows the attitude; flying with the wind feels no air; loads
    depend on airspeed, not ground speed; density read at the current altitude;
    commands read when the forces are evaluated. Glide: the analytic glide is an
    equilibrium (4 elevators, 1e-12); its numbers; a glider released level and slow
    settles into it (V ±0.01 m/s, γ ±0.02°, α ±0.01°).
  - trim (15): an equilibrium at 4 speeds (1e-9); faster needs less α, less elevator,
    more throttle; thinner air needs more α; the aileron holds the propeller torque
    (1e-6); trimmed flight stays steady for 60 s (1e-9); a uniform wind changes the
    ground speed, not the trim; the course turns the whole trim; full throttle limits
    the level speed (32.4 m/s trims, 33 does not); trims at every speed from 15 to
    32.25 m/s (regression); the textbook estimate is close; rejects a glider and a
    speed ≤ 0; trim state read-only.
  - stability (25): Jacobian of a linear map and of a nonlinear function;
    mass-spring-damper mode exact; real modes do not oscillate; neutral directions
    left out. At 18, 25 and 30 m/s: short period vs the classic approximation (2 %);
    phugoid frequency vs the refined approximation (1 %) with Lanchester 5–15 % high;
    phugoid damping (20 %) and > 2.5 times the constant-thrust value; roll subsidence
    vs L_p (3 %); Dutch roll (15 % / 20 %); spiral unstable as the criterion says,
    every other mode stable. Simulated phugoid period (1 %) and damping (3 %);
    simulated spiral growth rate (5 %).
  - fixed-wing autopilot (20): roll and pitch use the whole surface at the
    saturation error, with the right signs; outer loops slower by the separations;
    faster flight gives faster inner loops; `wrap_angle`; rejects invalid input;
    commands start at the trim. Closed loop: holds the trim and cures the spiral
    (10° of bank); course step 90° (settled within 0.5° by 15 s, altitude ±1 m);
    altitude step +20 m (overshoot < 3 m); airspeed steps to 20 and 30 m/s; fast
    descending reversal does not dive (regression); crosswind flown with the crab
    angle asin(5/25); reset repeats the flight. Orbit guidance: tangent on the circle
    (both directions), at the centre from far away, away from it inside; rejects
    invalid input; the autopilot holds a 150 m orbit (±0.5 m, bank atan(V²/gR) ±1°).
  - indoor geometry (56): box distances by hand (8 points: centre, face, past a
    face, edge and corner, inside); the box moves with its centre; yaw 90° turns
    its length to East, yaw 45° gives a diamond, yaw 30° shows the sense of the
    turn; turning shape and point together changes nothing (50 random, seeded);
    cylinder distances by hand (7 points); room clearance by hand (8 points) and
    `contains`; wall clearance by hand (9 points: the floor left out, below the
    floor, above the ceiling, 3-4-5 corners); it is never less than the clearance
    and equal in the upper half; rejects invalid input; read-only; for every shape
    (the room with and without its floor), many points at
    once equal one by one, the distance never changes faster than the point moves
    (1000 random pairs), and outside the convex ones the gradient has length 1.
  - ground contact (27): the Iris gear's corners and sizes (k = 3727 N/m, ζ = 0.5);
    rejects invalid gear (7 cases); read-only; no force above or on the floor;
    normal force = spring + damper and never pulls; friction against the slide,
    below μN (4 speeds); forces in body axes and their moment exact with the Iris
    banked 10°; a spinning Iris on a 30° heading braked by its four feet (exact
    moment); at rest it sinks mg/(4k) and is an equilibrium (1e-12); put down
    touching, it settles at 100 Hz (1e-6); every mode at rest stable with RK4 at
    100 Hz, ζ = 0.7 or v_s = 0.1 m/s unstable; the profile carries its gear; a drop
    matches the exact bounce (rebound 0.2 %, time below the floor 0.3 ms) with a
    restitution above the textbook one; a slide reaches 1 m/s in the closed-form time
    (1 %); a rough landing (0.5 m, banked, sliding, spinning) ends at rest and level.
  - take-off and landing (18): land detection rejects invalid settings (4 cases);
    every gain set has one; on the ground (holding, pushing down, climbing too
    slowly) the motors are stopped, the integrators zero and the state exactly at
    rest; waiting 1 s or 5 s before a take-off gives the same flight bit for bit;
    the take-off matches the same climb in the air (0.02 m/s after 0.3 s, 0.005 after
    1 s), straight up, never below the resting height; a 0.5 m/s landing touches at
    2 s, does not bounce, is detected within 2 s and ends at rest, motors stopped; a
    take-off after a landing (same autopilot) equals a fresh one (1e-6); an
    autopilot told it is flying finds the floor when asked to descend, and really
    hovers when asked to hold; no landing in the air (hover, cruise and brake, drop
    and stop, a 5 m/s descent); take off, land and take off again.
  - hitbox (15): the reach of a disc by hand (3-4-5, and 1.25 m where |c| + r gives
    1.40) and against its rim sampled every 0.1° (3 discs); box corners; the
    bounding sphere takes the farthest of rotors and body; the Iris sphere
    (0.3842 m) holds its four rotor discs and its collision box; the profile
    carries it; rejects invalid input (6 cases); read-only.
  - collisions (23): straight at the north wall, the ceiling, a crate and a pillar,
    the moment exact (1e-9) and the obstacle named by its index; a sphere ahead of
    the nose on a 90° heading; the earliest of two contacts in one step wins; a
    start inside a wall is reported at the first sample; the floor margin is the
    lowest foot nose up 20° on five headings (1e-12); a free fall onto the floor
    within g dt²/(8v) of the exact moment; standing 3 s and taking off to a hover
    are not crashes, coming back down is; reset; rejects invalid input (4 cases);
    tunnelling: a 2 cm slab seen at every phase with steps 1 % below 2R + w and
    jumped 1 % above it; grazing a 5 cm pillar at 20 m/s misses at most
    ρ − sqrt(ρ² − s²/4) (found within 5 %).
  - Kalman filter (20): the running mean and σ²/k exactly; the steady-state gains of
    Kalata's alpha-beta filter; the lag behind a constant acceleration (a T²/β and
    its two companions); two updates one after the other equal one stacked update
    (information adds up); the continuous noise does not depend on how an interval
    is split, the piecewise one does; NEES inside its χ² interval over 300 seeded
    runs, and far outside it for an overconfident filter.
- Examples (all run, `python -m examples.<name>`):
  - `projectile_drag.py` — vacuum vs drag.
  - `density_comparison.py` — constant vs exponential vs ISA density (~1.7 km apogee:
    ISA and exponential agree within ~1 %).
  - `density_comparison_2.py` — low-drag shot to ~15 km plus log-scale density
    profiles; the ISA bends at the tropopause.
  - `dzhanibekov.py` — intermediate-axis instability.
  - `tumbling_projectile.py` — tumbling body in vacuum, centre of mass on the exact
    parabola (deviation ~1e-11 m).
  - `wind_comparison.py` — body dropped from 500 m: still air vs uniform wind vs
    veering shear profile (ground tracks).
  - `wind_gust.py` — power-law mean wind + 1-cosine gust: the body lags the wind.
  - `iris_velocity_step.py` — Iris under the autopilot: 10 m/s North, then stop
    (speed, pitch, rotor speeds, altitude), plus the top speed (20.7 m/s at 45°).
- Physics note (intermediate-axis instability, torque-free): spin Ω about the
  intermediate axis grows any tiny perturbation at rate
  `λ = Ω·sqrt((I2−I1)(I3−I2)/(I1·I3))`. The example perturbation (0.01 rad/s on p
  and r) is an arbitrary stand-in for real imperfections, not a physical constant.
  The flip time grows only with the logarithm of the perturbation size
  (~ln(10)/λ ≈ 2 s per 10× smaller for I = diag(1,2,3), Ω = 2); it shrinks with
  faster spin and larger λ. Exactly zero perturbation never flips. Spin about the
  smallest or largest axis is stable.
- Still open, on purpose (each has its moment):
  1. After `stop` fires, the last sample overshoots the event (e.g. ~v·dt below
     the floor). Collisions now give their moment interpolated inside the step;
     gate crossings will too (phase 8, step 5).
  2. The ISA functions are scalar-only (`if` on the altitude, `math`). Vectorise
     with `np.where` in the batch engine (phase 9). Indoors the density is
     constant over the height of a room.
  3. `RotatingBody` and `RigidBody` still duplicate Euler's equations (inertia
     validation is already shared). Revisit when a vehicle needs attitude-only runs.
  4. Mixer saturation is plain clipping. It already happens for an instant at every
     velocity step (one rotor at 0, another at the maximum). A smarter strategy
     (give up yaw before roll and pitch, as PX4 does) when it starts to matter —
     an aggressive racing policy will saturate often (phase 10–11).
  5. Motor dynamics (first-order lag, 12.5 ms up / 25 ms down on the Iris) not
     modelled. Needed before a policy can move to a real drone (phase 12); it adds
     one state per rotor.
  6. Body drag is the weakest number: generic PX4 ballistic coefficient, and a
     placeholder for the vertical axis. The resulting top speed (20.7 m/s at 45°)
     is plausible but not validated against any Iris measurement.
  7. Rotor effects left out: axial-inflow thrust loss, rolling moment, gyroscopic
     moments; rotor inertia (1–3 %) left out of the Iris inertia. Ground and
     ceiling effect (more thrust close to a surface) matter indoors (phase 12).
  8. Controllers run at the simulation rate (1/dt). A policy will likely run slower
     than the rate loop (e.g. 50 Hz over 100 Hz or more); settle it in phase 10.
  9. An agile multirotor profile (thrust-to-weight around 5–10, as data with open
     sources): the Iris (1.9) is enough to start, not to race.
  10. No position loop: the velocity loop holds vertical speed, not altitude (the
      Iris climbs ~0.6 m during a 10 m/s cruise). The baseline of phase 10 may need
      one to hold a point or follow a line.
  11. The attitude loop treats the whole rotation vector with per-axis gains; PX4
      gives the tilt priority over yaw ("reduced attitude", `MC_YAW_WEIGHT`). Not
      needed so far (tilt recovers in ~0.6 s with heading errors present).
  12. With the slow PX4 horizontal integral, braking leaves a small velocity tail
      while the drag compensation unwinds (a few tenths of m/s for a few seconds).
      PX4 reduces this with feedforward from its trajectory generator.
  13. The top speed of a multirotor depends on the flight path (in a climb the
      velocity loop gives more thrust, and the same tilt more horizontal force), so
      it is a function of the manoeuvre, not one number. 20.7 m/s is level flight
      only.
  14. Walls, ceiling and obstacles are detected but exert no force: a collision
      ends the run (`CollisionMonitor` as `stop`). If a later task needs to bounce
      off a wall, it needs a contact model like the floor's.
  15. The source's stall gives C_L_max ≈ 2.2 near α ≈ 22° (C_L_alpha = 5.61 and
      α₀ = 27°), high for a small UAV (typically 1.2–1.5). Irrelevant while the
      aircraft flies at α < 10°; revisit if it has to manoeuvre hard.
  16. The side force and the lateral moments use β in body axes, and drag has no
      side component: the small-β simplification of the source.
  17. Control-surface limits live only in the autopilot (±30°): the vehicle model
      accepts any deflection, and the trim at low speed asks for more (−22° of
      elevator at 18 m/s, −35° at 15 m/s). Move the limits into the vehicle when a
      second controller (or a policy) could exceed them.
  18. F_y differs from the source's case 3 by 0.0015 N (3e-5) with every input it
      depends on matching to 1e-9. Not explained; the test tolerates it with a
      comment.
  19. The fixed-wing autopilot gains are designed at one trim (the operating speed)
      and not rescheduled with speed. Tested from 18 to 30 m/s.
  20. The course loop uses the airspeed in its gains (Beard & McLain use the ground
      speed): equal in still air, different in a strong wind.
  21. Speed: ~1 000 steps/s for the Iris with its autopilot at 100 Hz on one core
      (~10× real time, measured in Claude's workspace at the end of phase 7).
      Training needs much more: the batch engine of phase 9.
  22. Stochastic turbulence (Dryden/von Kármán) is not modelled; indoors it only
      matters as the drone's own wash off walls and floor, which is beyond this
      model.
  23. Floor contact at 100 Hz is stable but coarse for hard impacts (a bounce lasts
      ~5 steps; −1.9 % in rebound height), and friction is soft below 0.2 m/s (a
      foot creeps under a slow sideways push). Both come from the step: sub-steps
      for the contact, or an implicit friction, if a task needs better.
  24. Only multirotors have a landing gear; the fixed wing has no contact model.
  25. The floor is judged by the landing gear only: a drone banked hard or upside
      down close to the floor touches it with a rotor first, and the crash is seen
      only when a foot (or its centre of mass) gets there.
  26. Collisions are checked at the samples: fine for the Iris (79 m/s to jump a thin
      wall; grazes up to ~1.3 cm deep at 20 m/s go unseen). A small fast profile
      (radius 0.1 m, 30 m/s) would jump a thin wall at 100 Hz: then check the
      segment between samples (conservative advancement) or sample faster.
- Next: phase 8, step 4b (shape of the drone with several spheres). The step plan is
  in section 5.

## 8. Working with Claude

- **Language:** replies in Spanish; code, comments and commits in English.
- **Teaching first:** explain every new concept (physics, maths, machine learning,
  Python, Git or commands) before using it, with intuition and analogies. The
  author writes or pastes the code, runs it locally and reports back. Visual
  explanations are welcome when a concept is hard to picture.
- **Small verified steps:** each step ends with tests passing (`pytest -n auto`),
  ruff clean, and a commit + push.
- **Accuracy:** Claude checks what it states against the real repo and by running
  the code (test counts, numbers, behaviour) before handing it over. When something
  could not be verified, it says so.
- **Honesty:** if something is wrong or could be better, Claude says so even when
  not asked.
- Claude never commits to the repo (its access to GitHub is read-only): it hands
  over complete files as a zip per step with the repo's folders, which the author
  keeps in `..\buffer` and expands over the repo
  (`Expand-Archive -Path "..\buffer\<zip>" -DestinationPath . -Force`). Files to
  delete or move are named with their `git rm` / `Move-Item` command. When a change
  touches several places of one file, Claude gives the whole file: partial
  insertions lose imports and blocks. Each step ends with two blocks of commands:
  the checks, and then, once they passed, the commit (one-line message) and push.
- Claude checks every push against its own copy (same files, `ruff check`, the
  tests) before going on. Its working copy carries local commits that never leave
  the workspace. Package versions in Claude's workspace can differ slightly from
  `requirements.txt`; the author's run is the reference.
- **End of each phase:** update this document (roadmap, conventions, decision log,
  current state, loose ends), hand it over to be committed, and update the copy in
  the Project.
- **Session hygiene:** Claude must tell the author when to start a new chat — at
  the end of a phase block, or when the conversation is getting long enough that
  older details risk being lost. Before switching, make sure this document is up
  to date. A new chat starts inside the "Jineta" Project with:
  "Seguimos con Jineta, fase N: <topic>".
- If a chat stops or breaks mid-task, open a new one in the Project: the repo and
  this document are the source of truth, not the chat history.
- **A phase plan is a contract.** When a phase spans several chats, the step list
  in section 5 (with its ✅ marks) is followed as written; changing it is a decision
  to discuss and record, not something a new chat does on its own.
- Vehicle parameters come from open sources, each number with its source in a
  comment.
