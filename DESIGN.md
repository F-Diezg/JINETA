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

**Where it stands (end of phase 11):** the physics engine is built and validated
(phases 1–6c): integrators, rigid body, atmosphere and wind, a quadrotor with its
cascaded autopilot, a fixed-wing aircraft with its own autopilot, and a Kalman
filter. The indoor world is built (phase 8): rooms with obstacles and gates read
from course files, take-off from the floor, collisions with a 13-sphere hitbox,
a referee that times a lap. The batch engine (phase 9) flies many drones at
once, equal to the reference engine to rounding. The training environment
(phase 10) commands collective thrust and body rates at 50 Hz, as PX4 accepts
them from a companion computer, and the classical baseline flies the first
course in 6.77 s. **The MVP works (phase 11):** PPO, written in NumPy in the
project, taught the Iris to fly the first course in **3.23 s**, half the
baseline's time with the same thrust and rate limits, never closer than 13 cm to
anything; the reference engine replays the lap exactly. Random courses with
gates and obstacles, a policy that sees the nearest surfaces, and training on
many courses at once are built; a pilot for any course needs a longer training
than one session here (loose end 40). Next: phase 12, realism for the real
drone.

Name: *Jineta* is a working name until the project has its own.

Author: 3rd-year aerospace engineering student. First large software project, so
code should be clear and every concept explained before it is used.

## 2. Environment

- OS: Windows (two machines: laptop + desktop, synced via GitHub)
- Python **3.13** in a `.venv` virtual environment
- The author's computer has 8 cores; a training uses one (loose end 21). Several
  trainings can run at once, one per PowerShell window, each started after
  `$env:OPENBLAS_NUM_THREADS = 1` (otherwise NumPy's threads compete for the
  cores).
- Editor: VS Code (extensions: Python, Pylance, Ruff, GitLens)
- Dependencies: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `pytest`, `pytest-xdist`,
  `ruff`, and `gymnasium` (1.4.0, phase 11; only the adapter and its tests use
  it, and those tests are skipped without it), pinned in `requirements.txt`.
  After pulling on the other computer, run `pip install -r requirements.txt`
  when `requirements.txt` changed: `git pull` brings the list, not the packages.
  The learning algorithm is the project's own (`jineta/doma`, NumPy): no PyTorch.
- Training outputs (`runs/`) are ignored by Git; a trained policy kept in the
  repository lives in `policies/` (a .npz of ~120 kB).
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
- The full test run, in one process, takes ~3.5 min (836 tests, in Claude's
  workspace at the end of phase 11, whose speed changes from session to session);
  the slowest are PPO's regulator test (~14 s), the training tests (~5-15 s) and
  the batch tests that fly the first course against the reference (~5-11 s).
  `pytest -n auto` divides that by the number of cores.
- `scipy` is used for the trim (`scipy.optimize.root`) and in the Kalman tests
  (`scipy.stats.chi2`).
- Regenerate dependencies with `pip freeze | Set-Content -Encoding ascii
  requirements.txt`. PowerShell's `>` writes UTF-16, and `Out-File -Encoding utf8`
  in Windows PowerShell 5 adds a BOM (an invisible mark at the start of the file;
  pip tolerates it, but it does not belong there). Package names are ASCII.
- A block of commands pasted into PowerShell does not stop when one line fails:
  the next lines still run. So the checks come first, and the commit only once
  they passed.
- After changing code on purpose to see a test fail, delete the `__pycache__`
  folders (`Get-ChildItem -Recurse -Directory __pycache__ | Remove-Item -Recurse`):
  Python reuses its compiled copy when the file has the same size and the same
  time to the second.
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
  with it, the floor with the landing gear (a crash once the vehicle has left the
  floor, every gear point more than `TAKEOFF_CLEARANCE` = 1 cm up, and touches it
  again). `CollisionMonitor` is a `stop` for
  `simulation.run`; it checks every sample and, for walls, obstacles and gates,
  the straight move of each sphere between two samples (conservative advancement,
  a contact within `SWEEP_TOLERANCE` = 1 µm); it reports the first contact
  (`Collision`: kind "room", "obstacle", "gate" or "floor", index, time and
  position inside the step).
- **Gates** stand upright and are crossed along their heading `yaw` (like a box's
  yaw). Their own axes: n along the heading, u to the right, v down. A gate is
  crossed when the centre of mass goes from n < 0 to n >= 0 between two samples and
  the point where it cuts the plane is in the opening (edges included).
- **Course files** (`courses/*.yaml`, read by `load_course`) are strict: unknown
  or missing keys, wrong types or sizes, anything outside the room and a start
  inside an obstacle are errors that name the place (`gates[1].width`). Positions
  NED (z down: 1.5 m up is z = -1.5), angles `yaw_deg` in degrees, stored in
  radians. The format is in the docstring of `campo/course.py`, and a test reads
  that example as a course so the two cannot drift apart.
- **A lap** is judged by a `Referee` (the `stop` of the run): it starts at t = 0
  and ends "finished" (last gate crossed; lap time = that moment), "crashed" or
  "timeout". Gates count only in order and forward. It takes
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
  `RatePilot(vehicle, rate_gains, dt, landed)` is the other hook: it takes
  `thrust_command` [N] and `rate_command` [rad/s] (what a policy commands, and
  what PX4 accepts from a companion computer) through the rate loop; while landed
  its integral is held at zero, and it takes off when the thrust exceeds the
  weight. Both have a batch version (`BatchAutopilot`, `BatchRatePilot`).
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
- **Batch engine (`escuadron`):** the states of n drones are the rows of an
  (n, 13) array; every batch class takes n (`BatchMultirotor(params, n)`, ...), and
  all its drones share one vehicle profile and one course. Each drone's memory
  (integrals, landed, previous sample, collision, passes) is its own row, and
  `reset(drones)` takes all (None), a boolean mask (n,) or indices. No operation
  mixes rows (sums over rotors or feet are loops over them, 3 x 3 products are
  `core.vector.matvec`), so a drone gives the same numbers alone or in any batch.
  Every batch module mirrors a reference one term by term and is tested against it
  to rounding (1e-12 per call, 1e-9 over flights); the reference stays the truth.
  An `if` per drone becomes a mask (`np.where`, all branches computed). Times can
  be one per drone (each on its own lap clock). Environment models are the
  reference's, called with arrays (altitudes (n,), positions (n, 3)).
- **Training environment (`picadero`):** `CourseEnv(course, n, profile, limits,
  weights, dt, action_repeat, lookahead)` has the shape of Gymnasium's interface
  without depending on it: `reset(seed)` → (observations (n, size) float32, info);
  `step(actions (n, 4))` → (observations, rewards (n,), terminated (n,),
  truncated (n,), info); `observation_space`, `action_space` (`BoxSpace`). Each
  drone is its own episode: one lap from standing on the floor; terminated by a
  crash or the finish, truncated by the time limit; restarted in the same step
  (its last observation in `info["final_observation"]`). The action, four numbers
  in [-1, 1], is collective thrust and body rates inside an `ActionLimits`
  envelope (by default the autopilot's); it is held for `action_repeat` physics
  steps (2: the policy acts at 50 Hz over the 100 Hz rate loop). The observation
  is in body axes (velocity, attitude matrix, rates, four points on each of the
  next `lookahead` openings, previous action), SI units. The reward of a physics
  step is progress towards the gate targeted at its start + a bonus per gate - a
  penalty per crash (`RewardWeights`); an environment step adds up its physics
  steps. A classical controller can fly through the same interface
  (`ActionLimits.actions`).
- **Training (`doma`, `picadero.training`):** the learning algorithm (`doma`:
  network, policy, PPO) imports only NumPy and knows nothing about drones; it
  trains on anything with `CourseEnv`'s shape. An `Agent` (observation
  normaliser, Gaussian policy, critic) is what training produces and what flies
  (`agent.act(observations)`: the mean action; with a generator, drawn from the
  policy); it is saved to one .npz with an `info` dictionary (what it was trained
  on, how). `CourseEnv` options of phase 11: `margin` (every hitbox sphere that
  much larger: a safety margin, 5 cm in training), `nearest` (the nearest
  surfaces in the observation, 4 on random courses), and the reward's `time`
  weight (0.25 per second in training). A policy is measured by its deterministic
  lap (status, lap time, closest approach) and by the success rate of laps flown
  with its own noise; on random courses, on courses it never saw (seed 2000; the
  courses that choose the best agent during training are seed 1000).
- **The classical baseline** of a course is the fastest robust lap of the
  velocity autopilot following the path through its gates (`picadero.baseline`,
  `sweep`): every copy with its settings changed by 2 % finishes, never within
  5 cm of anything. A policy is measured against it on the same course.
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
│   │               #   and obstacles by signed distance), gates.py (frames and
│   │               #   crossings), collision.py (collisions in flight),
│   │               #   referee.py (gates in order, lap time, end of a lap),
│   │               #   course.py (course files)
│   ├── jinete/     # vehicles: aero, multirotor, control (multirotor autopilot),
│   │               #   contact (landing gear on the floor), hitbox (collision
│   │               #   spheres), fixed_wing, propulsion, trim, stability,
│   │               #   fixed_wing_control, profiles
│   ├── atalaya/    # estimation: kalman.py (later: onboard sensors)
│   ├── escuadron/  # batch engine: quaternion, rigid_body, multirotor, contact,
│   │               #   control (whole autopilot, rate pilot), collision,
│   │               #   referee, simulation (BatchFlight: n drones flying laps,
│   │               #   reset per drone)
│   ├── doma/       # learning: network (MLP, backpropagation, Adam), policy
│   │               #   (Gaussian policy, critic, normaliser, Agent), ppo
│   └── picadero/   # training layer: spaces (action, observation, surroundings),
│                   #   reward (and end of an episode), env (CourseEnv), baseline
│                   #   (the classical path follower and its sweep),
│                   #   gymnasium_adapter, training (train, evaluate), courses
│                   #   (random courses, many courses at once)
├── tests/
├── courses/        # course files (YAML)
├── policies/       # trained policies kept in the repository (.npz)
├── examples/       # runnable scripts (python -m examples.<name>)
├── pyproject.toml  # pytest + ruff configuration
├── DESIGN.md
└── README.md
```

Planned (each package arrives with its phase; names are proposals, settled when the
phase starts):

- `atalaya/` grows **onboard sensors** (IMU, position from a motion-capture system
  or a camera) for the realism phase (12).
- a **visualisation layer** outside `jineta/` (phase 13), which reads recorded
  flights and never runs the physics.

Dependency rule: `core` imports nothing of the project; `campo` and `jinete` import
`core`; layers that compose others (world, batch engine, training) sit on top and
are never imported by the engine: `escuadron` imports `core`, `campo` and
`jinete`; `doma` imports only NumPy; `picadero` imports all of them and nothing
imports `picadero`.

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
| 8 | Indoor world: room, floor, gates, obstacles, collisions; course files | ✅ Done |
| 9 | Fast batch engine (many drones at once), checked against the reference | ✅ Done |
| 10 | Training environment (observation, action, reward) + classical baseline | ✅ Done |
| 11 | **MVP:** the Iris learns the course (reinforcement learning) | ✅ Done |
| 12 | Realism for the real drone: motor lag, ground effect, sensor noise, latency, randomised parameters | 🔄 In progress |
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

**Phase 8 — done.** The indoor world: geometry by signed distance, ground contact
and a landed state for the autopilot, collisions with a hitbox of spheres, gates,
a referee and course files, and a first course flown by the Iris. A step 4b was
added during the phase (the drone's shape). Tests: 594, ruff clean.

Plan agreed with the author at the start of the phase (each step: tests green,
ruff clean, commit + push). Decided with the author: the
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
   `Room.wall_clearance`, `IRIS_BOUNDING_SPHERE`): the drone's spheres against walls,
   ceiling and obstacles; the floor by the landing gear, a crash once the drone has
   left it; the moment of contact interpolated inside the step. Tests: straight at a
   wall, the ceiling, a crate or a pillar, the moment exact (1e-9); spheres placed
   with the attitude; the earliest contact inside a step wins; a free fall onto the
   floor within g dt²/(8v); standing still and taking off are not crashes, coming
   back down is; the tunnelling limit measured (a thin wall needs a step longer
   than 2R + w, 79 m/s for the Iris; a graze misses at most the depth of a chord as
   long as the step, 1.2 cm at 20 m/s).
4b. ✅ **Shape of the drone with several spheres** (added during step 4;
   `rotor_spheres`, `box_spheres`, `union`, `IRIS_HITBOX`): one sphere per rotor and
   nine for the body (3 x 3 cells of its collision box) instead of one around
   everything: 0.275 m high instead of 0.769 m. Tests: the rotor cylinders and the
   whole collision box inside the spheres (sampled); its reach up, down, forward and
   sideways by hand; climbing to the ceiling it touches 0.233 m higher, rolled 90°
   0.197 m sooner than level; nose first into a wall at 0.280 m; take-off and hover
   still no crash; the tunnelling limit measured again (58 m/s head on; grazes up to
   3.1 cm at 20 m/s).
5. ✅ **Gates, rectangular and circular** (`jineta/campo/gates.py`,
   `CollisionMonitor(..., gates=...)`): crossing between samples, through the
   opening and in the right direction, with the fraction of the step interpolated;
   the frame (four square bars, or a torus) is an obstacle reported as "gate".
   Tests: frame distances by hand (3-4-5, 5-12-13), the rectangle equal to its four
   bars and the torus to its centre circle (brute force); the yaw is a heading;
   distance properties; straight, slanted and backward crossings (exact fractions),
   moves that miss the opening, edges, the opening judged where the plane is cut;
   one crossing in a sampled flight at the exact moment; a frame hit as "gate".
6. ✅ **Course and referee** (`jineta/campo/referee.py`): gate order, lap time,
   end by crash, finish or time limit. Tests: three gates in order with exact
   moments; a later gate first and a gate crossed backwards do not count (and the
   gate can be flown again); a crash ends the lap; a finish 2 ms before a crash in
   the same step is a finish, a crash 2 ms before a gate is a crash; time limit;
   reset repeats a lap.
7. ✅ **Course files in YAML** (`jineta/campo/course.py`,
   `courses/first_course.yaml`, `jinete.contact.resting_height`): a strict format
   with clear errors, and a first course (18 x 10 x 4 m, four gates, a pillar and
   two crates). Tests: the first course and the documented format load; optional
   parts; 19 mistakes in the format and 7 in the layout, each with its place in the
   message (and the z-down hint for a gate below the floor); the file is named in
   the error; the start stands on the floor at its heading; the course builds its
   monitor and referee; a straight lap of a course finishes at the exact moment.
8. ✅ **See a flight** (`examples/iris_course.py`): the Iris takes off from the
   floor, flies at 3 m/s straight at each gate centre with its velocity autopilot
   and finishes the first course in 8.81 s; the figure shows the course from above
   and from the side, the path coloured by speed and the crossings. Tests: the lap
   finishes with the four gates in order and no contact (lap time as a regression
   value, ±0.02 s); it starts on its gear and stays inside the room; every crossing
   point is on the gate's plane, inside the opening.
9. ✅ **Close the phase**: this document and the README.

**Phase 9 — done.** The batch engine (`jineta/escuadron/`): the rigid body, the
multirotor forces, the floor contact, the whole autopilot, the collision monitor
(with the swept check) and the referee for n drones at once, and the batch loop
with each drone on its own lap clock and reset on its own; every part equal to
the reference to rounding, a drone's numbers independent of its batch. The
reference was made 1.7 times faster without changing a number (`core.vector`) and
its collisions are now checked between samples too (loose end 26). Measured
speed: ~40 times the reference from ~1 000 drones (~440 times real time at 1 024
drones on the author's computer). Tests: 688, ruff clean.

Plan agreed with the author at the start of the phase
(each step: tests green, ruff clean, commit + push). Decided with the author: the
batch engine carries the whole multirotor cascade (velocity, attitude, rate, landed
state), not only the rate loop the policy needs; all the drones of a batch fly the
same course (arrays shaped so that a course per drone can come in phase 11); and
loose end 26 (grazes missed between samples) is closed in this phase, in the
reference first. The batch engine lives in `jineta/escuadron/` (a squadron: many
riders moving together).
1. ✅ **Measure the reference, and speed it up without touching the physics**
   (`examples/benchmark.py`, `jineta/core/vector.py`): a lap of the first course
   timed (best of several runs) and profiled. `np.cross` took over half of the
   lap; `vector.cross`, the formula written out, gives the same numbers bit for
   bit and the lap runs 1.7 times faster with exactly the same states. Tests:
   equal to `np.cross` bit for bit for single pairs and stacks with broadcasting
   (components over 16 orders of magnitude), basis vectors, perpendicular to both,
   wrong shapes rejected.
2. ✅ **Batch rigid body** (`escuadron/quaternion.py`, `escuadron/rigid_body.py`,
   `core.vector.matvec`): rotation matrices, quaternion derivative, Euler's
   equations and the renormalisation for N drones at once, states (N, 13). Tests:
   every drone equal to `RigidBody.derivative` (500 random seeded states, full
   inertia tensor, 1e-12; measured 4e-15); 2 s of RK4 with state-dependent forces
   equal to the reference flights (1e-12; measured 7e-15); the quaternion
   derivative bit for bit, matrices and normalisation within 1e-15; a drone alone
   or among 1000 gives the same numbers bit for bit; free fall of tumbling bodies
   exact; one time per drone reaches the force model; invalid mass and inertia
   rejected, inertia read-only.
3. ✅ **Batch multirotor forces and floor contact** (`escuadron/multirotor.py`,
   `escuadron/contact.py`): rotors through the mixer, rotor drag, body drag, the
   landing gear. Tests: every drone against `Multirotor` with `GroundContact`
   (400 states from 0.3 m up to 0.1 m below the floor, any attitude, rotor
   commands below zero and above the maximum; with the defaults, and with a density
   changing with height and a constant wind; 1e-12), and without contact; the
   contact alone against the reference on two floor heights (feet touching, in the
   air, and sinking points the floor does not push); the Iris standing on its gear
   is at rest (1e-12); a drone alone or among 1000 bit for bit; the 882 rotor
   commands of the reference lap replayed by three drones (open loop): each within
   1e-9 of the reference states (measured 2.1e-13), the three identical bit for
   bit.
4. ✅ **Batch rate loop** (`escuadron/control.py`: `BatchRateController`,
   `rotor_speeds_for_wrench`): integrators, anti-windup, mixer clipping, reset per
   drone. Tests: the mixer's inverse with both clips against the reference (500
   wrenches); 300 updates of 50 drones with random commands and states, rotor speeds
   and integrals equal to the reference's, the anti-windup reached; 3 s of flight
   of 12 drones with commands changing every 0.5 s, each within 1e-9 of its
   reference flight (measured 1.6e-14, up to 29 m/s and 4 rad/s); the exact
   discrete first order of the P-only loop on each axis; a drone alone or among
   1000 bit for bit; reset of some drones (indices, mask) or all.
5. ✅ **Batch attitude and velocity loops, landed state and land detection**
   (`BatchAttitudeController`, `BatchVelocityController`, `BatchAutopilot`;
   quaternion product, conjugate, rotation vector and `from_matrix` on stacks): the
   whole `Autopilot` for N drones. Tests: the quaternion functions against the
   reference on random attitudes, both signs of w, tiny rotations, half turns and
   half turns tilted 1e-6 (every branch of Shepperd's method); the attitude loop
   (clipping included) and 200 updates of the velocity loop for 40 drones (tilt,
   thrust and integral limits reached) equal to the reference's; four drones of one
   batch standing on the floor wait, ask to climb too slowly, take off at different
   moments, climb, turn, descend and land: each within 1e-9 of its reference flight
   (measured 4.8e-14) and landed at the same steps; a drone alone or in the batch
   bit for bit; drones that stay landed keep their loops at zero, a drone declared
   landed in the air takes off from clean loops, reset of some drones keeps
   `landed`; the lap of the first course flown by three drones of a batch with the
   reference's gate order: each within 1e-9 of the reference lap to its last step
   (measured 3.9e-14), the three identical bit for bit.
6. ✅ **Collisions between samples in the reference** (loose end 26, closed;
   `CollisionMonitor` swept check, `SWEEP_TOLERANCE`): each sphere checked along
   its straight move between two samples by conservative advancement, not only at
   the samples. Tests: the thin wall the samples jump at 80 m/s is found at every
   phase, at the exact moment; grazes of a pillar at 20 m/s from 2 µm deep to past
   the deepest the samples miss, at 17 phases, all found within 2e-5 s of the
   first touch; passing 10 µm or 1 mm away is not a contact, within 1 µm is; a
   graze between samples beats a contact at the sample in the same step; the move
   is the chord (a sphere turned 90° in one step touches a post the chord crosses,
   at the exact fraction); far from everything nothing is advanced; the Iris's 13
   spheres' grazes are found. Every earlier collision, gate, referee and course
   test still passes unchanged (the lap of the first course: same states, same lap
   time, no advancement run).
7. ✅ **Batch collisions, gates and referee** (`escuadron/collision.py`:
   `BatchCollisionMonitor`; `escuadron/referee.py`: `BatchReferee`,
   `forward_crossings`, `batch_referee`): status per drone (codes FLYING,
   FINISHED, CRASHED, TIMEOUT), lap time, passes; a drone whose lap ended is not
   judged again (nor checked by the monitor) until it is reset. Tests: 95 sampled
   flights in the first course (random places, speeds, accelerations, headings,
   tilts and turns; drones taking off from the floor and coming back down, three
   standing on their gear all along, twelve grazing the pillar at 20 m/s from 5 µm
   to 2 cm deep, seven of them only between samples): every collision (kind,
   index, moment, place, 1e-9) and `airborne` as the reference monitor's, every
   kind of contact present; margins as the reference's; a drone alone or in the
   batch bit for bit; only the chosen drones checked and reset. Laps: the reference
   lap, the same lap moved sideways and up, flown backwards, and random flights,
   with the course's time limit and with 3 s: every status, end and pass (gate,
   moment, point, 1e-9) as the reference referee's (the reference lap finishes at
   8.8145 s); forward crossings of 3000 random moves through a rectangular and a
   circular gate as the gates' own; in one batch a finish 2 ms before a crash, a
   crash 2 ms before a gate, and a crashed drone reset onto a lap it finishes; a
   drone on its own clock repeats the lap time of another; ended laps are not
   judged again.
8. ✅ **Batch loop** (`escuadron/simulation.py`: `BatchFlight`;
   `examples/batch_course.py`: the gate guidance for n drones, `fly_laps`, and the
   speed against N): `step` (controller, RK4 and renormalisation, referee; returns
   the laps that ended in that step) and `reset` of chosen drones; each drone on its
   own lap clock; a drone whose lap ended stays where it ended. Tests: four Iris
   at 2, 2.5, 3 and 4 m/s in one batch fly the first course as the reference
   engine flies it at each speed (states within 1e-9 at every step, the same end,
   moment and passes; the 4 m/s one clips gate 2's frame, as measured in phase 8);
   a drone that crashes and starts its next lap at once repeats its first lap bit
   for bit, while the other drone flies exactly as in a batch where nobody is
   reset; each lap's end is reported once; `fly_laps` restarts each drone on its
   own; inconsistent input rejected. Speed (table below).
9. ✅ **Close the phase**: this document and the README.

**Phase 10 — done.** The training environment and the classical baseline, in a
new package `jineta/picadero/` (the riding school where horse and rider train):
the rate pilot with a landed state (reference and batch), the action and the
observation, the reward and the end of an episode, `CourseEnv` (n drones, each
its own episode, checked against the reference engine), and the baseline, whose
robust best lap of the first course, 6.77 s, is the time to beat. On the way the
batch engine got a third faster (loose end 29) and an unloading gear's hop stopped
counting as a take-off. Tests: 777, ruff clean.

Plan agreed with the author at the start of the phase (each step: tests green,
ruff clean, commit + push). Decided with the author: the environment has the
shape of Gymnasium's interface (`reset(seed)`, `step(action)` returning
observation, reward, terminated, truncated, info; spaces with limits and shape)
without depending on the library, which cannot be installed in Claude's
workspace; a thin adapter to the library chosen in phase 11 comes at its start
and is checked there with Gymnasium's own environment checker. The policy acts
at 50 Hz: each action is held for two steps of the physics and the rate loop
(100 Hz) (loose end 8). Loose end 29 (+40 % for the batch) is the first step.
1. ✅ **Speed: the overhang formula written out** (loose end 29,
   `campo.indoor._distance_from_overhangs`, used by boxes, the room, columns and
   gate frames): the two or three components added and compared one by one
   instead of NumPy's reductions. Tests: equal to the old formula bit for bit on
   100 000 random overhangs (2 and 3 components, 12 orders of magnitude, zeros,
   points inside) and one point at a time; every geometry, collision, gate,
   referee, course and batch test unchanged. Speed (Claude's workspace, before
   and after alternated twice in one session): the batch at 1 024 drones from
   ~59 000-64 000 to ~86 000-88 000 drone-steps/s (+~45 %); the reference lap
   unchanged within the noise (~1 500-1 600 steps/s both). On the author's
   computer (after the phase): 43 764 to 58 093 drone-steps/s (+33 %).
2. ✅ **Rate pilot with a landed state** (loose end 32; `jinete.control.RatePilot`,
   `escuadron.control.BatchRatePilot`; `campo.collision.TAKEOFF_CLEARANCE`): the
   entry point of a policy (collective thrust and body rates through the rate
   loop), with the landed state of PX4: while landed the rate integral is reset
   before and after each update; it takes off when the commanded thrust exceeds
   the weight, and never lands again by itself. Found on the way: a gear that
   unloads without a take-off (90 % of the weight) lifts its feet a fraction of a
   millimetre, which the monitor took for a take-off, and the feet coming back
   down for a floor crash. A vehicle has now left the floor once every gear point
   is more than 1 cm up (reference and batch). Tests: 2 s on the floor asked to
   turn charge nothing, while the rate loop alone charges its roll and pitch
   integrals to their limits; take-off at the first step above the weight, not
   at exactly the weight; a take-off after 1 s on the floor is the one a fresh
   pilot flies from that state, bit for bit; in the air it is `RateController`
   bit for bit; a pilot declared landed with charged integrals starts from a
   clean loop; it never lands by itself; eight drones on the floor and in the air
   taking off at different moments, each within 1e-9 of the reference and landed
   at the same steps; alone or in a batch bit for bit; reset per drone; in the
   batch loop a drone at 90 % of its weight stays on its gear without a crash;
   hops of 0.2 mm and 9.9 mm are not take-offs, one of 10.1 mm and back is a
   floor crash (reference and batch monitors). Checked by breaking the code (7
   mistakes: no clearance, a reset left out before or after the update in either
   pilot, the take-off at 90 % of the weight, at exactly the weight); three
   survived the first tests and each got a test (the reset after the update is
   only visible in the integrals right after the call; the reset before only in a
   pilot declared landed with charged integrals; the threshold only at exactly the
   weight).
3. ✅ **Action and observation** (`picadero/spaces.py`: `BoxSpace`,
   `ActionLimits`, `autopilot_envelope`, `opening_points`, `Observer`): the action
   is four numbers in [-1, 1] (clipped) mapped to collective thrust and body
   rates, by default inside the autopilot's own envelope (thrust 10-90 % of the
   maximum, 2.83-25.44 N for the Iris; rates 3.5 / 3.5 / 1.5 rad/s), so the
   comparison with the baseline measures the pilot, not the limits; `actions` is
   the inverse, for a classical controller flying through the same interface. The
   observation (43 numbers with two gates ahead), in body axes: velocity, attitude
   matrix row by row, body rates, four points on the edge of each of the next two
   openings (a rectangle's corners, the square inscribed in a circle; top left,
   top right, bottom right, bottom left), the last gate repeated where the course
   runs out, and the previous action; SI units, float64 here and float32 at the
   environment's interface. Tests: spaces (shape, limits, contains, seeded
   samples, read-only, bad limits); actions by hand, clipped, inverse, the Iris
   envelope (hover inside it); every opening's points by hand (gate turned 30°,
   circle) and on the edge of the opening in the right order; observations by
   hand nose North and nose East; the attitude part is the rotation matrix; for
   200 random attitudes the body offsets rotate back to the NED points; targets
   repeat the last gate; many drones equal to one by one bit for bit. Checked by
   breaking the code (6 mistakes: the gate's u axis, R for R^T in the velocity and
   in the offsets, wrapping instead of repeating the last gate, R^T in the
   attitude part, the thrust mapping); each made a test fail.
4. ✅ **Reward and end of an episode** (`picadero/reward.py`: `RewardWeights`,
   `RewardTerms`, `Reward`, `episode_end`): per step of the physics, progress (the
   metres closer to the centre of the gate targeted at the start of the step, so
   crossing it gives no jump), a bonus per gate crossed and a penalty for a crash;
   starting weights 1 per metre, 1 per gate, 5 per crash (to be tuned in phase
   11). Terminated by a crash or the finish, truncated by the time limit. Tests: a
   straight lap of three gates 5 m apart, sampled every 0.1 m, gives 0.1 m of
   progress at every step (crossing steps included), 14 m in all and three
   bonuses at the crossing steps; through a gate backwards: no bonus and the
   progress back to zero; into a wall: the penalty once and nothing afterwards;
   the weights scale the terms; the gate guidance's lap of the first course adds
   up gate by gate to the distances between where the drone started flying to
   each gate and where it was just after crossing it (17.29 m, against 17.93 m
   from the start through the four centres); drones not flying at the start of a
   step get nothing; many drones equal to one by one; bad weights rejected.
   Checked by breaking the code (4 mistakes); a crash penalty not masked for the
   drones already ended survived the first tests and got its own.
5. ✅ **The environment** (`picadero/env.py`: `CourseEnv`;
   `examples/env_random.py`): n drones of one profile on one course, each its own
   episode (one lap from standing on the floor, pilot landed), `reset(seed)` and
   `step(action)` with Gymnasium's shape; the policy acts every `action_repeat`
   = 2 physics steps (50 Hz), the reward of an environment step is the sum over
   its physics steps; a drone whose episode ends is restarted in the same step
   (same-step autoreset: its last observation, status, lap time, return and
   length are in `info`); observations float32, rewards float64. Tests: the reset
   observation; one row per drone; least thrust stands on the floor for the whole
   time limit and is truncated, not a crash, with no reward; three scripted
   episodes against the reference engine (`Multirotor`, `RatePilot`, the course's
   referee, the reward recomputed from its states): every state within 1e-9, the
   same rewards, the same crash and final observation; a drone alone or among
   four (with restarts inside the batch) bit for bit; a crashed drone restarts
   standing landed, flies its next episode as a fresh environment's first bit for
   bit and its neighbour is not disturbed; a time limit of 0.5 s truncates at the
   25th step; seeds repeat; `action_repeat` = 1 with each action twice is the
   same flight and the same reward; the previous action in the observation,
   clipped; bad input rejected. Checked by breaking the code (6 mistakes): the
   reward of the last physics step only and the previous action not cleared on a
   restart failed; the pilot not set back to landed survived (the scripts took off
   at once) and got its own test; the loop not reset and the crash not masked are
   equivalent (a landed pilot resets its loop; the reward masks by itself); the
   target gate read after the step survived because no script crosses a gate: the
   lap flown through the environment in step 6 checks it. Random actions
   (64 drones, 10 s): every episode a crash after 0.84 s on average, return -5.0
   +- 0.5. Speed with random actions and restarts (Claude's workspace):
   ~210 drone-steps/s at n = 1, ~9 700 at 64, ~28 600 at 256, ~47 800 at 1 024
   (each is two physics steps: ~960 times real time at 1 024 drones); on the
   author's computer 32 475 at 1 024 (~650 times real time; table in the
   decision log).
6. ✅ **Classical baseline** (`picadero/baseline.py`: `BaselineSettings`,
   `GatePath`, `PathGuidance`, `BaselinePilot`, `baseline_flight`,
   `fly_baseline`, `sweep`; `examples/baseline_course.py`): the velocity
   autopilot at 100 Hz along a path through the gate centres (above the start,
   then entry, centre and exit `approach` metres along each gate's heading),
   followed by pure pursuit (the closest point searched a few segments ahead and
   never going back; the command aims at the point `lookahead` metres further on,
   which is the position loop of loose end 10 and the look-ahead of loose end 28)
   at a planned speed (cruise on straights; through a corner of angle theta, cut
   by an arc of radius lookahead / tan(theta / 2), at most sqrt(lateral_accel R);
   braking towards it at most braking_accel); straight up first to 0.5 m. Every
   setting is one value per drone, so `sweep` flies a grid of them at once (432
   settings x 5 drones: each with four copies whose settings are changed by up to
   +-2 %, seeded) and keeps the fastest lap that finishes robustly (every copy
   finishes and none comes within 5 cm of a wall, an obstacle or a frame). The
   first course: 135 of 432 settings finish, 54 robustly; the best (approach 2 m,
   cruise 5 m/s, lateral 8 m/s^2, braking 5 m/s^2, lookahead 3 m) finishes in
   **6.773 s, the time to beat** (straight at each centre: 8.81 s), top speed
   4.6 m/s, 6.8 cm from the nearest frame or obstacle. Tests: the path by hand
   (gate turned 30°, no turn at the centres), points along it and past its ends,
   the closest point searched only a few segments ahead; the speed plan by hand
   around a right-angle corner; straight up first, progress never going back
   (also when the drone is pushed back); the best lap (regression 6.773 +- 0.02 s,
   > 5 cm clear); the batch lap equal to the reference engine's (the same
   guidance driving `Autopilot`: every state within 1e-9, the same lap time);
   drones with different settings fly as alone; the baseline flown through the
   environment's actions (its velocity and attitude loops at 50 Hz giving thrust
   and rates inside the action envelope) finishes in 6.76 s, so the envelope
   allows a winning lap; at 100 Hz its return is the progress gate by gate plus
   four bonuses (1e-9), which also checks that the environment reads the target
   gate before the step (step 5's surviving mistake); the sweep keeps the robust
   setting and rejects 20 m/s^2 in the corners, and fails loudly when nothing
   finishes. Checked by breaking the code (7 mistakes); progress allowed to go
   back survived (the lap never pushes it back) and got its own test. Measured on
   the way: the fastest setting without the robustness test (6.48 s) cleared a
   frame by 6 mm and crashed in 7 of 33 copies changed by 2 %; the robust one
   finished all 65 such copies and 64 of 65 changed by 5 %. The sweep takes ~30 s
   in Claude's workspace.
7. ✅ **Close the phase**: this document and the README.

**Phase 11 — done.** The Iris learned to fly the first course with
reinforcement learning: PPO written in NumPy (`jineta/doma`, checked against the
LQR optimum of a double integrator), an adapter to Gymnasium, a time penalty and
a 5 cm safety margin found by measurement, and two-stage training. Its policy
(trained on the author's computer) flies the first course in 3.235 s against
the baseline's 6.773 s, 13 cm from everything at the closest, its lap replayed
exactly by the reference engine. Random
courses with gates and obstacles, the surroundings in the observation and
training on many courses at once are built and tested; a first training on them
(23 M steps) is not yet a pilot for any course (loose end 40). Tests: 836 (with
Gymnasium), ruff clean.

Plan agreed with the author at the start of the phase (each step: tests green,
ruff clean, commit + push). Decided with the author: the learning algorithm is
written in NumPy in a new package `jineta/doma/` (the *doma* is where a horse is
trained), because Claude's workspace has no PyTorch, Gymnasium or
Stable-Baselines3 and cannot install them, and every step must run and be tested
before it is handed over; the environment is also given to Gymnasium through a
thin adapter (an optional dependency), checked with Gymnasium's own checker on
the author's computer. The randomised courses have gates and obstacles, so the
observation grows the surroundings (loose end 33). The action envelope stays the
autopilot's (loose end 36), so the comparison with the baseline measures the
pilot. A prototype of PPO in NumPy, measured before planning (Claude's
workspace, 256 drones, 20 minutes): ~14 000 samples/s (the environment ~2/3 of
the time, the update ~1/3); after ~9 M steps it crossed gates and finished a few
laps (15.2 s), then settled on passing some gates and hovering until the time
limit: with the starting reward, hovering without a crash pays more than
risking the next gate (loose end 38).
1. ✅ **Adapter to Gymnasium** (loose end 34; `picadero/gymnasium_adapter.py`:
   `CourseGymEnv`, `CourseVectorEnv`, `to_box`): one drone as a `gymnasium.Env`
   (its last observation returned when an episode ends) and n drones as a
   `VectorEnv` with same-step autoreset (`final_obs`, `final_info` and their
   masks, Gymnasium's vector format). Tests (skipped without Gymnasium): it passes
   Gymnasium's `check_env`; spaces; one and n drones fly exactly as `CourseEnv`
   (bit for bit, final observations included); the same observations, rewards,
   ends, final observations and final infos as Gymnasium's own `SyncVectorEnv`
   over copies of `CourseGymEnv`; the baseline flown through it finishes and its
   info has the lap time. Checked by breaking the code (3 mistakes: the restarted
   observation returned at the end, no `super().reset(seed)`, the wrong final
   observations); each made a test fail.
2. ✅ **Neural network in NumPy** (`doma/network.py`: `MLP`, `Adam`,
   `orthogonal`, `clip_by_global_norm`, `save_arrays`, `load_arrays`):
   multilayer perceptron with tanh, backpropagation by hand, orthogonal initial
   weights (gain sqrt(2) in hidden layers), Adam with its bias correction,
   clipping by global norm, saving to .npz. Tests: the forward pass by hand;
   gradients against central differences on three shapes (1e-6); the initial
   weights orthonormal and repeatable; Adam three steps by hand (1e-15) and its
   first step the learning rate whatever the gradient; sin(x) on [-pi, pi]
   learned to an RMS error below 0.02; clipping; a network and its optimiser
   saved, loaded and stepped on bit for bit; a row alone or in a batch of 2048
   to 1e-13 (the plan said bit for bit: NumPy's matrix product adds in another
   order for other sizes, see the decision log); bad input. Checked by breaking
   the code (4 mistakes: tanh's slope, no bias correction, mean for sum in the
   bias gradient, the input gradient of the first layer); the last is
   equivalent (that gradient is never used).
3. ✅ **Gaussian policy and observation normaliser** (loose end 35;
   `doma/policy.py`: `RunningNormalizer`, `gaussian_log_prob`,
   `gaussian_log_prob_gradients`, `gaussian_entropy`, `GaussianPolicy`,
   `Agent`): the network gives the mean action, the spread (one log standard
   deviation per action number, the same for every observation) is learned; the
   observations are scaled by their running mean and variance (combined exactly,
   Chan et al.) and clipped to +-10; the `Agent` (normaliser, policy, critic)
   acts with the mean and is saved to one .npz. Tests: log-probability and
   entropy against `scipy.stats` (1e-12); their gradients against central
   differences; 100 000 samples with the policy's mean and spread, their average
   -log-probability the entropy; a new policy starts near zero; the normaliser
   fed in seven pieces equals the mean and variance of all the data (1e-12), and
   epsilon's effect on a tiny variance as computed; clipping; an agent saved and
   loaded acts bit for bit. Checked by breaking the code (4 mistakes: the
   log-probability without log sigma, the log sigma gradient without -1, the
   variances combined without the means' difference, acting without the
   normaliser); each made a test fail.
4. ✅ **PPO** (`doma/ppo.py`: `PPOSettings`, `gae`, `Rollout`, `Episodes`,
   `PPO` with `collect`, `update`, `learn`): experience from any environment with
   `CourseEnv`'s shape (same-step restarts; after a truncation the critic's value
   of the episode's last observation, after a termination nothing), advantages
   by GAE, the clipped objective, the critic's squared error, advantages
   normalised per minibatch, gradients clipped by global norm (0.5), the
   learning rate falling linearly to zero, the normaliser frozen during a
   rollout and updated after it; diagnostics (approximate KL, clipped fraction,
   explained variance, entropy, spread). Tests: GAE by hand, with lambda = 1 the
   Monte Carlo return minus the value, with lambda = 0 one step, cut at an
   episode's end (truncated and terminated); the experience of a scripted
   environment completed as each step ended; a continuous bandit's best action
   learned (RMS error < 0.03); the double integrator's optimal regulator: every
   episode cut after 8 steps, the learned feedback within 5 % of the discounted
   LQR gain of the Riccati equation (measured 1.6 %: [1.740, 1.769] against
   [1.723, 1.742]); the same seed the same training bit for bit; the learning
   rate schedule and the diagnostics; a short training on the first course.
   Checked by breaking the code (6 mistakes: truncations treated as ends (the
   regulator's gain falls to about a third), GAE not cut at ends, the clipped
   objective's gradient mask inverted, terminated steps given the next value,
   the probability ratio inverted, log_std never learned); each made a test
   fail. The experience test was added when it showed that no learning test
   had a terminated episode with gamma > 0.
5. ✅ **The Iris learns the first course** (`picadero/training.py`: `fly`,
   `evaluate`, `Evaluation`, `run_training`, `train`, `TRAINING_WEIGHTS`,
   `TRAINING_MARGIN`; `RewardWeights.time`; `CourseEnv(margin=...)` and
   `info["final_state"]`; `examples/train_course.py`,
   `examples/policy_course.py`; `policies/first_course.npz`): the reward gets a
   time term (a penalty per second of flight), the environment a safety margin
   (every hitbox sphere that much larger: coming that close to anything is a
   crash); training writes a log (CSV) and keeps the best agent (the fastest lap
   of the mean action that keeps the margin); an agent is measured by its
   deterministic lap (status, time, closest approach) and by the success rate of
   laps flown with its own noise. Five reward experiments (decision log) gave
   the recipe: progress 1, gate 1, crash 5, time 0.25 per second; 20 M steps
   without the margin, then 10 M with 5 cm, starting from the first stage's best
   agent (256 drones, ~45 minutes in Claude's workspace). **The policy flies the
   first course in 4.325 s (the baseline 6.773 s: 36 % faster), never closer
   than 15.8 cm to anything; 97 % of 256 laps flown with its noise finish (4.35 s
   on average); its actions replayed in the reference engine give the same lap
   (4.3247 s).** Tests: the time term counts the seconds flown while the lap goes
   on and nothing after; the baseline flown through `fly` (lap, gates,
   clearance, the recorded states ending where the lap ended, not where the
   drone was put back); drones standing (time out, no gate) and climbing into
   the ceiling (crash); the evaluation repeated by its seed; which evaluation is
   better (a lap that keeps the margin beats a faster one that does not); a
   short training writes its log and agents and is repeated by its seed; the
   margin turns the baseline's closest pass into a crash and a slightly smaller
   one does not; the saved policy finishes in 4.325 +- 0.05 s, more than 2 s
   faster than the baseline, more than 5 cm from everything, and the reference
   engine gives the same lap (1e-6). Checked by breaking the code (4 mistakes:
   the time penalty after the lap's end, the margin not applied, the final
   state taken after the restart, the margin ignored when choosing the best);
   each made a test fail. **After the phase, the author ran the same recipe on
   his computer (72.7 minutes): 3.235 s, 13.0 cm from everything at the
   closest, every one of 256 noisy laps finished (3.26 s on average); the
   reference engine gives the same lap (3.2346 s).** That policy replaced this
   one in `policies/first_course.npz` (decision log); the regression test is
   now 3.235 +- 0.05 s.
6. ✅ **Surroundings and randomised courses** (loose ends 33 and 30;
   `spaces.Surroundings`, `closest_points`, `CourseEnv(nearest=...)`;
   `picadero/courses.py`: `CourseRanges`, `random_course`, `CourseSetEnv`;
   `training.evaluate_courses`, `train_random`; `examples/train_random.py`):
   the observation can include the nearest surfaces (the room's six faces,
   every obstacle and frame; the vector to each closest point, in body axes,
   nearest first; 4 of them: 12 more numbers); random courses with 3 to 6 gates
   (in practice mostly 3 and 4: longer ones rarely fit the room) and 1 to 4
   obstacles beside the path, every one flyable along the baseline's path;
   training on k courses at once, n drones each, each course replaced by a new
   one in turn (its drones' episodes truncated); the best agent chosen on 8
   validation courses (seed 1000), measured on 20 test courses (seed 2000)
   against the baseline swept on each (a smaller grid, ~10 s a course: it
   finishes all 20, 6.29 s on average). Tests: closest points by hand (box face,
   edge and corner by 3-4-5; column; torus); the room's faces and the nearest
   first, nose North and East; an obstacle among them; the observation with
   surroundings keeps the old part and gives each drone the same alone or in a
   batch; 200 random courses keep their promises; a seed repeats a course; a
   cautious baseline (2.5 m/s) finishes six of them; many courses fly as their
   own environments until renewed, then truncated with their last observation;
   seeds repeat; a short training on random courses. Checked by breaking the
   code (5 mistakes: the surfaces left in NED axes, the farthest first, the
   closest point on the wrong side, a renewal without truncation, obstacles
   not kept off the path); each made a test fail. **Result: not yet a pilot
   that flies any course.** Trained here 23 M steps (2 courses of 128 drones at
   a time, the time penalty, no margin; ~70 minutes): the best agent on the
   validation courses (1 of 8 finished, at 15 M steps) finishes 0 of the 20 test
   courses (18 of their 68 gates); the last one 1 of 20 (4.61 s where the
   baseline flies 3.94 s), 27 gates. The return swung from update to update
   (only 2 courses per update), and late in the run episodes grew long with few
   crashes: the waiting habit again, this time until the course is renewed. The
   longer training on the author's computer, with 8 courses per update, is loose
   end 40.
7. ✅ **Close the phase**: this document and the README.

**Phase 12 — in progress.** Realism for the real drone.

Plan agreed with the author at the start of the phase (each step: tests green,
ruff clean, commit + push). Measured before planning (a throwaway prototype,
Claude's workspace): the phase 11 policy still finishes the first course with
the Iris's motor lag alone (3.27 s) or with 20 ms between what it sees and what
it commands alone (3.45 s), but crashes at the first gate with both, and with
40 ms of latency. Decided with the author:
- every effect can be switched on and is off by default, so the engine and the
  phase 11 policy give the same numbers bit for bit, and each step measures how
  much worse the phase 11 policy flies with its effect;
- sensors "B light": the policy sees what an onboard estimate would give, built
  from simulated sensors (a motion-capture system for position and attitude, the
  gyroscope for the body rates) and a Kalman filter for the velocity, with the
  latency to the motors; not a full EKF (the real one, PX4's EKF2, comes with
  PX4 software-in-the-loop in phase 14);
- the real drone is assumed to be positioned by motion capture (the usual lab
  setup; not decided yet, to be confirmed in phase 14: onboard vision would
  have other errors);
- no battery model in this phase: its effect (less thrust as the voltage falls)
  is covered by randomising the thrust, and a battery is modelled in phase 14
  from the real vehicle's data;
- the drone must be under control after the last gate (loose end 39) in this
  phase;
- the long trainings are run on the author's computer (8 cores); Claude's
  workspace runs short ones to check that everything works.

1. ✅ **Training on several processes** (loose end 21; `picadero/parallel.py`:
   `ParallelEnv`, `SerialEnv`, `make_parallel`, `CourseEnvPart`,
   `CourseSetEnvPart`, `worker_seed`, `WorkerError`; `train(..., workers)`,
   `train_random(..., workers)`; `--workers` in `examples.train_course` and
   `examples.train_random`; `examples/parallel_speed.py`): the environment
   split into parts, each in its own process (started with "spawn", the way
   Windows has), stepped at the same time and joined as one environment with
   all the drones; the learning stays in the main process. Tests: two workers of
   three drones give the same observations, rewards, ends and infos as one
   environment of six, bit for bit, restarts included; random courses in two
   workers as the same parts in one process (each part its own seed); a failure
   while building a part or in a step reaches the main process with its
   traceback, and the workers stop; a short training gives the same log with one
   worker and with two. Speed (Claude's workspace, 2 cores, 512 drones): one
   process ~30 000 steps/s, two workers of 256 ~33 000-35 000; two separate
   processes of 256 give 35 400 together, so the pipes cost nothing measurable
   and the gain is limited by batch size (a batch of 256 is less efficient per
   drone than one of 512). On the author's computer, 8 workers of 128 drones are
   estimated at ~3x one process of 1 024 (not measured: `python -m
   examples.parallel_speed`).
2. ✅ **Motor lag** (loose end 5; `jinete.multirotor.motor_derivative`,
   `Multirotor(..., motor_lag=True)`, `state_size`, `initial_state`,
   `rotor_speeds`; the same in `BatchMultirotor`; `MultirotorParams.
   motor_time_constant_up` / `_down`, the Iris's 12.5 / 25 ms from
   `iris.sdf.jinja`; `picadero/realism.py`: `Realism(motor_lag)`,
   `CourseEnv(realism=...)`; `examples/realism_policy.py`): each rotor's speed
   follows its command (clipped) as a first order, (command - speed) / tau, tau
   up while the command is above the speed and down otherwise (PX4's Gazebo
   motor model); with the lag the rotor speeds are part of the state, after the
   body's 13 elements (17 for the Iris), integrated by RK4 with the rest, and
   the forces use them. Without it, the state and every number are as before.
   Tests: the derivative by hand (up, down, both clips); a step of the command
   is the exact discrete solution of RK4 for every rotor (1e-9 rad/s) and within
   0.3 % of the continuous exponential speeding up (dt / tau = 0.8) and 0.02 %
   slowing down; after 0.1 s what is left is R(-0.8)^10 up and R(-0.4)^10 down;
   hover with the rotors at hover speed is an equilibrium (1e-12, 10 s within
   1e-9); the forces are those of instant motors at the state's speeds, bit for
   bit, whatever the command; without the lag the state is 13 and the command is
   used, clipped; 300 drones' derivatives (17 each) against the reference
   (1e-12); an episode of the environment with the lag against the reference
   engine with it (every state within 1e-9, rewards, end); the rate loop stays
   stable (a 2 rad/s roll-rate step overshoots 10 % with the lag, 5 % without,
   and both settle); a lag without time constants rejected. Checked by breaking
   the code (5 mistakes: the time constants swapped, the forces from the command
   in either engine, the command not clipped, the environment ignoring the
   switch); each made a test fail. **The phase 11 policy with the lag**:
   finishes in 3.276 s (3.235 s without), 8.7 cm from everything at the closest
   (13.0 cm), every one of 64 noisy laps finished.
3. ✅ **Ground effect** (loose end 7; `jinete/proximity.py`: `GroundEffect`,
   `ground_effect_factor`; `Multirotor(..., ground_effect=...)` and the batch;
   `MultirotorParams.propeller_radius`, the Iris's 0.128 m; `Realism(
   ground_effect)`). Changed from the plan with the author's knowledge: the
   ceiling effect is left out (loose end 44). Each rotor's thrust multiplied by
   Cheeseman and Bennett's factor 1 / (1 - (R / 4z)^2 / (1 + (V / v_i)^2)),
   z the height of its hub above the floor (not below R / 2: at most 4/3), V the
   airspeed of the hub along the disc, v_i the induced velocity of its thrust
   free of the floor; the reaction torques left as they are free of the floor.
   Tests: the factor by hand (16/15 at z = R, 64/63 at 2R, 256/255 at 4R, 4/3
   at and below R / 2, 32/31 at V = v_i, 1 without thrust, within 1.1e-5 of 1 at
   10 m); rotors lifting 90 % of the weight hold the Iris where their factor is
   1 / 0.9 (hubs 10.1 cm up; 1e-12), it sinks above and rises below; flying at
   6 m/s 10 cm up it feels less than standing still; rolled or pitched 10° at
   15 cm up, the low rotors' extra thrust levels it (no such moment free of the
   floor) and the yaw moment is unchanged; 300 drones' derivatives against the
   reference with and without the motor lag (1e-12; over 50 of them feel it);
   with the rotors stopped every force as without it, bit for bit; an episode
   with the ground effect and the lag against the reference engine (1e-9).
   Checked by breaking the code (7 mistakes: 2z for 4z, the reaction torque
   scaled in either engine, the hub height from the wrong column of R or from
   the centre of mass, the forward speed ignored, the height's sign); each made
   a test fail. **The phase 11 policy with it**: 3.26 s, 21.8 cm from everything
   at the closest; with the motor lag too, 3.33 s and 20.0 cm; every noisy lap
   finished.
4. **Sensors and estimator** (`atalaya`): motion capture (position and attitude
   with noise, at its rate, with its delay), gyroscope (noise and bias, values of
   PX4 SITL's Iris IMU), and a Kalman filter for position and velocity from the
   motion-capture positions (`atalaya.kalman`); for many drones at once. Tests:
   noise statistics in their intervals, delays exact, the filter's consistency
   (NEES), seeds repeat.
5. **The policy sees the estimate, with latency** (`CourseEnv`): the
   observation built from the estimated state, the action applied after the
   latency to the motors. Tests: with no noise and no delay, the environment of
   before bit for bit; delays exact; restarts clear the memory of sensors and
   filter.
6. **Parameters per drone and domain randomisation** (half of loose end 30):
   every drone of a batch with its own parameters (mass, inertia, thrust and
   moment constants, drag, motor lag, ...), equal to the reference engine with
   those parameters; ranges drawn at every restart (and the sensors' noise and
   the latency), the controller keeping the nominal values. Tests: each drone
   equal to the reference with its parameters; draws inside their ranges,
   repeated by the seed; no ranges, the nominal drone bit for bit.
7. **Under control after the last gate** (loose end 39): the episode goes on
   past the finish until the drone is stopped and level (or crashes). Tests to
   be planned with the step.
8. **Training with realism and measuring it**: a recipe, and the policy
   measured over many drawn conditions (success rate, lap time, closest
   approach) against the phase 11 policy and the baseline in the same
   conditions; a short training here, the long one on the author's computer.
9. **Close the phase**: this document and the README.

## 6. Decision log

Phase 12 (realism for the real drone):
- **Training is split over processes, not threads**: Python runs one line at a
  time per process (its global interpreter lock), so the parts of the
  environment run in separate processes, each with its own copy of the code,
  and send their arrays through pipes. The learning stays in one process, so
  PPO did not change. Processes are started with "spawn" on every system,
  because that is the only way on Windows and the tests should see what the
  author runs.
- **A training gives the same result with any number of workers**: a drone
  flies the same alone or in any batch, and the parts are joined in a fixed
  order, so the network sees the same batch; a test checks it. Each part has
  its own seed (`worker_seed`), so random courses differ between parts; one
  worker is the bare environment, reset with the seed as before.
- **More workers pay through larger batches**: splitting a fixed number of
  drones makes each batch smaller, and NumPy's cost per call is paid per batch
  (measured in phase 9: the batch engine gains up to ~1 000 drones). With 8
  cores a training can fly more drones at once for the same time per step.
- **The rotor speeds are states, not a filter in the controller's step**
  (step 2): PX4's Gazebo model filters the commanded speed once per physics
  step; here the lag is an equation integrated by RK4 with the rest, so the
  thrust changes inside a step as the rotors speed up. At 100 Hz with tau_up =
  12.5 ms (dt / tau = 0.8) RK4 follows the exponential within 0.3 % of a step
  of the command (measured; exact for its own discrete solution). The state
  grows only with the lag (13 + rotors), so nothing else changed: every module
  reads the body's elements by position and passes the rest on.
- **The controllers do not know the lag** (step 2): the rate loop still asks
  for rotor speeds as if they were reached at once, as PX4's does. With the lag
  a roll-rate step overshoots 10 % instead of 5 % (measured), still stable.
- **The ground effect is Cheeseman and Bennett's, rotor by rotor** (step 3): a
  formula with no fitted number (the rotor's radius, its hub's height, its
  airspeed and its thrust), applied to each rotor at its own hub, so a tilted
  drone is levelled near the floor. Multirotor corrections (the neighbouring
  rotors' and the body's images, Sanchez-Cuevas, Heredia and Ollero 2017) add
  to it, but their coefficients could not be read from an open copy in this
  session; left for when the real drone's thrust is measured near the floor
  (phase 14). For the Iris standing on the floor (hubs 7.7 cm up, z/R = 0.6)
  the factor is 1.21: the take-off is the moment it matters.
- **The ceiling effect is left out** (step 3, changed from the plan): the open
  models found are fitted to small propellers flown millimetres from the
  ceiling (Hsiao and Chirarattananon 2019: 23 and 50 mm propellers), and the
  usual multirotor model (Sanchez-Cuevas et al. 2017) could not be read from an
  open copy. With the training's 5 cm margin the Iris's hubs stay at least
  0.18 m (1.4 R) below the ceiling, where the effect is small, and the courses
  do not fly close to it. Loose end 44.

Phase 11 (the Iris learns the course):
- **Gymnasium 1.4.0** (the newest, for Python 3.10-3.14) is an optional
  dependency: only `picadero/gymnasium_adapter.py` and its tests need it (they
  are skipped without it). Its checker found that the lap time, nan until a lap
  finishes, made two equal runs look different (nan != nan): the lap time is now
  in the info only once there is one. Gymnasium's source was read and the
  adapter tested against it in Claude's workspace, where it cannot be installed.
- **The network's products use NumPy's matrix product (BLAS)**: a row gives the
  same numbers alone or in a batch to ~1e-14, not bit for bit (measured: BLAS
  adds the terms in another order for other sizes, and for other processors).
  Writing the products out in a fixed order, as the physics does, would make
  training many times slower and the policy no more right. So a policy flown on
  another computer gives actions equal to rounding, and a lap's regression test
  allows for it.
- **PPO's choices are the usual ones** (Stable-Baselines3 and CleanRL): a
  Gaussian with a spread learned apart from the observation, actions clipped by
  the environment (not squashed by tanh), separate networks and optimisers for
  the policy and the critic, advantages normalised per minibatch, gradients
  clipped to 0.5, Adam's epsilon 1e-5, the learning rate falling to zero, no
  clipping of the critic. The normaliser is updated once per rollout (frozen
  while it is collected, so a rollout's observations are all scaled alike).
- **The reward and the training recipe were found by measurement** (Claude's
  workspace, the first course, 256 drones, PPO's defaults, two runs at a time
  on two cores, ~6 000 steps/s each). A: the reward of phase 10 (progress 1,
  gate 1, crash 5): first laps finished at ~5.7 M steps; best mean-action lap
  6.43 s (20 M), but 1.3 cm from a frame and 87.5 % of its noisy laps
  finished. (The prototype before the phase, with a constant learning rate and
  no gradient clipping, had settled on hovering; A did not.) B: a time penalty
  of 1 per second: stuck after the first gate for 7 M steps (stopped). C: 0.25
  per second: its mean action finished a lap from ~2.6 M steps (its noisy laps
  from ~3.8 M), best 4.39 s at 20 M (32 minutes), every noisy
  lap finished, but it passed 0.8 mm from something: it flies on the edge of
  the collision model, where the baseline was required to keep 5 cm. D: 0.25
  per second with the 5 cm margin from the start: crashes were so frequent that
  by 6 M steps every episode ended by the time limit (the drone waited:
  discounted at 0.99, waiting costs at most ~0.5, a crash 5); stopped. E: C's
  best agent trained 10 M more steps with the margin (learning rate 1e-4): the
  policy kept (4.325 s, 15.8 cm; 15 minutes). Two stages (a curriculum) is the
  recipe of `examples.train_course`; run as written it repeats C and E on Claude's
  workspace, and gives a different policy of the same kind on another computer
  (the products round differently and a training amplifies it).
- **The time penalty is small on purpose**: rewarding speed only through the
  discount (A) gives a slower lap; a penalty larger than about the crash
  penalty times (1 - gamma) per step makes an early crash look cheap (B).
- **The safety margin is the baseline's robustness test made a rule of
  training**: the baseline's laps count only if they keep 5 cm from
  everything; the policy's are trained to keep it (the hitbox spheres 5 cm
  larger), and its best agent is chosen among laps that keep it. Judged on the
  course as it is (no margin).
- **How the policy beats the baseline** (its lap, measured): it takes off tilted
  54° forward and reaches 8.3 m/s (the baseline 4.6), banks up to 64° in the
  turns (the baseline's velocity loop limits the tilt to 45°, PX4's
  MPC_TILTMAX_AIR; the policy commands rates, within the same thrust and rate
  limits, and nothing limits its tilt), overshoots gate 3 and comes back
  through it (the referee counts only forward crossings of the opening, so
  this is allowed), and rolls past 90° (137°) to reach the last gate at 1.3 m
  up: the lap ends there, so what comes after does not count (loose end 39).
  The comparison is fair in the commands and in the 5 cm, not in the attitude:
  the baseline keeps PX4's tilt limit.
- **The surroundings are the nearest surfaces' closest points** (step 6): the
  room's faces analytically, each obstacle and frame by p - d grad(d) with the
  gradient of its signed distance by central differences (1e-5 m), the four
  nearest in body axes, nearest first. A fixed number of numbers whatever the
  course (unlike a list of obstacles), and the same kind of information a range
  sensor gives. Rejected: rays cast in fixed directions (a depth image), many
  signed-distance evaluations per ray, left for when sensors come (phase 12).
  Cost (Claude's workspace, 256 drones, random actions): the environment at
  ~0.75 of its speed without them.
- **Random courses are drawn, checked and redrawn** (step 6): every course must
  be flyable along the baseline's path (it keeps 1 m from every obstacle, 0.6 m
  from the walls and the ceiling, and passes no frame but its gate's), so a
  course the policy fails is the policy's failure, and the baseline can be
  measured on every test course (it finished all 20). Validation (choosing the
  best agent, seed 1000) and test (measuring it, seed 2000) courses are never
  trained on.
- **One course per batch, several batches** (`CourseSetEnv`; loose end 30): the
  batch engine flies one course per batch, so k courses are k environments
  stepped in turn. Small batches pay NumPy's fixed cost per call k times: in
  Claude's workspace 2 courses of 128 drones trained at ~5 000 steps/s and 8 of
  32 at ~3 000 (one course of 256: ~12 000). A course per drone in the batch
  engine would remove that cost (loose end 30 stays open).
- **Few courses per update make training swing** (measured, step 6): with 2
  courses at a time the return jumped between -2 and +7 from one update to the
  next (each update learns from the 2 courses in flight); the recipe of
  `examples.train_random` flies 8 courses of 128 drones (1 024 drones, meant for
  the author's computer).
- **A warm start from the first course's policy did not help** (step 6, measured):
  its first layer widened with zero weights for the 12 new inputs (so it starts
  flying as before and learns to use them) crossed 3 validation gates after
  5.2 M steps, against 5 for a policy from scratch at the same point; its spread
  had already shrunk to 0.21 (from scratch: 0.42), and new courses need
  exploration. Not kept.
- **The trained policy is kept in the repository** (`policies/first_course.npz`,
  120 kB: the normaliser, the policy and the critic), so its lap is a
  regression test and `examples.policy_course` draws it without training.
- **The same recipe, two computers, two policies** (after the phase): run as
  written on the author's computer (`python -m examples.train_course`, 72.7
  minutes), the recipe gave a lap of 3.235 s where Claude's workspace had given
  4.325 s. Nothing differs but the rounding of the network's products, which a
  training amplifies (loose end 42): one run is one sample, and 4.3 s was not
  the recipe's limit. The author's policy is the repository's now: faster,
  every noisy lap finished (100 % against 97 %), and it does not flip at the
  finish (it crosses the last gate tilted 18° at 1.19 m, 7.6 m/s, still turning
  at ~3.6 rad/s in roll and pitch; the earlier one crossed it at 144°). Its
  flight: top speed 8.6 m/s, tilt at most 84.6° (more than 60° for 0.95 s), no
  detour at gate 3, every turn cut inside the baseline's path; checked in
  Claude's workspace (the same lap, 3.2346 s, in the batch and the reference
  engines). Its best agent came from the first stage: the second stage's best
  was its first evaluation, after 163 840 steps (weights changed by at most
  0.013), and no later agent beat it. That also showed that an agent's
  `info["steps"]` counted only its own stage; it now counts the steps the agent
  brought with it too (17 694 720 for this policy).
- **PPO is checked against a problem with a known optimum**: the discounted LQR
  of a double integrator, whose gain the Riccati equation gives exactly. With
  episodes cut after 8 steps, the right gain is learned only if truncations are
  completed with the critic's value; treating them as ends gives a gain of about
  a third (measured: [0.64, 0.44] against [1.72, 1.74]).

Phase 10 (training environment and classical baseline):
- **The environment has Gymnasium's shape without depending on it** (decided
  with the author): the library cannot be installed in Claude's workspace (no
  network), so code using it could not be run before handing it over; and the
  vectorised interfaces of Gymnasium (`VectorEnv`) and Stable-Baselines3
  (`VecEnv`) differ, so choosing one now would choose phase 11's library early.
  An adapter of a few dozen lines comes at the start of phase 11 and is checked
  there with Gymnasium's environment checker.
- **The policy acts at 50 Hz** (decided with the author; loose end 8): each action
  is held for two physics steps; the rate loop and the physics stay at 100 Hz.
  About 440 decisions per lap instead of 880 (shorter horizons are easier to
  learn), a usual rate for setpoints from a companion computer to PX4.
- **The environment is batch-native and has no reference twin**: what it adds
  (observation, reward, restarts) is tested by hand and its flight is checked
  against the reference engine (`Multirotor`, `RatePilot`, the course's referee)
  drone by drone, as every batch module is.
- **The landed state of the rate pilot is PX4's**: no integration while landed
  (reset before and after each update), take-off when the commanded thrust
  exceeds the weight, no landing detection (in a lap a touch-down is a crash).
  The motors follow the commands on the ground (the autopilot stops them), so a
  policy can spin up before it lifts off.
- **A vehicle leaves the floor only above 1 cm** (`TAKEOFF_CLEARANCE`, found in
  step 2): at 90 % of the weight on the floor, the gear (sinking 1 mm under the
  full weight) unloads and its feet hop a fraction of a millimetre; the monitor
  took that for a take-off and the feet coming back for a floor crash, so a
  policy hesitating on the floor would have crashed without moving. The
  autopilot never showed it (motors stopped while landed). Every earlier test
  passed unchanged.
- **The action envelope is the autopilot's** (thrust 10-90 % of the maximum,
  rates 3.5 / 3.5 / 1.5 rad/s, PX4's MC_ROLLRATE_MAX 220 deg/s): with the same
  limits, the comparison with the baseline measures the pilot. PX4's acro mode
  defaults to 100 deg/s and its documentation suggests 720 deg/s for experienced
  pilots; widening the envelope is a phase 11 decision (loose end 36).
- **The observation is in body axes**, so it does not depend on where the course
  lies or which way it faces, and describes each gate by four points on the edge
  of its opening (place, size and heading at once; a circle by its inscribed
  square, inside the opening). The attitude is the whole rotation matrix (no
  discontinuity, unlike Euler angles or a quaternion's sign). The previous action
  is included, so a policy can be smooth. Units stay SI: scaling for learning is
  the training's (phase 11). Obstacles are not observed (loose end 33).
- **Progress is measured against the gate targeted at the start of each physics
  step**, also in the step that crosses it: the target changing mid-step would
  pay a jump of minus the distance between gates for crossing one. Summed over a
  lap it pays about the course's length whatever the lap time; speed is rewarded
  through the discount of later rewards, as in Swift. The weights (1 per metre, 1
  per gate, 5 per crash) are a starting point.
- **A crash terminates, the time limit truncates**: after a crash or the finish
  nothing more comes; after the time limit the drone could have flown on, and the
  learning algorithm must not take what would have come as zero.
- **Same-step restarts**: a drone whose episode ends is put back on the floor in
  the same step, its last observation in `info` (as Gymnasium's same-step
  autoreset and Stable-Baselines3's `VecEnv`), so a batch never waits for its
  slowest episode.
- **Observations float32, rewards and physics float64**: float32 is what learning
  libraries use; the observation is converted at the interface only.
- **The baseline is a pure-pursuit path follower over the velocity autopilot**
  (the plan's "through the gate centres with the existing velocity autopilot",
  planned as loose ends 10 and 28 asked): a polyline through entry, centre and
  exit points along each gate's heading; aiming at a point `lookahead` metres
  further on the path corrects drift and cuts corners smoothly; the speed is
  planned from a lateral acceleration in the corners and a braking deceleration.
  It is a reasonable classical reference, not the classical optimum: a minimum-
  time trajectory through the gates, tracked with feedforward, would be faster.
- **The baseline's settings are swept in the batch and must be robust**: the
  fastest lap of a plain sweep (6.48 s) cleared a frame by 6 mm and crashed in 7
  of 33 copies with its settings changed by 2 %: luck, not a time a controller
  can be trusted to fly. A setting counts only if four copies changed by up to
  2 % (seeded) also finish and none comes within 5 cm of anything. The robust
  best (6.773 s) finished 65 of 65 copies at 2 % and 64 of 65 at 5 %.
- **The baseline is also flown through the environment's actions** (its
  velocity and attitude loops at 50 Hz turned into thrust and rates): it
  finishes in 6.76 s, so the action envelope and the 50 Hz rate allow a winning
  lap, and the lap checks the environment's reward over real gate crossings.
- **Loose end 29 is done first** (decided with the author): the overhang formula
  written out gives the same numbers bit for bit and the batch ~45 % more speed,
  which phase 11's training will use.
- **Speed on the author's computer after the phase** (2026-11-11, Windows,
  Python 3.13, one core; `python -m examples.batch_course` and `python -m
  examples.env_random`). Steps per second for all the drones together, times real
  time in brackets; the batch engine flies the gate guidance with the whole
  autopilot, the environment random actions through the rate pilot (each of its
  steps is two physics steps, so 0.02 s of flight):

  | drones | batch engine (phase 9 → 10) | environment (steps/s) |
  |---|---|---|
  | 1 | 241 → 272 (2.7×) | 166 (3.3×) |
  | 16 | 3 665 → 3 778 (38×) | 2 373 (47×) |
  | 64 | 11 813 → 14 290 (143×) | 7 992 (160×) |
  | 256 | 28 048 → 36 548 (366×) | 20 452 (409×) |
  | 1024 | 43 764 → 58 093 (581×) | 32 475 (650×) |

  Loose end 29 gave +33 % at 1 024 drones there (+~45 % in Claude's workspace:
  the collision checks are a larger share of the step on its machine). The
  environment's physics steps (2 × 32 475 = 64 950 drone-steps/s) are a little
  cheaper than the batch engine's, since the rate pilot skips the velocity and
  attitude loops. ~650 times real time is close to Swift's 700. The laps flown
  by `batch_course` are unchanged (8.81 s at 3 m/s; crashes at 3.5 and 4 m/s at
  the same moments).
- **Every step's tests were checked by breaking the code** (32 mistakes over the
  phase); seven survived a first version of the tests and each got its own test:
  the rate pilot's resets before and after the update and its threshold at
  exactly the weight; a crash penalty not masked for ended drones; the pilot not
  set back to landed on a restart; the target gate read after the step (no
  scripted episode crossed a gate; the baseline's lap through the environment
  does); progress allowed to go back. Equivalent mistakes (the same results by
  construction) are noted in each step.

Phase 9 (fast batch engine):
- **Loose end 26 closed by conservative advancement** (step 6). Of the two ways
  in the plan, sampling faster only shrinks the gap (a graze missed is ~ s²/8ρ:
  four times smaller at twice the rate, for twice the cost), while checking the
  move between samples closes it for straight moves at almost no cost. Each sphere
  moves in a straight line from its centre at one sample to its centre at the
  next; a signed distance never changes faster than the point moves, so a sphere
  with gap m can move m without touching, and the move is advanced by the smallest
  m_i / |move_i| over the spheres until a sphere comes within 1 µm (a contact) or
  the next sample is reached. Skipped when the margins at both samples add up to
  more than the longest move (then nothing can touch in between): in the lap of
  the first course the advancement never runs, and the lap costs the same.
- **Contacts with walls, obstacles and gates are timed by the advancement too**,
  not only grazes: when a shape is touched at a sample, the advancement finds where
  a sphere comes within 1 µm and the last bit is linear interpolation of the gap.
  Exact on a flat face approached in a straight line (every earlier exact-moment
  test still passes at 1e-9); on a pillar passed at a slant the old interpolation
  between samples was up to 3.3 ms late at 20 m/s (measured), now within 1.2e-5 s.
  Advancing to the surface exactly (tolerance 0) was tried and rejected: a 45°
  approach then stops a little short (5.5e-7 s early on the existing test), and a
  pass that only brushes a shape never ends. The floor keeps its interpolation:
  for a plane and straight moves it is exact, and nothing can touch it between
  samples.
- **1 µm (`SWEEP_TOLERANCE`) counts as touching**: below that the advancement's
  steps (gap / move) shrink without end on a shape the sphere only brushes. Every
  step advances at least 1 µm / longest move, so it always ends. Physically 1 µm is
  nothing next to the hitbox's own approximation (centimetres).
- **The move is the chord**, not the arc: a sphere off the centre of mass follows
  an arc as the vehicle turns, ~ r (ω dt)²/8 off the chord (0.04 mm for a rotor at
  3.5 rad/s). A test turns a sphere 90° in one step to show it.
- **The step 6 tests were checked by breaking the code** (10 mistakes: never
  sweeping, a prefilter twice too strict, steps twice too long, no last-bit
  interpolation, gaps without the radii, sweeps ignored, a zero move, a tolerance
  of 1 cm, a tolerance of 0). The tolerance of 0 survived the first version (it
  only changes when brushing contacts end); a test of a pass within the tolerance
  was added.
- **The batch monitor checks only the drones it is told to** (`drones`), and the
  batch referee tells it the ones still flying: a drone whose lap has ended is not
  checked, so a finished drone that drifts into a wall stays finished, as in the
  reference, where `run` stops at the end of the lap.
- **The advancement in the batch runs in rounds**: each round asks the shape about
  every sphere of every drone still advancing, and a drone drops out when it
  touches or reaches its next sample; only the drones near a shape take part. The
  rounds end with the slowest drone, so a rare near miss of a few micrometres (a
  few thousand rounds, step 6) holds up its batch for that step.
- **Statuses are codes** (`FLYING` .. `TIMEOUT`, words in `STATUS_NAMES`), a
  collision is arrays (`collided`, `check`, `time`, `position`) and the passes are
  (n, gates) arrays; `collision(i)` and `passes(i)` give one drone's in the
  reference's types, which is how the tests compare them.
- **The step 7 tests were checked by breaking the code** (17 mistakes). Five
  survived the first version of the tests, each a real gap: the floor counted
  before take-off (every drone that started on the floor took off at once), a
  rectangular opening judged by its width only and a circular one 10 cm too wide
  (no move through a gate plane outside its opening), a gate counted although the
  crash came first in the step, and the monitor not reset with the referee (no
  crashed lap was reset). Tests were added for each (drones standing still, random
  moves through both gate types, events in the same step, a crashed drone reset).
- **Cost of the batch referee** (Claude's workspace, the lap of the first course
  replayed, 13 spheres, 3 obstacles, 4 gates): ~1 000 drone-checks/s at N = 1,
  ~28 000 at 64, ~45 000 at 1024, about half the speed of the dynamics with RK4
  (step 3): the collision check is a large share of a batch step.
- **Speed of the batch engine, measured** (step 8, one core; the Iris flying the
  first course with the gate guidance, the whole autopilot, the floor contact and
  the referee with 13 spheres, 3 obstacles and 4 gates; drones that end a lap
  start the next at once; `python -m examples.batch_course`). Drone-steps per
  second, and times real time in brackets; the author's computer measured after
  the phase was closed (2026-11-11, Windows, Python 3.13):

  | drones | Claude's workspace (2026-11-04) | author's computer |
  |---|---|---|
  | reference engine | ~700 (~7×) | 1 031 (10.3×) |
  | 1 | ~200 (~2×) | 241 (2.4×) |
  | 16 | ~2 800 (~28×) | 3 665 (37×) |
  | 64 | ~7 900 (~79×) | 11 813 (118×) |
  | 256 | ~19 000 (~190×) | 28 048 (280×) |
  | 1024 | ~28 000 (~280×) | 43 764 (438×) |
  | 4096 | ~30 000 (~300×) | not measured |

  On the author's computer the batch at 1 024 drones is 42 times the reference
  (43 764 against 1 031), as in Claude's workspace (~40 times). Absolute numbers
  from Claude's workspace are not comparable from one session to the next: on
  2026-11-11, in another session, the same code gave 1 640-1 807 steps/s for the
  reference and 53 839 drone-steps/s at 1 024 drones, about twice the numbers of
  2026-11-04 (its machine changes between sessions). Ratios measured in one session
  (batch against reference, before against after a change) hold; absolute speeds
  are the author's computer's.

  About 40 times the reference from ~1 000 drones on; the gain flattens there, when
  every array operation is large enough that NumPy's fixed cost per call no longer
  matters. Below ~4 drones the batch is slower than the reference (every line is
  an array operation). Swift trained at over 700 times real time; at 1 024 drones
  the author's computer gives ~440 times. Where the time goes at 1024
  drones: the collision checks ~60 % (most of it NumPy's reductions over the last
  axis of small arrays, `np.linalg.norm(..., axis=-1)` and `np.max(..., axis=-1)`
  in the shapes' distances), the dynamics with RK4 ~35 %.
- **Speeds are judged on the author's computer** (after the phase): Claude's
  workspace runs on a different machine from one session to the next (the same
  code was about twice as fast there on 2026-11-11 as on 2026-11-04), so only
  ratios measured within one session of it count: batch against reference, before
  against after a change. Absolute speeds and comparisons with Swift's 700× come
  from `python -m examples.benchmark` and `python -m examples.batch_course` run by
  the author.
- **A further 40 % is measured but not taken** (step 8): writing the distance from
  overhangs (`campo.indoor._distance_from_overhangs`) without reductions, the two
  or three components added by hand, gives the same numbers bit for bit (checked on
  100 000 random points) and takes the batch at 1024 drones from ~28 000 to
  ~39 000 drone-steps/s. It changes the reference's geometry module, which the
  plan did not include: it is the first item of the next speed work (loose end 29).
- **The batch loop advances only the drones still flying**: every drone is
  integrated (the arrays keep their shape), and the ended ones get their old state
  and clock back. `step` returns the laps that ended in that step, so a caller can
  restart exactly those; `reset` starts a drone's next lap (state, clock, referee),
  and its controller's memory is the caller's to reset (`GateGuidance.reset` puts
  the autopilot back to landed).
- **The step 8 tests were checked by breaking the code** (7 mistakes). The
  referee not reset with the lap made two tests loop for ever (the drone never
  starts again): the loops in the tests and in `fly_laps` are bounded now, so such
  a mistake fails instead of hanging. One mistake is equivalent with this guidance:
  a drone reset without `landed` takes off the same way, because the guidance asks
  it to climb from the first step (a landed drone asked to climb resets its loops
  and takes off; a flying one with its loops just reset flies the same climb).
- **The whole cascade goes into the batch engine** (decided with the author). It
  adds no realism (the physics is the same), but it pays twice: the classical
  baseline of phase 10 can be flown on many courses at once (phase 11 compares the
  policy with it on randomised courses), and a whole closed-loop lap is the
  strongest check that the two engines agree.
- **One course per batch**: every drone flies the same room, obstacles and gates;
  their arrays are shaped so that a course per drone can be added in phase 11.
- **Loose end 26 is closed in this phase**, in the reference first (the reference
  is the truth) and then in the batch, so the check is written once against a
  validated model.
- **Before the batch engine, the per-call cost of NumPy was measured**: in Claude's
  workspace `np.cross` on two 3-vectors takes ~27 µs against ~2.7 µs for the
  formula written out, and it was over half of a lap (34 438 calls). NumPy's
  general functions check shapes and move axes before any arithmetic; on a
  3-vector that set-up is the whole cost. `core/vector.cross` writes the formula
  out (the same operations in the same order, so the same rounding): the lap of
  the first course gives the same states bit for bit and runs 1.7 times faster
  (~390 → ~650 steps/s there; 1 031 on the author's computer afterwards). The
  same fixed cost per call is what the batch engine spreads over N drones.
- **The benchmark keeps the fastest of several runs**: the slower ones were slowed
  down by the rest of the computer, not by the simulator.
- **The batch engine agrees with the reference to rounding, not bit for bit**
  (measured in step 2): in the reference, w, x, y, z are NumPy scalars, and a
  scalar's `x**2` goes through C's `pow`, which is sometimes one unit in the last
  place away from `x * x` (on an array `x**2` is `x * x`); `@` hands 3 x 3 products
  to a linear algebra library that may add the terms in another order. Making the
  batch copy those quirks would make it slower and no more right. Inside the batch
  engine bit-for-bit equality does hold: no operation mixes the rows (`matvec`, the
  terms added in a fixed order, instead of `@`), so a drone gives the same numbers
  alone or among any number of others.
- **One vehicle profile per batch**: every drone of a batch has the same mass,
  inertia and constants, like the same course. Parameters per drone come with the
  randomisation of phase 12.
- **Time can be one per drone**: the batch passes `t` to the force model as it
  comes, a float or an (N,) array, because drones reset at different moments
  (step 8) each run on their own lap clock.
- **The batch formulas are written term by term like the reference ones**, so a
  difference between them is a bug, not a choice. The tests of step 2 were checked
  by breaking the code (8 mistakes: a sign in the matrix and in the quaternion
  derivative, the norm squared, R^T instead of R, I instead of I^-1, gravity on
  the wrong axis, the renormalisation in place, the cross product reversed); each
  made a test fail. One lesson from that check: a mistake of the same length as
  the original, restored within the same second, left a stale `__pycache__` that
  Python kept using (it checks the file's time in seconds and its size); delete
  `__pycache__` after breaking code on purpose.
- **Sums over rotors and gear points are loops over them, each line acting on all
  the drones** (four rotors, four feet for the Iris): the terms are added one after
  the other as in the reference, and no operation mixes the drones. A sum along an
  axis of a (N, 4, 3) array would also work, but how NumPy orders such a sum is its
  own business.
- **The batch contact computes every foot of every drone**; a foot that does not
  touch gets exactly zero force (the reference skips it). Adding zero changes
  nothing, so the results are the same, and every drone follows the same
  operations whether it flies or stands.
- **Environment models are the reference ones, called with arrays**: the density
  gets the altitudes (N,), the wind the positions (N, 3), and each may return one
  value for all or one per drone. `sea_level_density`, `no_wind` and
  `ConstantWind` work as they are; the ISA functions do not (loose end 2), and
  indoors they are not needed.
- **The batch multirotor knows its number of drones** (`BatchMultirotor(params,
  n, ...)`), so its commands start as zeros (n, rotors), motors off, as the
  reference starts at zero.
- **The lap replay records the reference commands by wrapping `run`** in the test
  (monkeypatch of `examples.iris_course.run`), so the example does not change for
  the test's sake.
- **The step 3 tests were checked by breaking the code** (11 mistakes: the floor
  pulling, the foot velocity not rotated, the contact moment from the NED force,
  feet counted above the floor, rotor commands not clipped, a rotor left out of the
  mixer, the rotor drag out of plane, airspeed rotated the wrong way, no rotor drag
  moment, the wind's sign, the density ignored); each made a test fail.
- **Controllers in the batch keep one memory per drone** (integrals (n, 3)), and
  `reset(drones)` takes the drones to reset: all, a boolean mask or indices. In
  training each drone starts a new lap when its own ends, so a reset of all of
  them at once is not enough.
- **The batch loops take the reference gains** (`RateGains`, ...): the gains are
  profile data, one set per batch like the rest of the profile.
- **The step 4 tests were checked by breaking the code** (7 mistakes: a column of
  the mixer left out, no thrust clip, a looser anti-windup, the gyroscopic term
  with the wrong sign, a reset of every drone, no integral, the moment without the
  inertia); each made a test fail. A first version of one test drew moments of ~2 N·m,
  far beyond the Iris's yaw authority (at most ~0.85 N·m: two rotors at full
  thrust times the moment constant, 0.06 m), so 496 of 500 wrenches were clipped
  and the free case was hardly tested; with ~0.3 N·m, 320 are clipped, and the test
  now checks that both cases occur.
- **An `if` per drone becomes a mask** (step 5): the reference autopilot returns
  early for a landed vehicle; the batch runs every loop for every drone, then puts
  the integrators of the drones that stay on the ground back to zero and stops
  their motors, which leaves them where the early return leaves the reference.
  Branching formulas (Shepperd's method, the small-angle rotation vector, the
  horizontal thrust limit) compute every branch for every drone and keep the one
  each condition picks with `np.where`; the branches thrown away may divide by zero,
  so they run inside `np.errstate` with those warnings off.
- **The first course is flown in the batch with the reference's gate order**: the
  referee comes to the batch in step 7, so the test records which gate the
  reference referee gave the guidance at every step. The lap is closed loop
  (guidance, cascade, landed start, contact), so it checks the whole autopilot, not
  only the dynamics.
- **The step 5 tests were checked by breaking the code** (18 mistakes). Six
  survived the first version of the tests, and four of them were real gaps: a
  Shepperd branch chosen on half its condition (exact half turns never exercise it,
  and half turns tilted 1e-8 give diagonal terms equal to the last bit; tilted 1e-6
  they differ by ~2e-12 and only the right formula is accurate); the take-off
  threshold (no command asked to climb slower than 0.2 m/s); the reset after the
  loops of the drones that stay landed (only visible in their integrators after the
  call); and the reset before them (only visible in a drone declared landed in the
  air with charged integrators). A test was added for each. Two mutants are
  equivalent, not gaps: below sin(angle/2) = 1e-12 both formulas of the rotation
  vector agree to rounding, and "a flying drone lands" needs no `flying` in the
  mask because the drones on the ground are landed already.
- **First speed of the batch dynamics** (step 3, Claude's workspace; Iris forces,
  floor contact, RK4 and renormalisation, no control or collisions): ~300
  drone-steps/s at N = 1 (slower than the reference: every line is an array
  operation), ~4 600 at 16, ~40 000 at 256, ~98 000 at 4096. The real measure, with
  everything, is step 8.
- **A prototype of the batch dynamics was measured before planning** (Claude's
  workspace, dynamics + RK4 only): per derivative within 5e-15 of the reference;
  the 882 rotor commands of the reference lap replayed in it within 1.2e-13 m;
  ~740 drone-steps/s at N = 1, ~8 500 at 16, ~38 000 at 256 and ~50 000-57 000 at
  4096 (the gain flattens once each array operation is large enough).

Phase 8 (indoor world):
- **The demonstration flight is the simplest guidance**: climb 0.5 m, then fly at
  constant speed straight at the next gate's centre, nose along its heading. It
  finishes the first course at 3 m/s (8.81 s) but crashes into the frame of gate 2
  at 4 m/s (measured): it turns only once a gate is crossed, so at speed it
  overshoots. The classical baseline of phase 10 needs to plan through the gates.
- **The example's function is tested** (`tests/test_course_flight.py` imports
  `examples.iris_course.fly_course`): the whole of phase 8 working together, with a
  regression lap time. ~1 000 steps/s with the referee (13 spheres, 3 obstacles,
  4 gates).
- **Course files are YAML read strictly** (`yaml.safe_load`, then every key checked
  by hand): a typo must not pass silently, and every message names its place.
  `bool` is refused where a number is expected (in Python `True` is an int). The
  constructors' own checks (sizes > 0) are reported at the same place.
- **NED in the files too**, as in the code: one convention everywhere. Its usual
  mistake (a positive z, below the floor) is caught by the layout check, whose
  message reminds that z points down.
- **The layout is checked when a course is read**: a gate's whole frame inside the
  room (its centre at least the frame's reach from every surface: simple and a
  little cautious, independent of the heading), each obstacle's centre (box) or
  base (cylinder) inside the room, the start inside the room and not inside an
  obstacle.
- **The start stands on the floor**: `start_state(rest_height)` with
  `resting_height(gear, mass)`, the height at which the gear carries the weight
  (0.054 m for the Iris: its feet 0.055 m below the centre of mass, sinking 1 mm).
- **The first course** (`courses/first_course.yaml`): an 18 x 10 x 4 m hall with
  four gates (two rectangular 1.6 x 1.4 m, two round 0.8 m radius, two of them
  turned 30° and -40°), a floor-to-ceiling pillar and two crates; time limit 30 s.
- **The referee checks only the next gate, forward**: a gate crossed backwards or
  out of order is neither counted nor a crash, so a drone that misses a gate can
  turn back (a course with gates side by side cannot be cut short). At most one
  gate per step (0.2 m at 20 m/s; gates are metres apart).
- **The earlier of two events in one step wins**, by their interpolated moments:
  crossing the last gate 2 ms before touching a wall is a finish. The lap time is
  the moment of the last crossing, from t = 0 with the drone on the floor (take-off
  included, as in a race from a standing start).
- **The referee calls the monitor itself** and owns the end of the lap (`status`,
  `end_time`, `lap_time`, `passes`); `reset()` resets both. Checked by breaking it
  (6 mistakes); "reset keeps the monitor" passed at first, because the reset test
  flew a lap without a crash: it now repeats a crashed lap.
- **A gate is crossed by the centre of mass**, through the opening and forward: the
  spheres touching the frame are a crash anyway, so "the centre went through the
  opening and nothing touched" is "the drone went through". Backward crossings are
  reported (`forward=False`); the referee decides what they mean. A start exactly
  on the plane counts as after it, so touching the plane without going through
  crosses nothing, and one move is never counted twice.
- **The opening is judged where the plane is cut**, interpolated linearly between
  the two samples (exact for a straight move; the curvature of a 10 ms piece of
  flight moves it by ~a dt²/8, a quarter of a millimetre at 20 m/s²).
- **Frames by their signed distance**, so they are obstacles like any other (the
  monitor's `gates`, reported as kind "gate"). Rectangular: in the gate's plane the
  ring between the outer rectangle and the opening, max(d_outer, -d_opening) (exact
  for a ring of rectangles), extruded to the bar thickness with the box formula;
  tested equal to the union of its four bars (outside a union the distance is the
  smallest). Circular: a torus, hypot(sqrt(u² + v²) - R, n) - t/2.
- **Gates stand upright and turn only about the vertical**, like obstacles; a
  tilted gate is not needed for a first course.
- **The gate tests were checked by breaking the code** (8 mistakes). "The opening
  checked at the end sample" passed at first: every crossing tested went straight
  or ended inside the opening. A diagonal move into the opening and out to the side
  (and the reverse) now tells them apart.
- **The drone collides as a set of spheres fixed to its body** (`Hitbox`): a sphere
  touches a shape when the signed distance of its centre is below its radius, so any
  shape with a signed distance is checked the same way, for one drone or many. The
  monitor takes a list of spheres from the start; step 4 gives the Iris one sphere
  around everything (the plan's "sphere first"), step 4b several.
- **The Iris bounding sphere is 0.3844 m**, the reach of its front rotor discs: hub
  0.2555 m out and 0.023 m up, propeller radius 0.128 m and thickness 0.005 m
  (`rotor_N_collision` of `iris.sdf.jinja`): sqrt((0.2555 + 0.128)² + (0.023 +
  0.0025)²). The exact reach of a disc, not |hub| + radius (the triangle
  inequality, a little more). The corners of its collision box (0.337 m) are
  inside. (Step 4 left the thickness out, 0.3842 m; step 4b added it.)
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
- **The Iris collides as 13 spheres** (step 4b): one per rotor at its hub, radius
  sqrt(0.128² + 0.0025²) (holds the propeller cylinder exactly), and nine for the
  body. The plan said "one for the body", but the file's collision box (0.47 x 0.47
  x 0.11 m) takes in the arms and the legs: one sphere around it would be 0.337 m,
  0.67 m high, hardly better than one around everything. Cut into cells, each held
  by the sphere through its corners, the box is covered exactly (every point is in
  a cell). 3 x 3 cells (spheres of 0.124 m) chosen: 2 x 2 gives 0.350 m of height,
  3 x 3 0.275 m, 4 x 4 0.256 m (only the bottom improves; the rotors set the top).
- **The 13 spheres are tighter where it matters, not everywhere**: up 0.151 m
  (against 0.384 m), forward 0.280 m, sideways 0.348 m; but a sphere holding a disc
  bulges above and below its plane, so along a front hub the 13 spheres reach
  0.3846 m, 0.2 mm past the bounding sphere. A test first assumed "never farther
  in any direction" and failed on exactly that.
- **Smaller spheres miss deeper grazes** (measured): a graze is missed up to the
  depth of a chord of the sphere that grazes, and a rotor sphere (0.128 m) is a
  third of the bounding sphere: past a 5 cm pillar at 20 m/s up to 3.1 cm (1.2 cm
  before), 7 mm at 10 m/s. Head on, a thin wall needs 58 m/s (the drone's length
  along the path, 0.56 m, + w). Accepted for now; it is loose end 26.
- **Cost of 13 spheres**: ~76 µs per check with two obstacles (57 µs with one
  sphere): every shape takes all the centres in one call.
- **The step 4b tests were checked by breaking the code**: box spheres without the
  height, rotor spheres without the thickness, the disc reach with c_z signed, the
  grid not centred, `extent` not normalised, `contains` asking for all spheres; each
  made tests fail.
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
  Tested against free fall. `rk4_step` works on any array shape (an (n, 13) batch).
- `jineta/core/vector.py` — `cross(a, b)` (3-vectors or stacks, bit for bit equal
  to `np.cross`, ~10 times faster on one pair) and `matvec(matrix, vector)` (3 x 3
  products on stacks, the terms added in a fixed order).
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
  - `RatePilot(vehicle, gains, dt, thrust_command, rate_command, landed=False)`:
    callable `pre_step` hook through the rate loop alone (collective thrust [N],
    body rates [rad/s]); while landed the integral is held at zero; takes off
    when the thrust exceeds the weight; `reset()`.
- `jineta/jinete/profiles.py` — `flu_to_frd(vector)`, `IRIS` (PX4 SITL Iris, sources
  in the comments), `IRIS_RATE_GAINS`, `IRIS_ATTITUDE_GAINS`, `IRIS_VELOCITY_GAINS`,
  `IRIS_GAINS`, `IRIS_LANDING_GEAR` (the corners of its PX4 collision box),
  `IRIS_HITBOX` (13 spheres) and `IRIS_BOUNDING_SPHERE`, `SMALL_FIXED_WING`
  (generic small fixed-wing UAV, Aerosonde-class open data),
  `SMALL_FIXED_WING_PROPULSION` (its motor and propeller),
  `SMALL_FIXED_WING_AUTOPILOT` (design choices of its autopilot). The profiles by
  name: `MULTIROTOR_PROFILES` (`"iris"` → `MultirotorProfile(params, gains, gear,
  hitbox)`) and `FIXED_WING_PROFILES` (`"small_fixed_wing"` →
  `FixedWingProfile(params, propulsion, autopilot)`).
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
  (callable force model), `box_bottom_corners(size)`, `resting_height(gear,
  mass)`.
- `jineta/campo/indoor.py` — `Box(center, size, yaw=0)` (`to_local`,
  `signed_distance`), `Cylinder(base, radius, height)` (`signed_distance`),
  `Room(size)` (`center`, `clearance`, `wall_clearance` (no floor), `contains`),
  `box_distance(local_points, half_size)`. Frozen, validated, arrays read-only; one
  point or many.
- `jineta/jinete/hitbox.py` — `Hitbox(centers, radii)` (frozen, validated,
  read-only; `extent(direction)`, `contains(points)`), `union(*hitboxes)`,
  `disc_reach(center, radius, thickness=0)`, `box_corners(size)`,
  `bounding_sphere(rotor_hubs, propeller_radius, body_points,
  propeller_thickness=0)`, `rotor_spheres(rotor_hubs, propeller_radius,
  propeller_thickness)`, `box_spheres(size, divisions)`. The Iris's: `IRIS_HITBOX`
  (13 spheres), `IRIS_BOUNDING_SPHERE`, `IRIS_PROPELLER_RADIUS`,
  `IRIS_PROPELLER_THICKNESS`, `IRIS_COLLISION_BOX`; `MultirotorProfile` has a
  `hitbox`.
- `jineta/campo/gates.py` — `RectangularGate(center, yaw, width, height,
  thickness)`, `CircularGate(center, yaw, radius, thickness)` (frozen, validated;
  `to_local`, `in_opening(u, v)`, `signed_distance` to the frame, `crossing(start,
  end)`), `Crossing(fraction, point, forward)`, `Gate` (either type).
- `jineta/campo/course.py` — `Course(name, room, start, start_yaw, gates,
  obstacles, time_limit)` (`start_state(rest_height)`, `monitor(...)`,
  `referee(...)` from a hitbox and gear), `load_course(path)`,
  `course_from_dict(data)`, `CourseError`. The format is in its docstring.
- `courses/first_course.yaml` — the first course.
- `jineta/campo/referee.py` — `GatePass(gate, time, point)`, `Referee(gates,
  monitor, time_limit)`: callable `stop`; `status` ("flying", "finished",
  "crashed", "timeout"), `passes`, `end_time`, `next_gate`, `lap_time`, `reset()`.
- `jineta/campo/collision.py` — `Collision(kind, index, time, position)`,
  `CollisionMonitor(room, obstacles, sphere_centers, sphere_radii, floor_points,
  gates=())`: `margins(state)` ([room, obstacles..., gates..., floor], negative =
  touching), callable `stop` (`collision`, `airborne`), `reset()`; walls,
  obstacles and gates also checked along the spheres' straight moves between
  samples (conservative advancement, `SWEEP_TOLERANCE` = 1 µm). `Obstacle`
  protocol (anything with `signed_distance`).
- `jineta/atalaya/kalman.py` — `predict`, `update` (returns the innovation and its
  covariance too), `nees`, `constant_velocity_transition`,
  `white_noise_acceleration`, `piecewise_white_acceleration`, `alpha_beta_gains`.
- `jineta/escuadron/` — the batch engine (conventions in section 3):
  - `quaternion.py` — `to_matrix`, `derivative`, `normalize`, `multiply`,
    `conjugate`, `to_rotation_vector`, `from_matrix` on stacks (..., 4).
  - `rigid_body.py` — `BatchRigidBody(mass, inertia, force_moment)` with
    `derivative(t, states)` (n, 13); `BatchForceMomentModel`, `zero_force_moment`,
    `normalize_attitudes`.
  - `multirotor.py` — `BatchMultirotor(params, n, density, wind, contact)`:
    `rotor_speed_command` (n, rotors), `derivative`, `force_moment`.
  - `contact.py` — `contact_forces(gear, states, floor_z)`,
    `BatchGroundContact(gear, floor_z)`.
  - `control.py` — `rotor_speeds_for_wrench`, `BatchRateController`,
    `BatchAttitudeController`, `BatchVelocityController` (each `(params, gains,
    dt, n)` with `reset(drones)` where they have memory), `BatchAutopilot(vehicle,
    gains, dt, landed)` (`velocity_command` (n, 3), `yaw_command` (n,), `landed`
    (n,), `still_time`; callable `(t, states)`; `reset(drones)`),
    `BatchRatePilot(vehicle, gains, dt, landed)` (`thrust_command` (n,),
    `rate_command` (n, 3), `landed` (n,); callable `(t, states)`;
    `reset(drones)`); `Drones` (None, a mask or indices).
  - `collision.py` — `BatchCollisionMonitor(room, obstacles, sphere_centers,
    sphere_radii, floor_points, n, gates)`: callable `(t, states, drones)`,
    `margins`, `collided`, `check`, `time`, `position`, `airborne`, `kinds`,
    `collision(i)`, `reset(drones)`.
  - `referee.py` — `BatchReferee(gates, monitor, time_limit)`: callable
    `(t, states)`, `status` (codes `FLYING`, `FINISHED`, `CRASHED`, `TIMEOUT`;
    words in `STATUS_NAMES`), `end_time`, `passed`, `pass_time`, `pass_point`,
    `ended`, `next_gate` (-1 when none), `lap_time`, `passes(i)`, `status_name(i)`,
    `reset(drones)`; `forward_crossings(gate, start, end)`; `batch_referee(course,
    sphere_centers, sphere_radii, floor_points, n)`.
  - `simulation.py` — `BatchFlight(vehicles, referee, dt, start)`: `states`
    (n, 13), `clock` (n,), `step(controller)` (returns the laps that ended in that
    step), `reset(drones, start)`; `BatchController`.
- `jineta/picadero/` — the training layer (conventions in section 3):
  - `spaces.py` — `BoxSpace(low, high)` (float32; `shape`, `contains`,
    `sample(rng, n)`), `ActionLimits(thrust_min, thrust_max, max_rate)`
    (`space`, `commands(actions)`, `actions(thrust, rates)`),
    `autopilot_envelope(params, gains)`, `opening_points(gate)`,
    `closest_points(shape, points)`, `Surroundings(room, shapes, nearest=4)`
    (`size`, `points(positions)`, callable `(positions, to_body)`),
    `Observer(gates, lookahead=2, surroundings=None)` (`size`, `space`,
    `targets(next_gate)`, callable `(states, next_gate, previous_action)`);
    `ACTION_SIZE`.
  - `reward.py` — `RewardWeights(progress=1, gate=1, crash=5, time=0)`
    (`total`), `RewardTerms` (progress, gate, crash, time), `Reward(gates,
    weights, dt)` (`terms(...)`, callable), `episode_end(status)` →
    (terminated, truncated).
  - `env.py` — `CourseEnv(course, n=1, profile, limits, weights, dt=0.01,
    action_repeat=2, lookahead=2, nearest=0, margin=0)`: `reset(seed,
    options)`, `step(action)` (info: `final_observation`, `final_state`,
    `status`, `lap_time`, `episode_return`, `episode_steps`),
    `observation_space`, `action_space`, `step_time`, `np_random`; `flight`,
    `pilot`, `observer`, `reward`, `start`.
  - `baseline.py` — `BaselineSettings(cruise, lateral_accel, braking_accel,
    lookahead, approach, takeoff_height)` (one value or one per drone),
    `GatePath(points)` (`through(course, approach, takeoff_height)`, `point_at`,
    `closest`, `length`, `turns`), `PathGuidance(course, settings, n, path=None)`
    (`commands(states)`, `speed_limit(s)`, `reset(drones)`), `BaselinePilot`,
    `baseline_flight`, `fly_baseline` (statuses, lap times, clearances, flight),
    `sweep(course, ...)` → `SweepResult(best, lap_time, tried, robust,
    lap_times)`.
  - `gymnasium_adapter.py` — `CourseGymEnv(course, **env)` (`gymnasium.Env`,
    one drone), `CourseVectorEnv(course, n, **env)` (`VectorEnv`, same-step
    autoreset), `to_box`, `EPISODE_KEYS`. Needs Gymnasium; nothing imports it.
  - `training.py` — `fly(agent, env, rng=None, record=False)`,
    `Evaluation`, `evaluate(agent, course, n, seed, **env)`,
    `CourseSetEvaluation`, `evaluate_courses(agent, courses, **env)`,
    `run_training(env, steps, out, judge, ...)`, `train(course, steps, out, n,
    seed, settings, weights, evaluate_every, agent, nearest, margin, report)`,
    `random_courses(count, seed)`, `train_random(steps, out, k, n,
    renew_every, nearest, ...)`; `TRAINING_WEIGHTS` (time 0.25),
    `TRAINING_MARGIN` (0.05), `LOG_COLUMNS`, `VALIDATION_SEED` (1000),
    `TEST_SEED` (2000).
  - `courses.py` — `CourseRanges` (what a random course may be),
    `random_course(rng, ranges, name)`, `gate_count`, `CourseSetEnv(make_course,
    k, n, renew_every, **env)` (k courses of n drones, renewed in turn, with
    `CourseEnv`'s shape).
- `jineta/doma/` — learning (conventions in section 3):
  - `network.py` — `MLP(weights, biases)` (`create(sizes, rng, output_gain)`,
    `sizes`, `parameters`, `forward` → (outputs, cache), `backward(cache,
    grad)`, `state`, `from_state`), `orthogonal`, `Adam(parameters,
    learning_rate, beta1, beta2, epsilon)` (`step(grads)`, `state`,
    `load_state`), `clip_by_global_norm`, `save_arrays`, `load_arrays`.
  - `policy.py` — `RunningNormalizer(size, clip=10, epsilon=1e-8)` (`update`,
    callable, `state`, `from_state`), `gaussian_log_prob`,
    `gaussian_log_prob_gradients`, `gaussian_entropy`, `GaussianPolicy(network,
    log_std)` (`create`, `parameters`, `mean`, `sample`, `entropy`),
    `Agent(normalizer, policy, value, info)` (`create`, `act`, `values`,
    `save`, `load`).
  - `ppo.py` — `PPOSettings` (rollout_steps 64, epochs 5, minibatch 4096,
    gamma 0.99, gae_lambda 0.95, clip 0.2, learning rate 3e-4 annealed,
    max_grad_norm 0.5, entropy 0, hidden (64, 64), log_std -0.5), `gae`,
    `Rollout`, `Episodes`, `PPO(env, settings, seed, agent, episode_info)`
    (`collect`, `update`, `learn(steps, callback)`, `steps`, `agent`).
- `policies/first_course.npz` — the policy of phase 11 on the first course
  (3.235 s; trained on the author's computer).
- Tests (836 passing with Gymnasium installed, 830 and one skipped module
  without it; ruff check + ruff format clean):
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
  - indoor geometry (58): the overhang formula written out equal to NumPy's
    reductions bit for bit (2 and 3 components, 100 000 rows, one point at a
    time); box distances by hand (8 points: centre, face, past a
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
  - hitbox (33): the reach of a disc by hand (3-4-5, 1.25 m where |c| + r gives
    1.40, and with a thickness) and against its edges sampled every 0.5° (3 discs,
    3 thicknesses); box corners; the bounding sphere takes the farthest of rotors
    and body; rotor spheres hold their cylinders exactly (3-4-5); box spheres cover
    the box (4 grids, sampled) and reach by hand; union and extent; the Iris's 13
    spheres hold its rotor cylinders and its whole collision box, their radii and
    reach up, down, forward and sideways by hand (0.275 m high); the bounding
    sphere (0.3844 m) holds everything; the 13 spheres reach at most max(|c| + r)
    (0.2 mm past the bounding sphere) and give back 0.233 / 0.104 / 0.036 m up,
    forward and sideways; the profile carries them; rejects invalid input (8
    cases); read-only.
  - collisions (40), with the bounding sphere unless said: hops of the gear of
    0.2 mm and 9.9 mm are not take-offs, one of 10.1 mm and back is a floor crash
    (reference and batch monitors); straight at the north
    wall, the ceiling, a crate and a pillar, the moment exact (1e-9) and the
    obstacle named by its index; a sphere ahead of
    the nose on a 90° heading; the earliest of two contacts in one step wins; a
    start inside a wall is reported at the first sample; the floor margin is the
    lowest foot nose up 20° on five headings (1e-12); a free fall onto the floor
    within g dt²/(8v) of the exact moment; standing 3 s and taking off to a hover
    are not crashes, coming back down is; reset; rejects invalid input (4 cases);
    tunnelling: a 2 cm slab seen at every phase with steps 1 % below 2R + w and
    jumped 1 % above it; grazing a 5 cm pillar at 20 m/s misses at most
    ρ − sqrt(ρ² − s²/4) (found within 5 %). The Iris's 13 spheres: climbing to the
    ceiling they touch 0.233 m higher than the sphere, rolled 90° 0.197 m sooner
    than level (exact moments); nose first into a wall at 0.280 m; take-off and
    hover are no crash; a thin wall head on needs 58 m/s; a graze at 20 m/s is
    missed up to 3.1 cm by the samples alone, deeper than with the sphere. The
    swept check: the thin wall jumped by the samples found at every phase at the
    exact moment; pillar grazes from 2 µm deep found within 2e-5 s of the first
    touch; passing 10 µm or 1 mm away is no contact, within 1 µm is; a graze
    between samples beats a contact at the sample; the move is the chord; far from
    everything nothing is advanced; the Iris's 13 spheres' grazes found.
  - gates (41): rectangular frame distances by hand (7 points) and equal to its four
    bars (3000 random, seeded); torus distances by hand (5 points) and against its
    centre circle by brute force; the yaw is a heading (30°); many points at once,
    1-Lipschitz and gradient of length 1 outside the frame (both shapes); straight
    crossings forward and backward with exact fractions, a slanted one, six moves
    that do not go through, edges inside, the opening judged where the plane is cut;
    one crossing in a sampled flight at the exact moment (1e-12); a frame hit
    reported as a gate; rejects invalid input (6 cases); read-only.
  - referee (8): three gates in order (pass moments and lap time 1e-9); a later gate
    first does not count (timeout); a gate crossed backwards does not count and is
    then flown forward; a crash ends the lap (exact moment); a finish 2 ms before a
    crash in the same step is a finish, a crash 2 ms before the last gate is a
    crash; reset repeats a crashed lap; rejects invalid input.
  - course files (34): the first course loads (types, headings in radians); the
    format in the docstring is a valid course; optional parts; 19 mistakes in the
    format and 7 in the layout name their place (the z-down hint for a gate below
    the floor); a broken file and a typo name the file; the Iris rests at 0.054 m;
    the start state; the course builds its monitor and referee; a straight lap of a
    course finishes at the exact moment (1e-9).
  - course flight (3): the Iris finishes the first course at 3 m/s through the four
    gates in order without a contact (lap 8.81 ± 0.02 s); it starts on its gear,
    takes off and stays inside the room; every crossing is on the gate's plane and
    in its opening.
  - vector (22): `cross` bit for bit equal to `np.cross` (single pairs, stacks,
    broadcasting, components over 16 orders of magnitude), basis vectors,
    perpendicular to both, wrong shapes rejected; `matvec` equal to `@` to rounding
    on five shape combinations, its terms added in order, rows independent of the
    batch, wrong shapes rejected.
  - batch rigid body (12): quaternion matrices, derivatives (bit for bit) and
    normalisation against the reference; every body's derivative (1e-12) and 2 s
    of RK4 flight (1e-12) against `RigidBody`; a body alone or among 1000 bit for
    bit; free fall of tumbling bodies; one time per drone; invalid input.
  - batch multirotor (11): the floor contact against the reference on two floor
    heights; no force in the air; every drone's derivative against `Multirotor`
    (with and without contact, with a changing density and wind; 1e-12); alone or
    in a batch bit for bit; resting on the gear is at rest; commands start at zero;
    the 882 rotor commands of the reference lap replayed within 1e-9.
  - batch rate control (11): the mixer's inverse with both clips; 300 updates of
    50 drones against `RateController`; alone or in a batch bit for bit; 3 s of
    closed-loop flight of 12 drones within 1e-9; the exact discrete first order of
    the P-only loop; reset of chosen drones; invalid input.
  - batch autopilot (12): quaternion product, conjugate, rotation vector and
    `from_matrix` (every branch) against the reference; the attitude loop and 200
    updates of the velocity loop against the reference; four drones taking off at
    different moments, climbing, turning, descending and landing within 1e-9 of
    their reference flights and landed at the same steps; alone or in a batch;
    drones on the ground keep their loops at zero; a drone declared landed in the
    air takes off from clean loops; reset keeps `landed`; the first course flown
    by three drones within 1e-9 of the reference lap.
  - batch referee (12): 95 sampled flights (every kind of contact, grazes between
    samples, drones standing on the floor) against the reference monitor; margins;
    alone or in a batch; chosen drones checked and reset; laps (the reference lap,
    moved, backwards, random flights; two time limits) against the reference
    referee; forward crossings of both gate types; events in the same step and a
    crashed drone reset; a drone on its own clock; ended laps not judged again;
    invalid input.
  - batch flight (5): four speeds in one batch against the reference engine
    (states 1e-9, end, passes); a reset drone repeats its lap bit for bit and
    disturbs nobody; each lap's end reported once; `fly_laps`; invalid input.
  - Kalman filter (20): the running mean and σ²/k exactly; the steady-state gains of
    Kalata's alpha-beta filter; the lag behind a constant acceleration (a T²/β and
    its two companions); two updates one after the other equal one stacked update
    (information adds up); the continuous noise does not depend on how an interval
    is split, the piecewise one does; NEES inside its χ² interval over 300 seeded
    runs, and far outside it for an overconfident filter.
  - rate pilot (13): on the floor turning charges nothing (the rate loop alone
    charges its limits); take-off at the first step above the weight, not at
    exactly the weight; a take-off after waiting is a fresh pilot's bit for bit;
    in the air it is `RateController` bit for bit; declared landed it starts from
    a clean loop; it never lands by itself; the batch against the reference (8
    drones, 1e-9, landed at the same steps), alone or in a batch, reset per
    drone, landed integrals zero after every call, in the batch loop at 90 % of
    the weight no crash.
  - spaces (30): `BoxSpace` (limits, contains, seeded samples, read-only, bad
    limits); actions by hand, clipped, inverse, the Iris envelope; opening points
    by hand (rectangle, turned 30°, circle) and on the edge in order; the
    observation by hand nose North and East, the attitude matrix, 200 random
    attitudes rotated back, the last gate repeated, many drones bit for bit.
  - reward (13): a straight lap by hand (0.1 m every step, 14 m, three bonuses);
    backwards through a gate; a crash paid once; weights; the first course's lap
    adds up gate by gate; drones not flying get nothing; many drones; the time
    term (dt per step while the lap goes on); episode end; bad weights.
  - environment (14): reset and step shapes and types; least thrust stands and is
    truncated; three episodes against the reference engine (states 1e-9,
    rewards, end, final observation); alone or among four with restarts; a
    restarted drone as a fresh environment, landed, clean loop; time limit;
    seeds; action repeat; previous action clipped; bad input.
  - baseline (15): the path by hand, points along it, the closest point only a
    few segments ahead; the speed plan by hand; straight up first, progress never
    back (also when pushed back); the best lap (6.773 +- 0.02 s, > 5 cm clear);
    against the reference engine (1e-9); different settings as alone; through
    the environment's actions at 50 Hz (finishes) and at 100 Hz (the return gate
    by gate, 1e-9); the sweep keeps the robust setting and fails loudly; bad
    paths and settings.
  - Gymnasium adapter (6, skipped without Gymnasium): `check_env` passes;
    spaces; one and n drones bit for bit as `CourseEnv` (final observations
    included); the same as Gymnasium's `SyncVectorEnv` over copies; a finished
    lap's time in the info.
  - network (12): the forward pass by hand; gradients against central
    differences (3 shapes); orthogonal initial weights; a row alone or in a
    batch to 1e-13; Adam by hand; sin(x) learned; clipping; save and load bit
    for bit; bad input.
  - policy (8): log-probability and entropy against `scipy.stats`; their
    gradients; samples with the policy's mean and spread; a new policy near
    zero; the normaliser in pieces equal to all the data at once; clipping and
    a constant input; the agent acts with the mean and is saved bit for bit;
    bad input.
  - PPO (11): the experience of a scripted environment completed as each step
    ended; GAE by hand, with lambda 1 (Monte Carlo) and 0 (one step), cut at an
    end; a continuous bandit; the double integrator's LQR gain within 5 %
    (measured 1.6 %); the same seed the same training; the learning-rate
    schedule and diagnostics; a short training on the first course; bad
    settings.
  - training (10): the baseline through `fly` (lap, clearance, states recorded
    to where the lap ended); laps that time out and crash; an evaluation
    repeated; which evaluation is better (the margin first); a short training's
    log and agents, repeated by its seed; the margin turns the closest pass into
    a crash; a set of courses evaluated; a short training on random courses; a
    second stage counts the first's steps; the saved policy (3.235 +- 0.05 s,
    > 5 cm, the same lap in the reference engine).
  - courses and surroundings (11): closest points by hand (box face, edge and
    corner; column; torus); the room's faces and the nearest first, in body
    axes (nose North and East); an obstacle among the surfaces; the
    observation with surroundings (the old part unchanged, many drones bit for
    bit); 200 random courses keep their promises (ranges, spacing, the path's
    clearance from obstacles and walls); a seed repeats a course; a cautious
    baseline finishes six random courses; many courses fly as their own
    environments until renewed, then truncated with their last observation;
    seeds repeat; bad input.
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
  - `iris_course.py` — the Iris flies the first course through the gate centres at
    3 m/s (lap 8.81 s): the course from above and from the side, path coloured by
    speed, crossings marked. `fly_course(course, speed)` is reused by tests.
  - `benchmark.py` — times a lap of the first course in the reference engine
    (best of several runs), `--profile` shows where the time goes.
  - `batch_course.py` — five Iris at 2 to 4 m/s fly two laps each in one batch
    (the 3.5 and 4 m/s ones clip a gate), and the batch's speed against n.
    `GateGuidance`, `course_flight` and `fly_laps` are reused by tests.
  - `env_random.py` — the environment with random actions (64 drones: every
    episode a crash after ~0.8 s) and its speed against n.
  - `baseline_course.py` — the baseline's sweep on the first course (time to
    beat 6.77 s) and the best lap drawn with the planned path.
  - `train_course.py` — trains the Iris on the first course in two stages
    (without and with the 5 cm margin) and measures it against the baseline.
  - `policy_course.py` — flies a trained policy (by default
    `policies/first_course.npz`) and draws its lap with the baseline's.
  - `train_random.py` — trains on random courses and compares the policy with
    the baseline swept on 20 test courses (`--compare` for an agent).
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
     the floor). The events that matter give their moment interpolated inside the
     step (collisions, gate crossings, the lap time); the state at that moment is
     not stored (the run keeps samples only).
  2. The ISA functions are scalar-only (`if` on the altitude, `math`), so the batch
     engine cannot use them (it calls environment models with arrays). Indoors the
     density is constant over the height of a room; vectorise them (`np.where`)
     only if a batch ever flies outdoors.
  3. `RotatingBody` and `RigidBody` still duplicate Euler's equations (inertia
     validation is already shared). Revisit when a vehicle needs attitude-only runs.
  4. Mixer saturation is plain clipping. It already happens for an instant at every
     velocity step (one rotor at 0, another at the maximum). A smarter strategy
     (give up yaw before roll and pitch, as PX4 does) when it starts to matter —
     an aggressive racing policy will saturate often (phase 11).
  5. Motor dynamics (first-order lag, 12.5 ms up / 25 ms down on the Iris) not
     modelled. Needed before a policy can move to a real drone (phase 12); it adds
     one state per rotor.
  6. Body drag is the weakest number: generic PX4 ballistic coefficient, and a
     placeholder for the vertical axis. The resulting top speed (20.7 m/s at 45°)
     is plausible but not validated against any Iris measurement.
  7. Rotor effects left out: axial-inflow thrust loss, rolling moment, gyroscopic
     moments; rotor inertia (1–3 %) left out of the Iris inertia. Ground and
     ceiling effect (more thrust close to a surface) matter indoors (phase 12).
  8. (Closed in phase 10: the policy acts at 50 Hz, `action_repeat` = 2 over the
     100 Hz rate loop.)
  9. An agile multirotor profile (thrust-to-weight around 5–10, as data with open
     sources): the Iris (1.9) is enough to start, not to race.
  10. No position loop in the autopilot: the velocity loop holds vertical speed,
      not altitude (the Iris climbs ~0.6 m during a 10 m/s cruise). The phase 10
      baseline closes it outside, by pure pursuit of its path; a vehicle asked to
      hold a point still needs one.
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
  21. (Phase 12, step 1: training can be split over processes,
      `ParallelEnv`; its speed on the author's computer is to be measured.)
      Speed (one core, the first course, author's computer, end of phase 10): the
      batch engine at 1 024 drones does 58 093 drone-steps/s (~580× real time),
      the environment 32 475 steps/s (~650×; table in the phase 10 decision log).
      Swift trained at over 700×. Training (phase 11, Claude's workspace, two
      runs on two cores): ~10 500 environment steps/s with 256 drones on one
      course (20 M steps in 32 minutes), ~5 000 on two random courses of 128 with
      the surroundings. Next gains: several processes (one batch per core), a
      course per drone (loose end 30); compiled code (Numba) or a GPU only if
      that is not enough.
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
  26. (Closed in phase 9, step 6: the swept check.) What is left: the move
      between samples is a straight line, so a graze shallower than the gap between
      the true curved path and its chord (~ a dt²/8, 0.25 mm at 20 m/s² and 100 Hz)
      can still be missed, and a near miss within ~10 µm can take the advancement
      a few thousand iterations (measured: ~2 000 at 2 µm, ~100 at 1 mm).
  27. The hitbox spheres hold the PX4 collision geometry, which is itself a coarse
      envelope (a 0.47 x 0.47 x 0.11 m box for arms, body and legs). A real drone's
      shape, measured, comes with its profile (phase 14).
  28. (Closed in phase 10: the baseline plans through the gates, looking ahead
      along its path; the straight-at-each-centre guidance stays as an example.)
  29. (Closed in phase 10, step 1: the overhang formula written out, the same
      numbers bit for bit, the batch a third faster on the author's computer.)
  30. A batch has one vehicle profile and one course. Phase 11 trains on random
      courses with a batch per course (`CourseSetEnv`), which costs speed with
      small batches (decision log); a course per drone in the batch engine, and
      the vehicle's parameters per drone (phase 12), need per-drone arrays
      (gate positions, obstacles, mass, inertia).
  31. The batch rounds of the swept check end with the slowest drone, so a rare
      near miss of a few micrometres (a few thousand rounds, loose end 26) holds up
      its whole batch for that step.
  32. (Closed in phase 10: `RatePilot` has a landed state, and the environment
      puts a restarted drone back landed.)
  33. (Closed in phase 11, step 6: the nearest surfaces in the observation,
      `CourseEnv(nearest=4)`.)
  34. (Closed in phase 11, step 1: `picadero/gymnasium_adapter.py`, which passes
      Gymnasium's checker; the learning algorithm is the project's own.)
  35. (Closed in phase 11, step 3: a running normaliser, part of the agent.)
  36. The action envelope stays the autopilot's (decided at the start of phase
      11): the policy beat the baseline within it. Widening it (PX4's
      documentation suggests 720 deg/s for acro pilots) is for an agile profile
      (loose end 9).
  37. The baseline is a robust pure-pursuit follower, not the classical optimum;
      a minimum-time trajectory through the gates with a tracking controller
      would set a harder time to beat. The sweep's robustness test (four copies
      at 2 %, 5 cm) is a choice, and the best lap is sensitive to it.
  38. (Closed in phase 11, step 5: a time penalty of 0.25 per second, found by
      measurement; the decision log has the experiments.)
  39. A lap ends at the last gate, so a policy does not care what comes after
      it: Claude's first policy crossed the last gate rolled 144° at 1.25 m, and
      a real drone would crash right after (the author's, now the repository's,
      crosses it tilted 18° but still turning fast). A real finish needs the drone
      under control after the last gate (an episode that goes on to a hover, or
      a penalty on the attitude at the finish).
  40. The policy on random courses is undertrained (step 6: 1 of 20 test
      courses after 23 M steps): it needs more experience, and more courses per
      update, than Claude's workspace gives in a session; `python -m examples.train_random` on the
      author's computer (8 courses of 128 drones) is the next measurement.
  41. The first course's policy flies hard (8.6 m/s, up to 84.6° of tilt) with
      instantaneous motors (loose end 5), a perfect state and no delay: what it
      does may not survive the realism of phase 12. Its margin (13 cm) and its
      100 % of noisy laps are a start, not a proof.
  42. A training repeats only on the same computer (the network's products
      round differently elsewhere, and a training amplifies it): a policy is
      kept as a file, and its regression tests allow for rounding. Measured: the
      same recipe gave 4.325 s in Claude's workspace and 3.235 s on the author's
      computer; one training is one sample.
  43. Training saves only the best and the last agent. To show how a policy
      learns (the 3D viewer of phase 13: from crashing to a clean lap), a
      training should keep an agent every so often (e.g. every 1 M steps).
  44. The ceiling effect is not modelled (phase 12, step 3: no open model with
      coefficients for a rotor of the Iris's size could be checked), nor the
      ground effect over the top of an obstacle, nor the multirotor terms of the
      ground effect (neighbouring rotors, the body). A drone flying close to a
      ceiling, or one with small propellers, would need them.
- Next: phase 12 (realism: motor lag, ground effect, sensor noise and delay,
  randomised parameters), whose step plan is drawn up with the author at its
  start. Proposed as its first step, to be decided then: a training spread over
  the author's 8 cores (one batch per process; estimated, not measured, x2-3 for
  one course and x3-5 for random courses, since the network's update stays in
  one process), because every step of phase 12 changes the physics and means
  training again. The random-course training (loose end 40) is worth running
  after that. New chat in the Project: "Seguimos con Jineta, fase 12:
  realismo".

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
  insertions lose imports and blocks. Each step ends with one block of commands
  (asked by the author): the checks, then the commit (one-line message) and
  push, guarded so that nothing is committed if a check failed (each check's
  `$LASTEXITCODE` gathered in `$ok`; `if ($ok) { git add -A; git commit ...;
  git push } else { ... }`).
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
