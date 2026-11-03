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
Monte Carlo analysis, and find which one works best in which scenarios.

**Where it stands (end of phase 8): the stage and the baseline.** Every engagement so
far is the baseline the strategies will be measured against, with each part in its
simplest form on purpose, so that each later phase changes one thing:
- knowledge: perfect, the true position of the threat (phase 9: sensors + Kalman
  filter);
- guidance: pure pursuit, straight at where the threat is now (phase 10: guidance
  that predicts, e.g. proportional navigation);
- assignment: fixed by hand in the scenario file, one interceptor per threat; it
  launches when its own threat enters its pad's alert range. There is no assignment
  logic yet, not even "in order of detection" (phase 11: assignment strategies, the
  simplest being in order of detection, and re-assignment);
- coordination: none, each interceptor flies alone (phase 11: centralised vs
  distributed command, communications);
- verdict: captured / missed per pair in one run (phase 13: statistics over many
  runs per strategy and scenario).
The scenario files make the comparison possible: the same threats, pads and
interceptors flown with one strategy and then another.

Name: *jineta* = historical Spanish light cavalry — small, agile units that
coordinated hit-and-run attacks. Team brand: **Ibero**.

Author: 3rd-year aerospace engineering student. First large software project, so
code should be clear and every concept explained before it is used.

## 2. Environment

- OS: Windows (two machines: laptop + desktop, synced via GitHub)
- Python **3.13** in a `.venv` virtual environment
- Editor: VS Code (extensions: Python, Pylance, Ruff, GitLens)
- Dependencies: `numpy`, `scipy`, `pyyaml`, `matplotlib`, `pytest`, `pytest-xdist`,
  `ruff` (pinned in `requirements.txt`). After pulling on the other computer, run
  `pip install -r requirements.txt` when `requirements.txt` changed: `git pull`
  brings the list, not the packages.
- Run tests from repo root: `pytest -n auto` (pytest-xdist spreads the tests over
  every core) or `pytest -v` (one by one, every name listed); one file:
  `pytest tests\test_world.py`; by name: `pytest -k cyclic`; the last failures:
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
- The full test run, in one process, takes ~60 s (463 tests; 62 s in Claude's
  workspace at the end of phase 8): the closed-loop tests fly up to 120 s each at
  100 Hz; the fixed-wing ones are ~18 s and the engagement and launch ones ~20 s of
  the total. `pytest -n auto` divides that by the number of cores.
- The interactive viewers (`docs/*_viewer.html`) open in their own desktop window
  with `pywebview` (optional: `pip install pywebview`; without it they open in the
  browser). They need internet: three.js and the fonts come from CDNs.
  `python -m viz.app` opens the start menu (`docs/index.html`, the scenario pages by
  type of threat); `python -m viz.app <page>` reopens one page without simulating
  again; `python -m examples.run_scenarios --no-window` flies every scenario file and
  rewrites every page and the menu (~50 s in Claude's workspace); the examples take
  `--no-window` to only write their files.
- `scipy` is used since phase 6b (trim: `scipy.optimize.root`).
- Regenerate dependencies with `pip freeze | Set-Content -Encoding ascii
  requirements.txt`. PowerShell's `>` writes UTF-16, and `Out-File -Encoding utf8`
  in Windows PowerShell 5 adds a BOM (an invisible mark at the start of the file;
  pip tolerates it, but it does not belong there). Package names are ASCII.
- A block of commands pasted into PowerShell does not stop when one line fails:
  the next lines still run (in phase 8 a commit and push went ahead after
  `pytest -n auto` failed on the computer without pytest-xdist). So the checks come
  first, and the commit only once they passed.

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
  axes are the principal axes, and "axis 1, 2, 3" means x, y, z. These are FRD axes
  (forward-right-down). Sources in FLU (Gazebo, ROS: x forward, y left, z up) are
  converted with `jinete.profiles.flu_to_frd` (flip y and z).
- **Reference point:** the body's position is its centre of mass, and the inertia
  tensor is expressed about the centre of mass in body axes.
- Euler angles (roll φ, pitch θ, yaw ψ, ZYX order) only for display/input —
  never as internal state (gimbal lock at θ = ±90°).
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
  negative down component.
- **Airspeed** is the velocity relative to the air, `v − wind`. Aerodynamic forces
  use airspeed; the state keeps the ground velocity. In body axes:
  `jinete.aero.body_airspeed(t, state, wind) = R(q)ᵀ (v − wind)`, components u
  (nose), v (right wing), w (down); w > 0 means positive angle of attack.
- **Vehicle profiles are data, not code.** Each vehicle is a frozen (immutable)
  parameter dataclass (e.g. `MultirotorParams`), one constant per vehicle in
  `jinete/profiles.py`, with a comment citing where every number comes from. Models
  never know which vehicle they fly. The controller gains live next to the
  physical parameters of each profile, not inside them. The scenario files name a
  profile (`MULTIROTOR_PROFILES`, `FIXED_WING_PROFILES` in `profiles.py`); the
  numbers stay in code next to their sources.
- **Vehicle interface:** a vehicle contains a `RigidBody` and hands it its own
  `force_moment` (vehicles do not subclass `RigidBody`). It exposes
  `derivative(t, state)` and is commanded through attributes that a controller sets
  in `pre_step` (multirotor: `rotor_speed_command`, one value per rotor [rad/s]).
- **Rotor spin:** `+1` = counter-clockwise seen from above (PX4/Gazebo "ccw"); the
  body reacts the other way, so a `+1` rotor yaws the body right (positive r).
- **Simulation step order:** `pre_step(t, y)` (controllers) → one RK4 step →
  `post_step(y)` (e.g. renormalise the quaternion) → `stop(t, y)`.
- **Controllers** are dataclasses with `update(...)`, run once per step from
  `pre_step`; the ones with memory (integrals) have `reset()`. Per-axis gains are
  3-vectors: `[roll, pitch, yaw]` for the rate and attitude loops, `[north, east,
  down]` for the velocity loop. Gains are expressed so that they do not depend on
  the vehicle (angular acceleration per rate error, rate per attitude error,
  acceleration per velocity error). All the gains of one vehicle live in a
  `MultirotorGains` next to its profile (`IRIS_GAINS`).
- **Autopilot interface:** `Autopilot(vehicle, gains, dt)` is a callable; pass it
  as `pre_step`. Whoever guides the vehicle sets `autopilot.velocity_command`
  (NED [m/s]) and `autopilot.yaw_command` ([rad]).
- **Target track interface:** a target is any callable
  `track(t) -> (position NED [m], velocity NED [m/s])` (`TargetTrack` in
  `jinete/targets.py`). Scripted targets are closed-form, so they hold no state and
  are not integrated. `track_positions(track, times)` samples one as an (n, 3) array.
- **Guidance interface:** a guidance law maps positions (later also velocities and
  estimates) to a NED velocity command; it meets the autopilot only through
  `autopilot.velocity_command`, so changing the law never touches the control loops.
- **Recorded targets:** a vehicle that does not react to anyone can be flown first
  and replayed as a target: `RecordedTrack(times, positions, velocities)` (cubic
  Hermite between samples, no extrapolation) is a `TargetTrack` like the scripted
  ones.
- **Fixed-wing vehicle interface:** `FixedWing(params, propulsion=None,
  density, wind)`, built like `Multirotor` (holds a `RigidBody`, `derivative(t,
  state)`). Commanded through `surfaces` (`ControlSurfaces`, frozen: assign a new
  one) and `throttle` in [0, 1], both starting at neutral / 0. `propulsion=None` is
  a glider, i.e. "engine off" (throttle 0 with a propeller is not: it windmills and
  brakes). Signs as in the source: elevator > 0 trailing edge down (nose down),
  aileron > 0 rolls right, rudder > 0 yaws the nose left.
- **Trim:** `trim_level_flight(aircraft, speed, altitude, course)` returns a
  `Trim(state, surfaces, throttle, alpha, beta)` for wings-level flight (φ = 0) at
  that airspeed, and leaves the aircraft commanded with it.
- **Fixed-wing autopilot interface:** `FixedWingAutopilot(vehicle, trim, design,
  dt)` is a callable; pass it as `pre_step`. Whoever guides the aircraft sets
  `course_command` (direction of the ground velocity, clockwise from North [rad]),
  `altitude_command` [m] and `airspeed_command` [m/s]; they start at the trim. The
  design choices (limits, bandwidths, damping) are profile data
  (`SMALL_FIXED_WING_AUTOPILOT`, an `AutopilotDesign`); the gains are computed from
  them at the trim by `design_gains`. It runs at 100 Hz like the Iris.
- **Euler angles** for input, display and the controllers' angle errors only:
  `quaternion.from_euler(roll, pitch, yaw)` and `quaternion.to_euler(q)` (ZYX).
- **Relative position** between two tracks is always `r = r_target − r_pursuer`.
- **Engagement result:** `engage(...)` (`jineta.arena.engagement`) returns an
  `Engagement` with `times`, `states` (interceptor), `target_positions`, `closest`
  (`ClosestApproach`: `time`, `distance`, `step`), `captured` and
  `pursuit_feasible`. `captured` means the closest approach fell inside the capture
  radius (default 0.5 m).
- **World interface** (`jineta.arena.world`): a `World(entities, rules)` flies every
  entity in one `run` (`world.fly(dt, t_end, stop)` → `Flight`). Entities:
  `VehicleEntity(name, dynamics, initial_state, control, waiting)` (13 states),
  `PointEntity(name, initial_position, control, waiting)` (an ideal point flying
  `velocity_command`), `TrackEntity(name, track)` (a scripted track, no state). A
  control is `control(t, own_state, snapshot)`; a rule is `rule(snapshot)` and runs
  first. The snapshot (`position(name)`, `velocity(name)`, `of(name)`) is the world
  at the start of the step, read-only, the same for everyone. `world.launch(name, t)`
  and `world.destroy(name, t)` change an entity's status; a waiting or destroyed
  entity has a zero derivative. `Flight` has `times`, `states`, `state(name)`,
  `positions(name)` and `launched_at`.
- **Engagements in the world:** a `Pairing(interceptor, target, capture_radius,
  margin, dome)` per assignment; `fight(world, pairings, dt, t_end)` → `Battle`
  (`flight`, one `DuelResult` per pair: `outcome`, `closest`, `end_time`). Outcomes:
  `captured` (both destroyed), `passed`, `escaped` (the target left the pair's
  dome), `target lost` / `interceptor lost` (destroyed in another pair), `open`.
  `pursuit(autopilot, target, speed)` is the pure-pursuit control of a multirotor.
- **Launch rules** (`jineta.arena.launch`): `Dome(site, radius)`,
  `LaunchOnAlert(interceptor, dome, threats)` (the first step a watched threat is
  inside), `LaunchAt(interceptor, time)`.
- **Scenario files** (`scenarios/<folder>/<case>.yaml`, format in the docstring of
  `jineta/arena/spec.py`): positions `[north, east]` plus `altitude` (up), metres,
  m/s, seconds; degrees only in the file, in keys that say so (`course_deg`,
  `turn_rate_deg_s`); one atmosphere and one uniform wind for the whole scenario;
  vehicles by profile name; an interceptor starts on a `site` (waiting for its
  target's alert when the site has an `alert_range`) or at a `start` (hovering, from
  t = 0). `load_scenario(path)` → frozen `ScenarioSpec`; `build(spec)` → `Setup`
  (world, pairings); `play(spec)` → `(Setup, Battle)`. Errors are `ScenarioError`
  and say the file, the place and, when a name is close to a valid one, the
  suggestion.
- **Scenario pages:** a folder of `scenarios/` is one viewer page: `page.yaml`
  (`title`, `description`, `page`, `subtitle`, `cases`) plus one file per case (a
  tab; the tab id is the file name). `scenarios/menu.yaml` lists the folders of the
  start menu, in order; their pages link back to it.
- **Viewer pages and the start menu:** `docs/index.html` (written by
  `viz.menu.write_menu` from the scenario files) lists the scenario pages by type of
  threat and start ("Generic target · scramble", "Fixed wing · scramble", ...), never
  by phase number; the cases and how each of their pairs ended are read from the
  pages themselves. The export says what the target is (`target_kind`: "generic" or
  "fixed_wing") and may carry its attitude. A case with several entities adds
  `others` (every entity outside the first pair), `pairs` (each pair and its end) and
  `domes`; a case with one pair is exported exactly as before. In the viewer, the key
  data of every pair (range, status, closest approach) is always shown at once;
  choosing a pair only moves the camera and the detailed telemetry to it. The menu
  lists, for each page, every case and every pair with its result and closest
  approach.
- **Viewers read exported data in the simulator's conventions** (NED, quaternion
  body → NED, SI). Any change of axes happens inside the viewer, never in the
  export: the three.js viewer maps NED to (East, Up, −North).
- Parameter classes validate their inputs in `__post_init__` and raise
  `ValueError` with a clear message. Inertia tensors go through
  `core.inertia.inertia_tensor` (symmetric, positive definite, triangle inequality)
  and are stored read-only.
- Use `zip(..., strict=True)` so sequences of different length fail loudly instead
  of being silently truncated (ruff B905).
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
│   ├── campo/      # environment: atmosphere.py (ISA), wind.py; later terrain
│   ├── jinete/     # vehicles: aero, multirotor, control (multirotor autopilot),
│   │               #   fixed_wing, propulsion, trim, stability, fixed_wing_control,
│   │               #   profiles, guidance, targets
│   ├── atalaya/    # sensors (radar, EO, IMU) + Kalman filtering
│   ├── capitan/    # target assignment / threat prioritisation
│   ├── escuadron/  # inter-unit coordination and communications
│   └── arena/      # composes the layers: world.py (every entity in one loop),
│                   #   engagement.py (pairs, outcomes, engage), launch.py (alert
│                   #   rules), spec.py (scenario files), scenario.py (spec -> world)
├── tests/
├── scenarios/      # scenario files (YAML): menu.yaml + one folder per viewer page
│                   #   (page.yaml + one file per case)
├── examples/       # runnable scripts; run_scenarios.py flies every scenario file
├── viz/            # visualisation layer: viewer.py (export), viewer_template.html,
│                   #   pages.py (scenario folders -> pages), menu.py +
│                   #   menu_template.html (start menu), app.py (window)
├── docs/           # generated figures and viewer pages (committed, for the README)
├── pyproject.toml  # pytest + ruff configuration
└── DESIGN.md
```

`arena` sits on top of the other layers because it composes them (`core`, `campo`,
`jinete`): inside any of them the dependency arrows would point upwards. `viz` reads
what `arena` produces and never runs the physics; the scripts in `examples/` fly the
scenarios and hand the results to `viz`. The package was planned as
`jineta/scenario/`; next to the `scenarios/` data folder the name invited a mistake
(files ended up in the wrong folder), so it took a themed name of its own: the
arena, where everyone meets.

Simulation flow: `atalaya` detects → `capitan` decides → `escuadron` coordinates →
each `jinete` flies using `core` inside the `campo`.

**The simulator never draws.** It produces data (states + events over time).
Visualisation is a separate layer that reads that data:

1. matplotlib, for debugging and analysis plots.
2. Since phase 7: a standalone **three.js engagement viewer** (one HTML file, data
   embedded) to replay one engagement in 3D: orbit, zoom, timeline, telemetry.
3. Multi-drone scenarios: export to **CesiumJS** (CZML) for a web 3D globe with real
   terrain and satellite imagery — the showcase/demo view (phase 12).
4. Optional: Godot real-time viewer.

## 5. Roadmap

| Phase | Content | Status |
|---|---|---|
| 1 | Integrators (Euler, RK4) | ✅ Done |
| 2 | Point mass: gravity + drag, simulation loop | ✅ Done |
| 3 | Rotation: quaternions, Euler's equations | ✅ Done |
| 4 | Full 6-DOF rigid body | ✅ Done |
| 5 | Environment: ISA atmosphere, wind | ✅ Done |
| 6a | Quadrotor (Iris profile) + cascaded attitude/velocity control | ✅ Done |
| 7 | **End-to-end MVP:** one interceptor vs one target, simple guidance | ✅ Done |
| 6b | Generic fixed-wing threat model (aero coefficient tables) | ✅ Done |
| 8 | Scenarios in YAML, multiple entities | ✅ Done |
| 9 | Sensors + Kalman filter | 🔄 In progress |
| 10 | Guidance (proportional navigation) | ⏳ |
| 11 | Coordination strategies (assignment, comms) | ⏳ |
| 12 | Real terrain + Cesium visualisation | ⏳ |
| 13 | Monte Carlo analysis, final README, demo video | ⏳ |

Order: 6a → 7 → 6b → 8 … (integrate early: the MVP uses a scripted target, and the
fixed-wing model replaces it afterwards).

**Phase 6a — done**, in 8 steps (each with tests green, ruff clean, commit + push):
airspeed in body axes; rotors, mixer and Iris profile; multirotor force model;
`pre_step` hook (discrete control, zero-order hold); rate loop (PI); attitude loop
(quaternion error); velocity loop + `Autopilot` + example; this document.

**Phase 7 — done**, in 6 steps (tests green and ruff clean at each; steps 1 and 2
committed on their own, 3–6 together): scripted targets + closest approach interpolated
inside a step; pure pursuit + engagement loop; MVP example with figure, pass/fail
verdict and the first real README; interactive 3D viewer; scramble example (launch
on alert from the ground); this document. Every
layer worked together for the first time and the interfaces held (autopilot
`velocity_command`, `pre_step`, `stop`). Result: the Iris catches a slow target
(0.49 m at T+21.1 s) and cannot catch a fast one (miss 10.8 m): it was ordered to
30 m/s and never flew faster than 22.0 m/s.

**Phase 6b — done**, in 8 steps (tests green and ruff clean at each; committed as
1–2, 3 in three parts, 4, 5 and 6–8). A 6-DOF fixed-wing drone driven by
aerodynamic coefficients and an electric propeller, flown by its own autopilot,
replaces the scripted target. The plan was kept; step 7 was made more concrete
with the author during step 4 (see the decision log).
1. ✅ **Air data**: V, α, β from the body airspeed and the inverse.
2. ✅ **Aerodynamic forces and moments** (`fixed_wing.py`) and the generic small-UAV
   profile `SMALL_FIXED_WING` (Aerosonde-class open data, Beard & McLain).
3. ✅ **Propeller and vehicle**: electric motor + propeller model (`propulsion.py`),
   the `FixedWing` vehicle, both checked against the source's reference values;
   power-off glide test (tan γ = −C_D/C_L).
4. ✅ **Trim** (`trim.py`): wings-level flight from 12 to 32.45 m/s at 100 m.
5. ✅ **Dynamic modes** (`stability.py`): the five modes of the linearised model
   against the classic approximations; phugoid and spiral also in simulation.
6. ✅ **Autopilot** (`fixed_wing_control.py`): course, altitude and airspeed hold by
   successive loop closure; orbit guidance.
7. ✅ **The fixed wing as the threat**: recorded and replayed as the target of
   `engage`; example `fixed_wing_scramble.py` (launch on alert, as the phase 7
   scramble) with a captured and a missed case; the viewer draws an aircraft with
   its attitude; start menu with both scenario pages.
8. ✅ **Close the phase**: this document and the README.

Result: the Iris launches from its pad when the fixed wing comes within 400 m. A,
loitering at 18 m/s on a 150 m orbit: launch T+20.71 s, capture 0.49 m at
T+63.32 s. B, passing straight at 30 m/s: launch T+10.72 s, miss 49.74 m at
T+20.39 s; the run ends when it leaves the dome at T+34.93 s.

**Phase 8 — done**, in 8 steps (tests green and ruff clean at each; each step
committed on its own, after a first commit with the plan). Scenarios became data
(YAML files in `scenarios/`), and one simulation loop flies every entity of a
scenario together. Loose ends 16, 18, 19, 20, 22 and 32 resolved (31 at the start).
Decided with the author at the start: the vehicle profiles stay in `profiles.py` and
the files name them; the viewer learns to show several entities in this phase; the
MVP page gets a card in the menu. The plan was kept, with four changes discussed or
reported on the way: the package is `jineta/arena/`, not `jineta/scenario/` (step 1,
with the author); a scenario has one atmosphere for every vehicle, which changed the
fixed-wing scramble's numbers (step 4); the menu order comes from
`scenarios/menu.yaml` (step 6); an interceptor on a pad launches on its own target's
alert, not on any threat's (step 7).
1. ✅ **World loop** (`jineta/arena/world.py`): vehicles, ideal points and scripted
   tracks stacked in one state vector and flown by the existing `run`; every control
   reads the same read-only snapshot; a waiting entity has a zero derivative. Bodies
   flown together are bit-for-bit the same as alone; cyclic pursuit follows its
   exact discrete law (1e-9) and meets at the closed-form time (< 0.4 %).
2. ✅ **Engagements in the world** (`jineta/arena/engagement.py`): pairs, a referee
   after every step, captures that destroy both, passes, losses; `engage` rebuilt on
   it, bit-for-bit the same, and moved into `arena` (loose end 18).
3. ✅ **Launch on alert and threats in the same loop** (`jineta/arena/launch.py`):
   the alert dome and the launch rules leave the example (loose end 22); the dome
   ends a pair (escaped); the fixed wing flies in the same loop as the Iris and
   gives the recorded engagement (loose end 19); a threat that flees from its
   pursuer follows its exact 1-D law (1e-9).
4. ✅ **Scenario files** (`jineta/arena/spec.py`, `scenario.py`): YAML into frozen,
   validated specs, clear errors with suggestions, profiles by name, ISA or sea
   level and a uniform wind; an engagement in ISA and wind (loose end 16).
5. ✅ **The existing scenarios as files** (`scenarios/pursuit`, `generic-scramble`,
   `fixed-wing-scramble`), `python -m examples.run_scenarios`, examples that only
   load, report and plot (loose end 20); the MVP and generic-scramble pages carry the
   same data as before.
6. ✅ **Menu from the scenario files** (`scenarios/menu.yaml` + each `page.yaml`); the
   MVP page has its card and links back. Checked in headless Chromium; the author
   checks pywebview.
7. ✅ **Several entities in the viewer, and a scenario that needs them**
   (`scenarios/two-pads`): every vehicle drawn, every alert dome, every pair's data at
   once in a pair board (clicking a row, or key N, follows that pair).
8. ✅ **Close the phase**: this document and the README.

After the close, at the author's request (he tried the app): in a case with several
pairs the closest approach and the status were only visible for the selected pair, so
every pair's key data is now always on screen, and the start menu was redesigned to
read like the index of a test campaign.

Result: three threats against two pads (case A): the north Iris catches the slow
scripted threat (0.49 m at T+61.02 s, launched T+33.83 s), a south Iris the
loitering fixed wing (0.50 m at T+65.78 s, launched T+15.97 s), and the fast one
escapes the other (29.55 m, out of the dome at T+49.69 s). Two Iris on one loitering
drone (case B): the north one catches it (0.49 m at T+71.26 s), the south one is left
with "target lost" 48.05 m away. The phase 7 cases keep their numbers; the
fixed-wing scramble, now with the Iris in ISA air like the drone: A 0.48 m at
T+62.14 s (was 0.49 m at T+63.32 s), B missed by 49.80 m (was 49.74 m).

**Phase 9 — in progress.** Plan agreed with the author at the start of the phase
(each step: tests green, ruff clean, commit + push). The interceptors stop seeing the
true position of their threat: a ground radar measures it with noise, a Kalman
filter estimates position and velocity, and the guidance and the launch use that
estimate. A prototype run before the plan (not in the repo; radar at the pad, 10 Hz,
5 seeds, generic-scramble case A) showed why the phase has two sensors: with the
radar alone the capture at 0.5 m became a coin toss (1 to 3 captures in 5, misses
of 0.5–1.3 m, worse with more angle noise), because the cross-range error of a
ground radar grows with the range (r·σθ, ~3.5 m at 400 m and 0.5°); with an onboard
seeker in the last 100 m, 5 captures in 5. Decided with the author: the onboard
seeker (step 6) is part of this phase. Kept from the start: perfect knowledge stays
available (`knowledge: perfect`), so every existing page flies bit for bit as
before and the effect of the sensors can be measured against it; the referee keeps
judging with the truth (it is the umpire, not a participant); the interceptor knows
its own state perfectly (navigation errors are not modelled); the radar knows which
measurement belongs to which threat, detects with probability 1 and has no false
alarms; the link from the radar to the interceptor is perfect and instant (phase 11).
1. ✅ **Kalman filter** (`jineta/atalaya/kalman.py`): predict and update (Joseph
   form), the constant-velocity model with white-noise acceleration (continuous and
   piecewise), NEES, the alpha-beta gains. Tests: the running mean and σ²/k exactly;
   the steady-state gains of Kalata's alpha-beta filter; the lag behind a constant
   acceleration (a T²/β and its two companions); two updates one after the other
   equal one stacked update (information adds up); the continuous noise does not
   depend on how an interval is split, the piecewise one does; NEES inside its χ²
   interval over 300 seeded runs, and far outside it for an overconfident filter.
2. ⏳ **Ground radar** (`jineta/atalaya/radar.py`): range, azimuth and elevation from
   its site, Gaussian noise, its own rate and maximum range, a seeded random
   generator of its own; a generic short-range radar profile with open sources.
   Tests: noise-free = exact geometry; sample mean and covariance converge; the
   cross-range error is r·σθ; the measurement schedule; no measurement out of range.
3. ⏳ **EKF and tracks** (`jineta/atalaya/tracker.py`): the spherical measurement
   enters the filter through its Jacobian (azimuth wrapped); a track starts from two
   measurements and can be predicted to any time. Tests: the Jacobian against finite
   differences; NEES consistent on a straight line; the estimate beats the raw
   measurement; the lag in a turn.
4. ⏳ **Sensors in the world** (`jineta/arena`): the radar as a rule (the only thing
   that reads the truth from the snapshot); the guidance reads a knowledge source
   (perfect or a track); launch when the radar detects the target instead of on the
   ideal dome. Regression: without sensors the existing cases fly bit for bit.
5. ⏳ **Scenario files**: a `radar:` block on a site, a `seed`, `knowledge: perfect |
   radar` per interceptor, with their typical mistakes tested; a new page that flies
   the same threats with perfect knowledge and with the radar.
6. ⏳ **Onboard seeker and fusion**: range and angles measured from the interceptor,
   with a field of view and a maximum range; the interceptor points its nose at the
   target (loose end 15); the radar and the seeker feed the same filter. A case with
   the radar alone against radar + seeker.
7. ⏳ **Viewer**: the measurements, the estimated track, the 3σ uncertainty
   ellipsoid and the estimation error in the telemetry; checked in headless Chromium.
8. ⏳ **Close the phase**: this document and the README.

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
- **ISA scope:** two layers, troposphere (0–11 km, −6.5 K/km) and isothermal layer
  (11–20 km). Altitude is treated as geopotential without conversion (difference
  ~0.17 % at 11 km). Above 20 km results are not valid and this is documented, not
  enforced: interceptor/threat scenarios stay far below. Below sea level the
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
  Threats are generic, representative classes (e.g. "small fixed-wing loitering
  drone", "subsonic cruise-missile-like target") with plausible public-domain
  numbers, not reproductions of specific real weapons. The value of the project is
  the engagement and coordination analysis, not weapon data.
- Shared interfaces (`GRAVITY`, `DensityModel`, `WindModel`, `sea_level_density`,
  `no_wind`) moved from `point_mass.py` to `core/environment.py`, so `RigidBody`
  and `campo` no longer depend on the point-mass model. `point_mass` still imports
  them, so old imports keep working, but new code imports from `environment`.
- Physical validity is checked once, at construction: mass > 0, drag area ≥ 0,
  inertia symmetric + positive definite + triangle inequality (flat-plate equality
  allowed). The validated inertia is read-only because its inverse is cached; a
  silent in-place change would leave a stale inverse.
- An end-to-end MVP (phase 7) comes right after the first vehicle, before YAML
  scenarios and sensors: integrate early, then replace one simplified piece per
  phase.
- The README is filled in when there is something to show (phase 7 figure, phase 13
  final). Until then this document is the project log.

Phase 6a (quadrotor):
- **Phase 6 is split**: 6a quadrotor + control (needed by the MVP), then phase 7,
  then 6b fixed-wing. The MVP target is a scripted trajectory.
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
  at 3.73 N per rotor = 799 rad/s, 73 % of the 1100 rad/s maximum): enough for the
  MVP against a slow target, low for an interceptor. A more aggressive interceptor
  profile will be added later as data, with its own sources.
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

Phase 7 (MVP):
- **Scripted targets are closed-form tracks**, not integrated: exact at any time,
  cheap, and with no state, so the engagement evaluates them in `pre_step` at the
  time of each step. `TurningTarget` is a level circle of signed radius `V/ω`
  (centre to the right for ω > 0); ω = 0 falls back to a straight line.
- **Closest approach interpolated inside the step** (`core/events.py`): inside each
  step the relative motion is taken as a straight line, `r(s) = r_i + s·Δr`,
  minimised at `s* = −r_i·Δr / Δr·Δr` clipped to [0, 1]. Exact for constant
  relative velocity, O(dt²) otherwise; the nearest sample alone can be off by up to
  `v_rel·dt/2`. Resolves loose end 1 for the miss distance.
- **Pure pursuit** `v_cmd = speed · (r_t − r_p)/|r_t − r_p|`, zero below 1e-6 m
  (line of sight undefined). Tested against Bouguer's closed-form pursuit curve:
  the invariant `C = 2r / (s^(k−1)(1 + s²))`, `s = tan(φ/2)`, stays constant
  (1e-5), the capture time `T = r0 (k + cos φ0) / (Vt (k² − 1))` (to 0.02 s), and
  for k < 1 the minimum separation `r_min = C s*^(k−1)/(1 + k)`. The guidance tests
  fly an ideal point pursuer through the same RK4 loop, so they test the law, not
  the vehicle.
- **The real vehicle can only be later than the ideal pursuer**: its velocity lags
  the command. The closed-loop Iris test checks the capture time between `T` and
  `T + 3 s` (the margin is empirical, not derived; ~1.8 s was seen).
- **`engage` runs the interceptor only**; the guidance writes the velocity command
  in `pre_step` and then calls the autopilot. Step = `autopilot.dt`, so control and
  simulation cannot disagree. The loop integrators are reset at the start, so one
  autopilot can fly several engagements with identical results.
- **Stop condition with memory** (`EndOfPursuit`): stop on capture (inside the
  radius) or on a pass (the separation has grown `margin` above the smallest seen,
  default 10 m). Both are needed: after a capture the separation never grows, after
  a miss it never gets inside the radius. A pass only counts once the interceptor
  has closed in at least once: taking off from rest behind a fleeing target, the
  separation grows for a few seconds first (regression test).
- **Capture radius 0.5 m** (chosen; the plan said "e.g. 1 m"): about the
  motor-to-motor diagonal of the Iris (0.49 m). With a capture the run stops at the
  first sample inside, so the reported distance says "it got there", not how close.
- **`pursuit_feasible = speed > target speed`** (target speed at t = 0): pure
  pursuit cannot catch a target at least as fast. It only looks at the commanded
  speed; whether the airframe can fly it is not checked (loose end 14).
- **The MVP verdict comes from two engagements** (`examples/iris_pursuit.py`):
  slow threat (8 m/s, turning 2 deg/s; Iris ordered to 16 m/s) captured at 0.49 m,
  T+21.12 s; fast threat (25 m/s straight; Iris ordered to 30 m/s) missed by
  10.83 m at T+9.45 s. The fast case is kept on purpose: an unwinnable engagement
  is reported as such. Its 22.0 m/s peak is a transient above the 20.7 m/s steady
  top speed (velocity integral charged), not the top speed of the model.
- **Figures are committed** in `docs/` (the README shows them); examples write them
  there and create the folder if needed. Regenerating them is one command.
- **Interactive viewer** = data export + standalone page. `iris_pursuit_viewer.py`
  runs the engagements and writes their data (25 Hz, every 4th sample) as JSON
  inside `viz/viewer_template.html`, producing one self-contained file: a page
  opened from disk (`file://`) cannot fetch a separate data file. three.js 0.160.0
  from jsDelivr, pinned. The viewer interpolates positions linearly and attitudes
  with slerp, enlarges the vehicle icons with distance (tracks and numbers stay
  true), and has two palettes (cyan tactical, amber CRT) defined only as CSS
  tokens; the 3D materials register the token they use and are recoloured live. It
  complements Cesium (phase 12): one engagement in detail versus many agents over
  real terrain.
- **Lesson from the viewer:** an author rule `display: grid` beats the browser's
  `[hidden] { display: none }`, so a "hidden" full-screen error box was invisible
  but swallowed every click in the local file. The artifact host adds its own
  `[hidden]` rule, which masked the bug there. The page now carries
  `[hidden] { display: none !important }`. Test a page the way the user opens it.
- **The viewer runs in a desktop window** (`viz/app.py`, pywebview): a native window
  with the system web engine (WebView2 on Windows) instead of a browser tab. The page
  is the same; only the window changes, as in many desktop apps built on web
  technology. Chosen over Electron/Tauri (a second toolchain only for a window), a
  Python-native rewrite (PyVista/Qt: redo everything, less visual freedom) and Godot
  (a separate engine; still an option for a real-time demo). pywebview is optional:
  without it the page opens in the browser. Tested here with a stand-in module, not
  with pywebview itself (Claude's workspace cannot install it).
- **Shared viewer export** (`viz/viewer.py`: `scenario_data`, `write_viewer`), so
  every example writes the viewer the same way. `viz/` is a small Python package
  outside `jineta`: it reads results, it never runs physics. The refactor of
  `iris_pursuit_viewer.py` onto it produced a byte-identical page.
- **Scramble = launch on alert, with no new physics** (`examples/iris_scramble.py`):
  - No ground model, so a vehicle cannot rest on the ground inside the simulation.
    Before the launch the Iris is simply not integrated: it stays at rest on the
    pad (z = 0), which is what a vehicle with its motors off on the ground does.
  - The alert is checked once per control step: launch at the first `t = k·dt` at
    which the threat is within 400 m of the pad (slant range, as a radar measures).
  - `engage` starts its own clock at 0, so it is given the threat as seen from the
    launch, `t -> target(t + t_launch)`, and its times are shifted back. The standby
    samples and the pursuit are joined into one `Engagement` on one time axis.
  - Instant launch: no reaction time, no motor spin-up. The Iris rises from the
    first step (0.35 mm after 10 ms) and never goes below the pad.
  - The run ends at the capture or when the threat leaves the dome (its exit time
    is passed to `engage` as the duration, with `margin = inf` so a pass never
    stops it): a missed threat is chased for as long as the pad cares about it.
  - Results: A (8 m/s threat, gentle left turn, 110 m) launch T+13.14 s, capture
    0.48 m at T+36.87 s; B (25 m/s straight, 120 m) launch T+12.87 s, miss 31.11 m
    at T+23.00 s, then a stern chase until the threat leaves the dome at
    T+41.91 s: the Iris ends at 20.7 m/s (its level top speed) and 164 m behind.
- **The top speed depends on the flight path.** In case B the Iris reached
  28.6 m/s (26.0 horizontal, 11.9 climbing), well above the 20.7 m/s level-flight
  top speed. In level flight at the 45° tilt limit the thrust is only what holds the
  altitude (mg/cos 45° = 21.1 N); in a climb the velocity loop (vertical first) asks
  for up to 90 % of the maximum (25.5 N), and the same tilt then gives more
  horizontal force. Part of the peak is transient (velocity integral charged). So
  "top speed" is not one number for a multirotor (see loose end 14).
- **Viewer support for launch on alert** (optional `launch` block in the data:
  time, range, site): alert dome (latitude rings and meridians), STANDBY status
  before the launch, launch in the event log and on the range plot. Without the
  block the viewer is unchanged. Fixed on the way: removing a non-fat-line object
  called `splice(indexOf(...) = −1, 1)`, which would have dropped the last fat-line
  material from the resize list.

Phase 6b (fixed-wing threat):
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
  except F_y, 0.0015 N off (3e-5), not explained (loose end 28). The source's case 2
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
  autopilot clips, loose end 27); bank ≤ 35° (at 18 m/s and 45° the elevator
  saturated and a 90° turn lost 12 m); pitch ≤ 20°; course loop 8 and altitude loop
  15 times slower than their inner loops (12 left a slow oscillation after a 20 m
  step); airspeed 0.8 rad/s; sideslip 2 rad/s; anti-windup by conditional
  integration. 100 Hz: at 50 Hz the roll loop (11–19 rad/s) degrades.
- **Orbit guidance**: the vector field of Beard & McLain ch. 10,
  χ = φ + λ(π/2 + atan(k(d − R)/R)), gain 2. The autopilot holds a 150 m orbit at
  18 m/s within 0.5 m, at the bank of a steady turn, atan(V²/gR).
- **Step 7 made concrete with the author** (during step 4): the example mirrors the
  phase 7 scramble (Iris at rest on the pad, launch on alert, pure pursuit), one
  case captured and one missed; the viewer app opens on a start menu to choose the
  scenario page, named by the type of threat ("Generic target", "Fixed wing"), not
  by phase; the menu is a separate page (`docs/index.html`) rather than one page
  with every case, so each example still writes only its own viewer.
- **The threat is recorded, then replayed** (`RecordedTrack`) instead of integrating
  both bodies in one run: the fixed wing does not react to the Iris, so its flight
  does not depend on the pursuit, and `engage` and the phase 7 scramble stay
  unchanged. Replaying a recorded turn gives the same engagement as the turn itself
  to 1e-9. A threat that reacts (evasion) will need one loop for both (loose end 19,
  phase 8).
- **`iris_scramble.py` is importable** (`main()`): `fixed_wing_scramble.py` imports
  `scramble`, `report` and the constants, so the scenario logic is written once. The
  refactor gave a byte-identical viewer page.
- **Scenarios of the example**: A, the fixed wing flies in from the North-West and
  loiters on a 150 m clockwise orbit around (60, 40) m at 18 m/s and 100 m (slower
  than the Iris's ~20.7 m/s); B, a straight pass at 30 m/s and 120 m from the
  South-West (the course of the phase 7 case B). The Iris is ordered to 25 m/s, more
  than it can fly, so it flies flat out.
- **Viewer**: the fixed wing is drawn as an aircraft (wing, tailplane, fin) with its
  real attitude, so its bank shows (12.7° on the orbit, also in the telemetry); the
  scripted target is still a dart along its velocity. "Menu" button and key M on the
  scenario pages; the MVP page has no menu link (it is not in the menu).
- **The pages were checked in headless Chromium** with three.js r160 taken from
  GitHub (the CDN is not reachable from Claude's workspace): menu → viewer → menu,
  both pages, both cases, no script errors. Not checked in pywebview (loose end 31).

Phase 8 (scenarios and several entities):
- **One loop for everyone: the states are stacked**, `[vehicle 1 (13) | vehicle 2
  (13) | point (3) | track (0) ...]`, and flown by the existing `run`. The entities do
  not push on each other, so each derivative only reads its own slice; RK4 works
  component by component, so a body flown with others is bit-for-bit the body flown
  alone (tested with an Iris and the fixed wing). No new integrator, no second loop.
- **Every control reads the same snapshot** of the world at the start of the step, a
  read-only view: the order of the entities does not matter and nobody sees where
  another will be after the step (simultaneous turns, as in a board game). A control
  that tries to write into it fails. The rules (launches) run before the controls,
  so a launched interceptor already steers in the step of its launch.
- **Waiting and destroyed entities have a zero derivative**: they stay exactly where
  they are (the waiting Iris is bit-for-bit at rest on its pad; no ground model is
  needed for that). This replaces the phase 7 trick of not integrating the Iris and
  joining the samples afterwards. A destroyed scripted track stops at the point
  where it was hit. `Flight` records positions and launch times at the end of `fly`,
  because both are forgotten at the next `reset`.
- **Cyclic pursuit as the analytic test of the loop**: N ideal pursuers on a regular
  polygon, each chasing the next; the side shrinks at v(1 − cos 2π/N) and they meet
  at a / (v(1 − cos 2π/N)). With the zero-order hold one step takes a side d to
  exactly √((d − h)² + h² + 2(d − h)h cos θ), h = v·dt, which the world reproduces to
  5e-12 — only possible if everyone reads the same snapshot. The continuous meeting
  time is matched within 0.4 %; the first-order estimate of the difference,
  h sin²θ / (2(1 − cos θ)) · ln(a/d), predicted the measured 0.12 / 0.23 / 0.35 m
  before the run.
- **The package is `jineta/arena/`** (planned as `jineta/scenario/`): next to the
  `scenarios/` data folder the two names differed by one letter and the author's
  files landed in the wrong one; a themed name of its own avoids it.
- **A pair is judged by a `Duel`** (the phase 7 `EndOfPursuit`, renamed: same capture
  and pass logic) and a `Referee` runs every duel after each step as the stop
  condition of `fly`. A capture destroys both (they collide); a capture after a pass
  still counts; the other pairs of a destroyed entity become "target lost" /
  "interceptor lost"; captures in one step are judged in the order of the pairings.
  The run ends when every pair is decided. The closest approach of a pair is taken up
  to its end (capture, loss, escape).
- **`engage` is a world with two entities and one pair**, bit-for-bit the earlier one
  in five cases (capture, pass, infeasible, open, infinite margin) and in every
  example; its tests were the regression. It moved to `jineta/arena/` (loose end 18).
- **Launch on alert is a rule of the world** (`LaunchOnAlert`): the first step a
  watched threat is inside the dome. Against the earlier offline method (wait, then
  `engage` against the threat seen from the launch time) it matched bit for bit in
  practice; the test allows 1e-6 because the two accumulate the step times
  differently. A test that put the threat exactly on the edge of the dome at a sample
  time failed by rounding: tests keep crossings away from the sample grid.
- **The dome also ends a pair** (`escaped`): the target, after being inside, is
  outside again. It reproduces the phase 7 "the run ends when the threat leaves the
  dome".
- **A threat in the same loop gives the recorded engagement** (loose end 19): the
  guidance only reads the target at the start of each step, where the recording has
  its samples, so flying the fixed wing next to the Iris gives the replayed result
  bit for bit (the test allows 1e-6). And a threat can now react: an evader that
  flees along the line of sight once the pursuer is within 50 m makes the problem
  one-dimensional, the gap shrinks by exactly v_p·dt and then (v_p − v_e)·dt per
  step, and the world reproduces it to 1e-9.
- **Scenario files are strict**: `yaml.safe_load` (plain data, never objects), every
  key read or rejected, unknown names answered with the closest valid one
  (`difflib`), booleans refused where a number is expected (YAML reads `yes` as true
  and Python's `True` is an int). The files are written by hand, so a typo must stop
  the run and say where. 24 typical mistakes are tested.
- **Positions in the files are `[north, east]` plus `altitude`**, as on a map; the
  loader turns them into NED. Degrees only in the files, in keys that say so.
- **One atmosphere and one uniform wind per scenario.** In phase 6b the Iris flew at
  sea-level density while the fixed wing flew in ISA — an inconsistency nobody had
  seen. With one environment per file the fixed-wing scramble moved: A 0.48 m at
  T+62.14 s (was 0.49 m at T+63.32 s), B 49.80 m (was 49.74 m); the launches and the
  exit from the dome did not move. The MVP and the generic scramble stay at sea level
  and keep their numbers exactly.
- **An engagement in ISA and wind** (loose end 16): chasing East, the Iris catches the
  target at 8.6 s in still air, 9.55 s into a 5 m/s headwind and 8.0 s with it
  behind, all within the ideal-pursuer band [T, T + 3 s]. The velocity loop holds the
  ground velocity whatever the wind; the wind only shows while it accelerates.
- **An interceptor on a pad launches when its own target comes within the alert
  range.** Step 4 had it launch on any threat; with the assignment fixed in the file
  an Iris would take off for a threat it is not assigned to. With one threat both
  rules are the same. Choosing the assignment is phase 11.
- **A folder is a page, a file is a case** (`page.yaml`, `menu.yaml`):
  `examples/run_scenarios.py` flies them and writes every page and the menu; the
  examples only load their folder, report and plot (loose end 20: the MVP is no longer
  written twice). The tab id is the file name; the MVP subtitle lost its phase number
  ("generic target · in the air"). The menu cards come from the files, the cases and
  badges from the pages: nothing about a scenario is written by hand in `viz`.
- **Several entities in the viewer, without breaking the old pages**: the first pair
  is still exported as the viewer has always read it; a case with more entities adds
  `others`, `pairs` and `domes`, and the viewer makes one "view" per pair (the same
  tab, followed through another interceptor and its target). The other entities are
  drawn with their tracks and labels; a destroyed vehicle stays where it was hit and
  its speed reads 0; the range rings and the north arrow sit on the followed
  interceptor's pad. The single-pair pages carry the same data as before.
- **An interceptor that never launches is shown waiting until the end of the run**
  (its target never came within range). A test found that the export crashed on it.
- **pytest-xdist** (`pytest -n auto`) instead of marking slow tests (loose end 32):
  no test changes, one package more.
- **Two environment lessons**: `requirements.txt` was saved with a BOM by
  `Out-File -Encoding utf8` (now `Set-Content -Encoding ascii`), and a pasted block of
  PowerShell commands kept going after `pytest -n auto` failed on the other computer
  (pytest-xdist not installed there), up to the commit and push. The code was right,
  but since then the checks and the commit come as two separate blocks.
- In Claude's workspace the MVP figure came out in another font when the script runs
  with `python -m` (the workspace's matplotlib picks Inter); the same happened with
  the old code, so the figure differs only in the font, not in its data.
- For the browser checks Claude's workspace cannot reach the CDN, raw GitHub files
  or the npm registry; a sparse `git clone` of three.js r160 does work.
- **Every pair at once, the selection only for detail** (asked by the author after
  trying the app): the pair board lists every pair with its live range, status and
  closest approach; the scene shows every pair's closest-approach marker (once it has
  happened) and line of sight (while it is active); the range plot draws every
  pair's curve, the followed one bright; the event log merges every pair's events.
  Choosing a pair moves the camera (Follow) and the right-hand telemetry to it. The
  status rule is one function for the chip and the board.
- **The start menu reads like the index of a test campaign**, not like a game
  screen: no scanlines or corner ticks, sentence case, IBM Plex Sans for the text
  (Plex Mono only for the numbers), the same colour tokens and palettes as the
  viewer. One block per page: what it is and an outcome strip on the left (one
  segment per pair, coloured by how it ended: captured, missed or escaped, target
  lost, still open), the cases on the right with every pair's result and closest
  approach, and a line at the top that counts them all (6 of 11 pairs captured).
  Checked at desktop and phone width (no sideways scroll) in both palettes.

Phase 9 (sensors and tracking):
- **The Kalman filter is a set of plain functions** (`atalaya/kalman.py`): `predict`
  and `update` take matrices and return new arrays, and know nothing about targets
  or sensors; the trackers built on them (step 3) hold the state and the time.
- **Covariance update in Joseph form**, `(I − KH) P (I − KH)ᵀ + K R Kᵀ`: the same as
  `(I − KH) P` with the optimal gain, but always symmetric and positive
  semi-definite, also with rounding errors. The gain is solved with
  `np.linalg.solve`, not by inverting S.
- **Continuous white-noise acceleration for tracking** (`white_noise_acceleration`,
  spectral density q [m²/s³]): predicting an interval in one go or in pieces gives
  exactly the same covariance, so a track predicted at the simulation step between
  radar measurements does not depend on the step. The piecewise model (one random
  acceleration per interval) does depend on it; it is kept because the alpha-beta
  filter has its closed-form steady state (Kalata) with it, which tests the filter.
- **Consistency is tested with NEES**, `eᵀ P⁻¹ e`: over N seeded runs its average is
  a χ² with 6N degrees of freedom divided by N when the filter's model matches the
  truth (5.67–6.05 against the 99 % interval 5.50–6.53 over 300 runs); an
  overconfident filter gives 49–304. Every random test uses a seeded
  `numpy.random.default_rng`, so it gives the same numbers on every computer.

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
  3x3, `force_moment` model) with the 13-element `derivative()`: body force rotated to NED
  with `R(q)` + gravity, plus quaternion kinematics and Euler's equations. Also
  `ForceMomentModel`, `zero_force_moment`, `normalize_attitude` (post-step hook).
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
  - `Multirotor(params, density=sea_level_density, wind=no_wind)`: holds
    `rotor_speed_command` (starts at zero) and a `RigidBody`; `derivative(t, state)`
    and `force_moment(t, state)` (thrust + reaction torques via the mixer, rotor
    drag, body drag; all in body axes).
- `jineta/jinete/control.py` —
  - `RateGains`, `RateController(params, gains, dt)`: `update(rate_command,
    thrust_command, state) -> rotor speeds`, `reset()`.
  - `AttitudeGains`, `AttitudeController(gains)`: `update(attitude_command, state)
    -> rate command`.
  - `VelocityGains` (kp, ki, integral_limit, max_tilt, thrust_range),
    `VelocityController(params, gains, dt)`: `update(velocity_command, yaw_command,
    state) -> (attitude command, total thrust)`, `max_thrust`, `reset()`.
  - `MultirotorGains(rate, attitude, velocity)`.
  - `Autopilot(vehicle, gains, dt)`: callable `pre_step` hook with
    `velocity_command` and `yaw_command`; runs the three loops and writes the
    vehicle's `rotor_speed_command`.
- `jineta/jinete/profiles.py` — `flu_to_frd(vector)`, `IRIS` (PX4 SITL Iris, sources
  in the comments), `IRIS_RATE_GAINS`, `IRIS_ATTITUDE_GAINS`, `IRIS_VELOCITY_GAINS`,
  `IRIS_GAINS`, `SMALL_FIXED_WING` (generic small fixed-wing UAV, Aerosonde-class
  open data), `SMALL_FIXED_WING_PROPULSION` (its motor and propeller),
  `SMALL_FIXED_WING_AUTOPILOT` (design choices of its autopilot). The profiles by
  the names the scenario files use: `MULTIROTOR_PROFILES` (`"iris"` →
  `MultirotorProfile(params, gains)`) and `FIXED_WING_PROFILES`
  (`"small_fixed_wing"` → `FixedWingProfile(params, propulsion, autopilot)`).
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
- `jineta/core/events.py` — `ClosestApproach(time, distance, step)` and
  `closest_approach(times, relative_positions)`: smallest separation with the time
  interpolated inside its step (validated input: n ≥ 2 samples, (n, 3) array).
- `jineta/jinete/targets.py` — `TargetTrack` alias, `StraightLineTarget(start,
  velocity)`, `TurningTarget(start, speed, heading, turn_rate)` (frozen, validated,
  arrays read-only), `RecordedTrack(times, positions, velocities)` (cubic Hermite
  replay of a recording; validated, read-only, no extrapolation),
  `track_positions(track, times)`.
- `jineta/jinete/guidance.py` — `pure_pursuit_velocity(pursuer_position,
  target_position, speed, min_distance=1e-6)`.
- `jineta/arena/world.py` — `VehicleEntity`, `PointEntity`, `TrackEntity`
  (`Entity` is any of the three), `Snapshot(world, time, state)` (`of`, `position`,
  `velocity`), `World(entities, rules)` (`entity`, `slice_of`, `initial_state`,
  `reset`, `launch`, `destroy`, `snapshot`, `derivative`, `pre_step`, `post_step`,
  `fly(dt, t_end, stop)`) and `Flight(world, times, states, ...)` (`state`,
  `positions`, `launched_at`). `Control` and `Rule` aliases.
- `jineta/arena/engagement.py` — outcome names (`CAPTURED`, `PASSED`, `ESCAPED`,
  `OPEN`, `TARGET_LOST`, `INTERCEPTOR_LOST`), `Pairing(interceptor, target,
  capture_radius=0.5, margin=10, dome=None)` (validated), `Duel` (the judge of one
  pair: `update`, `lose`, `watch_dome`, `decided`, `finished`), `Referee(world,
  duels)` (stop condition), `DuelResult(pairing, outcome, closest, end_time)`,
  `Battle(flight, results)`, `fight(world, pairings, dt, t_end)`,
  `pursuit(autopilot, target, speed)` (pure-pursuit control), `Engagement` and
  `engage(autopilot, target, initial_state, speed, t_end, capture_radius=0.5,
  margin=10.0)` (one interceptor, one scripted target).
- `jineta/arena/launch.py` — `Dome(site, radius)` (`contains`), `LaunchAt(interceptor,
  time)`, `LaunchOnAlert(interceptor, dome, threats)`.
- `jineta/arena/spec.py` — the scenario file format (docstring), `ScenarioError`, the
  frozen specs (`ScenarioSpec`, `EnvironmentSpec`, `WindSpec`, `SiteSpec`,
  `InterceptorSpec`, `ThreatSpec`, `OrbitSpec`), `Reader` (reads one mapping of a file
  and says where a mistake is), `suggest`, `parse_scenario(data, source)`,
  `load_scenario(path)`.
- `jineta/arena/scenario.py` — `Setup(spec, world, pairings)`, `at_rest(position)`,
  `environment(spec)` (density and wind models), `threat_entity`,
  `interceptor_entity`, `build(spec)` and `play(spec)` → `(Setup, Battle)`.
- `viz/viewer.py` — `keep_indices(n, every)`, `scenario_data(result, target, every=4,
  launch=None, target_kind="generic", target_attitudes=None, **info)` (one
  engagement, downsampled, NED; labels in `info`, e.g. `threat_label`),
  `write_viewer(scenarios, path, dt, capture_radius, subtitle=..., menu=None)` (fills
  the template, writes one self-contained HTML file) and `read_viewer(path)` (the
  data inside a page, its inverse).
- `viz/pages.py` — `PageSpec` and `load_page(folder)` (a `page.yaml`),
  `menu_folders(root)` and `menu_entries(root)` (from `menu.yaml`),
  `page_folders(root)`, `load_cases`, `Case(file, setup, battle)` (the first pair:
  `interceptor`, `threat`, `result`, `launched`, `launch_time`, `dome`,
  `target_track`, `engagement`; `destroyed_at`), `entity_data`, `pair_data`,
  `case_data(case, every=4)` and `write_page(page, cases, folder, menu=False)`.
- `viz/viewer_template.html` — the three.js engagement viewer; `write_viewer`
  replaces the `"__ENGAGEMENT_DATA__"` placeholder with the data. Draws a scripted
  target as a dart and a fixed wing as an aircraft with its attitude (and its bank
  in the telemetry); "Menu" button and key M when the page has a menu. With several
  entities: every vehicle with its track and label, every alert dome, and a pair
  board with every pair's range, status and closest approach (its marker and line of
  sight in the scene, its curve on the range plot, its events in the log); clicking a
  row (or key N) moves the camera and the detailed telemetry to that pair.
- `viz/menu.py` — `OUTCOMES` (colour and words of each outcome), `MenuEntry(kind,
  description, page, command)`, `MenuPair(label, outcome, distance, time)`,
  `MenuCase(name, detail, pairs)` (`captured`), `page_cases(path)` (read from the
  page), `case_rows`, `card` (one page's block: outcome strip, cases table),
  `tally`, `write_menu(folder, entries)` (writes `index.html` from
  `viz/menu_template.html`; a missing page shows the command that writes it).
- `viz/app.py` — `show(path)` (desktop window or browser), `MENU` (the start menu,
  opened by `python -m viz.app` with no page), `window_requested`, `main`.
- Tests (463 passing, ruff check + ruff format clean):
  - integrators (2); rotation (5) — as in phases 1–3.
  - rigid body (10): the 8 of phase 4 + rejects mass ≤ 0 (2 cases).
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
  - aero (19): body airspeed for level / yawed 90° / pitched 30° (angle of attack =
    pitch) / rolled 90° flight (4 cases); flying with the wind feels no air; hovering
    in a headwind; rotation keeps the airspeed magnitude (20 random attitudes,
    seeded); the wind model receives time and position. Phase 6b: α and β for known
    attitudes (level, nose up 30°, nose ±10° off the flight path, knife edge — 5
    cases); round trip V, α, β → u, v, w → V, α, β (50 random, seeded); tail first is
    α = 180°; no airspeed gives zero angles; rejects wrong sizes (2) and negative
    speed.
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
  - control (10): P-only rate step is the exact discrete first order (3 axes);
    gyroscopic coupling cancelled; P only leaves the steady error `d/(I·kp)` under
    a disturbance moment; the integral removes it (critically damped PI); integral
    clamp and reset; rejects invalid gains (2 cases) and a non-positive period.
  - quaternion (11): conjugate is the inverse; rotation vector round trip (3
    angles); short way (350° → −10°); q and −q give the same vector; tiny angle;
    `from_matrix` round trip (100 random, seeded) and half turns (3 axes). Phase
    6b (13): Euler angles round trip (100 random, seeded); nose up, right wing down
    and heading East point the body axes where they should.
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
  - targets (20): straight line; a turn keeps speed and altitude; a turn is a circle
    of radius V/|ω| with the centre on the correct side (right and left — 2 cases);
    a full turn comes back to the start; velocity = derivative of the position
    (central differences); ω = 0 is a straight line; rejects invalid targets
    (3 cases). Recorded tracks: the samples exactly; exact for a cubic path; replays
    a turn recorded at 10 Hz within 1e-5 m; velocity = derivative of the position;
    no extrapolation; rejects invalid input (4 cases); read-only.
  - events (9): straight fly-by exact between samples (1e-12); interpolation beats
    the nearest sample; closest at the ends (moving apart, still closing — 2 cases);
    0.5 m pass on a curved path found within 0.1 mm and 10 µs while the nearest
    sample is > 0.1 m off; no relative motion; rejects invalid input (3 cases).
  - guidance (16): command points at the target with the given speed (3-4-5);
    speed independent of the distance (3 cases); zero command when the positions
    coincide; rejects invalid input (3 cases); pursuit-curve invariant constant
    (3 geometries, one with vertical motion); capture time matches the closed form
    (3 geometries); a faster crossing target is never caught and the minimum
    separation matches the closed form (~87.7 m); a faster receding target is never
    closer than at the start.
  - engagement (20): the judge of one pair: a capture is inside the radius; a pass is
    growing by the margin over the smallest; growing before closing in is not a
    pass; a capture after a pass still counts and a capture is final; an infinite
    margin never passes; rejects invalid pairings (4 cases). `engage` with the Iris:
    catches a slower target in 3D later than the ideal pursuer but within 3 s;
    catches a fleeing target it starts behind (regression); a faster target is
    reported infeasible and never approached; a pass ends the run after the closest
    approach; the same autopilot gives identical results twice; negative speed
    rejected; a recorded target gives the same engagement to 1e-9. Several pairs: two
    Iris against two targets in one world fly bit-for-bit as alone until their
    capture, then stay where they collided; two interceptors on one target, the
    second loses it; the fight ends when every pair is decided (capture + pass);
    rejects unknown names and no pairings.
  - world (14): bodies flown together are bit-for-bit the same as alone (Iris and
    fixed wing); a track is where its track says and a point flies its command;
    cyclic pursuit of 3, 4 and 6 points follows the exact discrete law (1e-9, still a
    regular polygon around the same centre) and meets at the closed-form time (1 %);
    a waiting body does not move until a rule launches it, then falls on the exact
    parabola; flying again starts from the same status; controls cannot change the
    snapshot; the snapshot reads every kind of entity; rejects invalid worlds; a
    destroyed entity stays where it was hit until the next flight.
  - launch (9): the dome contains its surface and rejects bad input; launch at a
    time; launch on alert at the first step the threat is inside (5.16 s) and the
    Iris exactly still until then; it watches several threats and ignores destroyed
    ones; a destroyed entity cannot be launched; the pair ends when the threat leaves
    the dome; the rule in the world gives the earlier offline wait + `engage` (1e-6);
    the fixed wing in the same loop gives the recorded engagement (1e-6); a threat
    that flees from its pursuer follows the exact 1-D law (1e-9) and is caught at the
    closed-form time (one step).
  - scenario files (27): every kind of entry read in SI and NED; the defaults of a
    minimal file; 24 mistakes, each with its place and, where it fits, a suggestion;
    load from a file, reject broken YAML and a top level that is not a mapping.
  - scenario building (3): the generic-scramble file flies bit-for-bit the world built
    by hand; each entry becomes its entity, rule and pair; an engagement in ISA and
    wind (headwind later than still air, later than tailwind, all within [T, T + 3]).
  - scenario pages (7): every scenario file of the repo loads and the menu shows the
    four pages in order; a folder becomes a page with one tab per case (launch time,
    fixed-wing label and attitude); page mistakes say where; the cases of a page
    share one step; the menu list says which folders and in what order (errors with
    suggestions); a case with several entities exports them all (`others`, `pairs`,
    `domes`, an interceptor that never launched); the menu lists every pair of a case
    with its outcome and a strip segment for each.
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
  - viz app (5): opens a window with the page URL and title, then starts it (with a
    stand-in for pywebview); falls back to the browser without pywebview; a missing
    page is an error; the `--no-window` flag; with no page it opens the start menu.
  - viz viewer and menu (7): a scripted target has no attitude; a fixed-wing target
    carries its attitude, downsampled; rejects an unknown kind or wrong attitudes;
    the page carries the subtitle and the menu link (none by default); reading a page
    gives back what was written; the menu's cases, results and closest approaches
    come from the page, with the strip note and the count at the top; it links the
    pages that exist and names the command for the others.
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
  - `run_scenarios.py` — flies every folder of `scenarios/` (or the ones given),
    prints each case's verdict (and every pair of a case with several), writes the
    viewer pages and the start menu, opens the app on the menu (`--no-window`: only
    the files). Its `play_folder`, `report` and `write_folder` are what the other
    examples use. ~50 s for the four pages in Claude's workspace.
  - `iris_pursuit.py` — the MVP (`scenarios/pursuit`): slow and fast threat, verdicts,
    3D intercept plus separation vs time (log scale); saves `docs/iris_pursuit.png`
    (the README figure).
  - `iris_pursuit_viewer.py` — the same folder into the viewer
    (`docs/iris_pursuit_viewer.html`, ~160 KB) and the menu, where it now has its
    card. Orbit / zoom / pan, timeline with range sparkline, 0.25–4× playback,
    overview / top / follow cameras, telemetry, event log, line of sight and
    velocity vectors, cyan or amber palette (keys: space, ← →, O, T, F, P, 1–9, N, M).
  - `iris_scramble.py` — `scenarios/generic-scramble`: the Iris waits on the pad and
    launches when the threat comes within 400 m; case A captured, case B missed and
    chased until the threat leaves the dome. Prints the verdicts, plots range and
    altitude against time (`plot_scramble`), writes its page (~290 KB) and the menu,
    opens the app.
  - `fixed_wing_scramble.py` — `scenarios/fixed-wing-scramble`: the same against the
    simulated fixed wing in the same world, both in ISA air: A loitering at 18 m/s on
    a 150 m orbit (captured, 0.48 m at T+62.14 s), B a straight pass at 30 m/s
    (missed by 49.80 m). Page ~440 KB; ~14 s to run.
  - `scenarios/two-pads` has no example of its own: `run_scenarios` writes its page
    (`docs/two_pads_viewer.html`, ~1.1 MB).
- Physics note (intermediate-axis instability, torque-free): spin Ω about the
  intermediate axis grows any tiny perturbation at rate
  `λ = Ω·sqrt((I2−I1)(I3−I2)/(I1·I3))`. The example perturbation (0.01 rad/s on p
  and r) is an arbitrary stand-in for real imperfections, not a physical constant.
  The flip time grows only with the logarithm of the perturbation size
  (~ln(10)/λ ≈ 2 s per 10× smaller for I = diag(1,2,3), Ω = 2); it shrinks with
  faster spin and larger λ. Exactly zero perturbation never flips. Spin about the
  smallest or largest axis is stable.
- Loose ends from phases 4–5, resolved: style (ruff clean, now with stricter
  rules), `run` docstring documents `stop` and `post_step`, input validation of
  `PointMass`, `RigidBody` and `RotatingBody` (`core/inertia.py`), shared
  interfaces moved to `core/environment.py`, `pyproject.toml` added.
- Loose end resolved in phase 7: the first real `README.md` (MVP figure, what works,
  how to run it, conventions, roadmap); the final one comes in phase 13.
- Still open, on purpose (each has its moment):
  1. After `stop` fires, the last sample overshoots the event (e.g. ~v·dt below
     ground). Resolved for the miss distance (`closest_approach`); other events
     (ground impact) still read the last sample.
  2. The ISA functions are scalar-only (`if` on the altitude, `math`). Vectorise
     with `np.where` only if profiling shows a need (Monte Carlo, many agents).
  3. `RotatingBody` and `RigidBody` still duplicate Euler's equations (inertia
     validation is already shared). Revisit when a vehicle needs attitude-only runs.
  4. (Resolved in phase 7: README.)
  5. Mixer saturation is plain clipping. It already happens for an instant at every
     velocity step (one rotor at 0, another at the maximum). A smarter strategy
     (give up yaw before roll and pitch, as PX4 does) when it starts to matter.
  6. Motor dynamics (first-order lag, 12.5 ms up / 25 ms down on the Iris) not
     modelled. Add if the rate loop needs it (it would add one state per rotor).
  7. Body drag is the weakest number: generic PX4 ballistic coefficient, and a
     placeholder for the vertical axis. The resulting top speed (20.7 m/s at 45°)
     is plausible but not validated against any Iris measurement.
  8. Rotor effects left out: axial-inflow thrust loss, rolling moment, gyroscopic
     moments; rotor inertia (1–3 %) left out of the Iris inertia.
  9. Controllers run at the simulation rate (1/dt).
  10. An interceptor-class multirotor profile (higher thrust-to-weight), as data
      with open sources. The MVP fast-threat case is the evidence: ordered to
      30 m/s, the Iris never exceeded 22.0 m/s.
  11. No position loop: the velocity loop holds vertical speed, not altitude (the
      Iris climbs ~0.6 m during a 10 m/s cruise). Add one if a scenario needs to
      hold a point.
  12. The attitude loop treats the whole rotation vector with per-axis gains; PX4
      gives the tilt priority over yaw ("reduced attitude", `MC_YAW_WEIGHT`). Not
      needed so far (tilt recovers in ~0.6 s with heading errors present).
  13. With the slow PX4 horizontal integral, braking leaves a small velocity tail
      while the drag compensation unwinds (a few tenths of m/s for a few seconds).
      PX4 reduces this with feedforward from its trajectory generator.
  14. `pursuit_feasible` compares the commanded speed with the target speed at
      t = 0. It knows neither what the airframe can fly nor a target whose speed
      changes. Better: the achievable speed as a property of the vehicle model
      (from the tilt limit, the thrust and the drag), so `engage` can warn when
      `speed` exceeds it. It depends on the flight path (20.7 m/s level, 28.6 m/s
      peak seen while climbing), so it is a function, not one number.
  15. Yaw is never commanded during an engagement (`yaw_command` stays 0): the Iris
      flies the whole pursuit nose North. Harmless for the physics now; it matters
      in phase 9, when a sensor with a field of view has to point at the target.
  16. (Resolved in phase 8: scenarios set ISA or sea level and a uniform wind; an
      engagement in ISA and wind is tested.)
  17. A capture stops the run at the first sample inside the radius, so the
      reported distance is "inside", not the impact precision. Comparing strategies
      by precision would need running past the capture or a smaller radius.
  18. (Resolved in phase 8: `engagement.py` lives in `jineta/arena/`.)
  19. (Resolved in phase 8: every entity flies in one world; a reacting threat is
      tested.)
  20. (Resolved in phase 8: the scenarios are YAML files, written once.)
  21. The viewer needs internet (three.js and fonts from CDNs). For offline use,
      ship a pinned copy of three.js in the repo; a page opened from `file://`
      cannot import local ES modules, so it has to be inlined (or served by the
      app window's local server). three.js r160 can be taken from GitHub in Claude's
      workspace (done in phase 6b to test the pages), not from the CDN.
  22. (Moved into the world in phase 8: `LaunchOnAlert`.) Still instant (no reaction
      time, no spin-up) and with perfect detection: in phase 9 the alert becomes a
      sensor detection.
  23. No ground: once flying, nothing stops a vehicle from going below z = 0 (it
      does not happen in any current example). Terrain comes in phase 12; a flat
      ground check could come earlier if a scenario needs it.
  24. Packaging as `Jineta.exe` (PyInstaller + pywebview); the start menu is now a
      launcher worth packaging.
  25. The source's stall gives C_L_max ≈ 2.2 near α ≈ 22° (C_L_alpha = 5.61 and
      α₀ = 27°), high for a small UAV (typically 1.2–1.5). Irrelevant while the
      threat flies at α < 10°; revisit if it has to manoeuvre hard.
  26. The side force and the lateral moments use β in body axes, and drag has no
      side component: the small-β simplification of the source.
  27. Control-surface limits live only in the autopilot (±30°): the vehicle model
      accepts any deflection, and the trim at low speed asks for more (−22° of
      elevator at 18 m/s, −35° at 15 m/s). Move the limits into the vehicle when a
      second controller or a scripted manoeuvre could exceed them.
  28. F_y differs from the source's case 3 by 0.0015 N (3e-5) with every input it
      depends on matching to 1e-9. Not explained; the test tolerates it with a
      comment.
  29. The autopilot gains are designed at one trim (the operating speed) and not
      rescheduled with speed. Tested from 18 to 30 m/s; a threat that changes speed
      a lot would want gain scheduling.
  30. The course loop uses the airspeed in its gains (Beard & McLain use the ground
      speed): equal in still air, different in a strong wind.
  31. (Resolved at the start of phase 8: the author checked the menu → viewer →
      menu navigation in pywebview on his computer.)
  32. (Resolved in phase 8: `pytest -n auto` with pytest-xdist.)
  33. A threat with no interceptor assigned is drawn and flies, but nobody judges it
      (no pair): whether it got through is not reported. Phase 11 assigns, and the
      verdict of a scenario should then count the threats that reached their aim.
  34. After its pair ends (escaped, passed, target lost) an interceptor keeps flying
      its pursuit: it chases a target outside the dome, or the wreck of one another
      Iris destroyed. It should return to its pad or be re-tasked (phase 11).
  35. A destroyed vehicle freezes where it was hit (no falling wreck): enough for the
      verdicts; a falling wreck needs a ground (loose end 23).
  36. Captures in the same step are judged in the order of the pairings: if two
      interceptors reach one target in the same step, the second is "target lost".
      Rare at 100 Hz; documented, not tested.
  37. The page grows with the entities (`two_pads_viewer.html` ~1.1 MB: 25 Hz, ten
      numbers per vehicle per sample). For many agents: downsample harder or leave it
      to the Cesium export (phase 12).
  38. The console report and the scramble plots follow the first pair of a case; the
      viewer shows every pair. Per-pair plots if a study needs them.
  39. All the cases of one page share the step and the capture radius (the viewer
      has one timeline and one capture line); `write_page` refuses otherwise.
- `jineta/atalaya/kalman.py` (phase 9, step 1) — `predict`, `update` (returns the
  innovation and its covariance too), `nees`, `constant_velocity_transition`,
  `white_noise_acceleration`, `piecewise_white_acceleration`, `alpha_beta_gains`;
  20 tests in `tests/test_kalman.py`. The full list above is updated at the close
  of the phase.
- Now: phase 9, step 2 (ground radar). The step plan is in section 5.

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
- Claude's workspace cannot install packages from PyPI: it runs the tests with a
  minimal pytest stand-in (`parametrize`, `raises` with `match` and `as`, `approx`
  for scalars, `tmp_path`) and ruff. The author's `pytest -n auto` is the reference.
  Claude never commits to the repo (its access to GitHub is read-only): it hands
  over complete files, since phase 8 as a zip per step with the repo's folders,
  which the author keeps in `..\buffer` and expands over the repo
  (`Expand-Archive -Path "..\buffer\<zip>" -DestinationPath . -Force`). Files to
  delete or move are named with their `git rm` / `Move-Item` command. When a change
  touches several places of one file, Claude gives the whole file: partial
  insertions were the source of the errors in phase 6a (a lost import, a missing
  block). Each step ends with two blocks of commands: the checks, and then, once
  they passed, the commit (one-line message) and push.
- Claude checks every push against its own copy (same files, `ruff check`, the
  tests, the regenerated pages compared number by number) before going on. Its own
  working copy carries local commits that never leave the workspace.
- Claude's workspace cannot reach CDNs either (nor raw GitHub files or the npm
  registry). It can still render the viewer pages in headless Chromium (Playwright
  is installed) by serving three.js r160, taken with a sparse `git clone` of
  `mrdoob/three.js` at tag r160, in place of the CDN (clicks, keys, script errors,
  screenshots; fonts fall back). Wait for a page to load before pressing keys: the
  fixed-wing page takes a few seconds. When the chat is linked to the author's
  computer, Claude can also test a page in the desktop app's built-in browser. If a
  page could not be rendered at all, Claude says so.
- **End of each phase:** update this document (roadmap, conventions, decision log,
  current state) and hand it over to be committed.
- **Session hygiene:** Claude must tell the author when to start a new chat —
  at the end of a phase block, or when the conversation is getting long enough that
  older details risk being lost. Before switching, make sure this document is up to
  date. A new chat starts inside the "Jineta" Project with:
  "Seguimos con Jineta, fase N: <topic>".
- If a chat stops or breaks mid-task, open a new one in the Project: the repo and
  this document are the source of truth, not the chat history.
- **A phase plan is a contract.** When a phase spans several chats, the step list
  in section 5 (with its ✅ marks) is followed as written; changing it is a decision
  to discuss and record, not something a new chat does on its own.