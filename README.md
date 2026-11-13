# Jineta

A realistic simulator to **train drones for indoor flight**: a 6-DOF physics engine,
indoor environments and a training loop in which a drone learns its task with
reinforcement learning, so that what it learns can later be flown by the real
vehicle.

*Jineta* is a working name.

> **Status: phase 12 done — realism for the real drone.** The MVP (phase 11):
> the Iris learned the first course with reinforcement learning (PPO, written in
> NumPy in this project) in **3.23 s against 6.77 s** for a classical controller
> with the same limits. Phase 12 added what separates a simulator from a real
> room: motor lag, ground effect, a motion-capture lab's sensors with an onboard
> estimator, the latency of the commands, drones drawn anew every episode, and
> a stop under control after the finish; training now uses every core. The
> phase 11 policy turned out fragile under them (10 % of drawn drones finish);
> training policies that cope is the next measurement. See
> [`DESIGN.md`](DESIGN.md) for the roadmap and the decision log (911 tests).

## The goal

**MVP — a drone that learns a course.** A quadrotor (the Iris profile from PX4's
simulator) learns to fly an indoor obstacle course in the shortest possible time:
cross a sequence of gates in order without touching a wall, an obstacle or the
floor. Its lap time is compared with a classical controller flying the same
course. The policy commands what a real PX4 autopilot accepts from a companion
computer — collective thrust and body rates — so it is meant to move to a real
drone, not only to live in the simulator.

**Final goal — your environment, your drone.** Any user describes the environment
(a racing course, a warehouse, an inspection route) and the drone profile (mass,
inertia, rotors, limits) they need, and trains a drone for their purpose.

## What works today

| Part | What it does |
|---|---|
| Rigid body | 13-state 6-DOF dynamics (NED position and velocity, quaternion attitude, body rates), fixed-step RK4 |
| Environment | ISA atmosphere 0–20 km; wind models (uniform, profile, boundary layer, gusts) |
| Quadrotor | Iris profile from PX4 SITL: rotors, mixer, rotor and body drag, density scaling |
| Quadrotor autopilot | Cascaded velocity → attitude → rate control at 100 Hz, PX4-like gains |
| Fixed wing | Aerodynamic coefficients, electric propeller, trim, dynamic modes, autopilot |
| Estimation | Linear Kalman filter (constant-velocity model, consistency by NEES) |
| Indoor world | Room, boxes and pillars by signed distance; rectangular and round gates |
| Ground and take-off | Landing gear on the floor (spring-damper with friction); the autopilot's landed state, take-off and landing detection |
| Collisions | The drone as spheres (13 for the Iris) against walls, ceiling, obstacles and gate frames, at every sample and along the moves between samples (no graze or thin wall slips through); the gear against the floor after take-off |
| Courses | Strict YAML course files; a referee for gates in order, lap time, crash or time limit |
| Batch engine | Many drones of one profile on one course at once (NumPy arrays; each drone may have its own numbers): dynamics, ground contact, the whole autopilot, collisions and referee, each drone on its own lap clock and reset on its own; equal to the reference engine to rounding, ~56× its speed at 1 024 drones (~580× real time on one core) |
| Training environment | n drones learning a course at once, with the shape of Gymnasium's interface (`reset`, `step`): actions are collective thrust and body rates at 50 Hz through the rate loop (as PX4 accepts them), observations in body axes (velocity, attitude, rates, the next two gates, the last action), reward for progress, gates and crashes; each drone's episode ends and restarts on its own |
| Classical baseline | The velocity autopilot following a planned path through the gates (pure pursuit, corner speed plan), its settings swept in the batch for the fastest robust lap: 6.77 s on the first course, the time a policy has to beat |
| Learning | PPO in NumPy (`jineta/doma`): neural networks with backpropagation by hand, a Gaussian policy, GAE; checked against the optimal regulator (LQR) of a double integrator. An adapter makes the environment a Gymnasium `Env` / `VectorEnv` |
| Realism | Each effect can be switched on: motor lag (rotor speeds in the state), ground effect, sensors (motion capture, IMU) and an onboard estimate, latency, drones drawn anew every episode (each equal to the reference engine with its own numbers), a stop under control after the finish; training spread over several processes |
| The learned pilot | The Iris flies the first course in 3.23 s (`policies/first_course.npz`): in half the baseline's time, never closer than 13 cm to anything, every one of its laps with exploration noise finishes. Random courses (gates and obstacles) to train on and to measure on; the policy can see the nearest walls and obstacles |

Every physics module is tested against known results: closed-form motion,
published reference values, exact discrete solutions of the control loops.

## Run it

Windows, Python 3.13, from the repo root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -n auto
python -m examples.iris_course
python -m examples.batch_course
python -m examples.env_random
python -m examples.baseline_course
python -m examples.policy_course
python -m examples.train_course
python -m examples.realism_policy
python -m examples.train_realism --workers 8
```

The examples in `examples/` show the engine at work (projectile with drag, the
intermediate-axis instability, a body in wind, the Iris accelerating to 10 m/s and
stopping) and the Iris flying the first course (`courses/first_course.yaml`): it
takes off, flies through the four gates and finishes in 8.81 s. `batch_course`
flies several Iris at different speeds at once and measures the batch engine's
speed against the number of drones; `benchmark` times the reference engine.
`env_random` flies the training environment with random actions (what the first
minutes of training look like) and measures its speed; `baseline_course` sweeps
the classical baseline's settings and draws its best lap of the first course.
`policy_course` flies the trained policy and draws its lap next to the
baseline's; `train_course` trains one from scratch (two stages, ~30 M steps:
about 45-75 minutes on one core), and `train_random` trains on random courses and
compares the result with the baseline on courses it never saw. `realism_policy`
flies a policy with each effect of a real lab; `train_realism` trains in the
randomised lab (`--workers` spreads any training over several cores;
`parallel_speed` measures it).

## Layout

```
jineta/core/      physics engine: integrators, rigid body, quaternions, run loop
jineta/campo/     environment: atmosphere, wind, indoor world (room, obstacles,
                  gates, collisions, referee, course files)
jineta/jinete/    vehicles: multirotor (motor lag, ground effect), fixed wing,
                  their autopilots, profiles
jineta/atalaya/   estimation: Kalman filter, sensors and the onboard estimate
jineta/escuadron/ batch engine: many drones at once, checked against the above
jineta/doma/      learning: neural network, Gaussian policy, PPO (NumPy)
jineta/picadero/  training layer: environment (action, observation, reward),
                  classical baseline, Gymnasium adapter, training and
                  evaluation, random courses, realism and randomisation,
                  stop after the finish, several processes
policies/         trained policies (.npz)
tests/            tests against analytic and published results
examples/         runnable scripts
courses/          course files (YAML)
```

## Conventions

SI units, NED frame (x North, y East, z Down), quaternions `[w, x, y, z]` from body
to NED, body axes forward-right-down. Vehicles are data (one frozen profile per
vehicle, every number with its source). Details in [`DESIGN.md`](DESIGN.md).

## Roadmap

| Phase | Content |
|---|---|
| 1–6c | Physics engine ✅ |
| 7 | Project goal, new README and design ✅ |
| 8 | Indoor world: room, gates, obstacles, collisions; course files ✅ |
| 9 | Fast batch engine (many drones at once) ✅ |
| 10 | Training environment + classical baseline ✅ |
| 11 | **MVP:** the Iris learns the course ✅ |
| 12 | Realism for the real drone ✅ |
| 13 | 3D viewer of recorded flights |
| 14 | Towards the real drone (PX4) |
| 15 | Environment and profile builder |
| 16 | Final README and demo |

## License

MIT — see [`LICENSE`](LICENSE).
