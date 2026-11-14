// What the viewer shows at each moment: the playback of recorded laps
// (phase 13). Plain JavaScript with no three.js and no page: it is tested with
// node (tests/web/playback.test.mjs).
//
// The player holds the recordings in the run's order (stage after stage, then
// steps; `tribuna.run`) and plays them one after the other, the way a film of
// the training would: every lap of the current recording (on the course being
// shown) flies at once, on its own lap clock, and once the last of them has
// ended and a short pause has passed, the next recording starts. When there is
// no next one yet, the last recording is played again until a new one arrives
// (a training still running adds them), so the window always shows the latest
// flying.
//
// The player works on summaries (`tribuna.server.summary`: how each lap ended
// and how long it lasts, and each course's first lap as a trail); the whole
// recording (every sample of every lap) is only needed to say where the drones
// are (`frame`), and the page asks the server for it when it is played.
//
// Times: `t` is the lap clock of the recording being played [s]; `advance(dt)`
// moves it by dt (the wall time since the last frame) times `speed`.
//
// Seeing a policy learn: the laps of the recordings before the current one
// stay drawn as trails, fading with their age, coloured by how they ended. A
// trail is the first lap of its recording on the course (the mean action's,
// the policy as it would fly): trajectories are never averaged (the average of
// a path into the pillar and one past it is a path nobody flew).

export const SPEEDS = [0.125, 0.25, 0.5, 1, 2, 4, 8]; // playback speeds offered
export const PAUSE = 1.0; // [s] of wall time between a recording's end and the next
export const TRAILS = 10; // earlier recordings drawn as trails
export const NEWEST_OPACITY = 0.8; // the trail of the recording just before
export const OLDEST_OPACITY = 0.1; // the trail TRAILS recordings back

// ---- between the samples ----------------------------------------------------
//
// A lap keeps a sample every environment step (0.02 s); the page draws many
// more frames than that, and plays laps slower or faster. Between two samples
// the position is interpolated along a straight line and the attitude by
// spherical linear interpolation (slerp): the rotation from one attitude to the
// next is cut in the same fraction, about the same axis. Interpolating the four
// numbers of the quaternions linearly would not stay a rotation (its length
// shrinks half-way); slerp moves along the unit sphere of quaternions at a
// steady rate. q and -q are the same attitude, so the shorter way round is
// taken.

function normalised(q) {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  return [q[0] / n, q[1] / n, q[2] / n, q[3] / n];
}

/** The attitude a fraction of the way from q0 to q1 (unit quaternions
 * [w, x, y, z]), the shorter way round; normalised. */
export function slerp(q0, q1, fraction) {
  const a = normalised(q0);
  let b = normalised(q1);
  let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  if (dot < 0) {
    // -q1 is the same attitude, nearer
    b = b.map((v) => -v);
    dot = -dot;
  }
  if (dot > 0.9995) {
    // nearly the same: a straight line is as good, and safe
    return normalised(a.map((v, k) => v + fraction * (b[k] - v)));
  }
  const angle = Math.acos(dot); // half the rotation between them
  const s0 = Math.sin((1 - fraction) * angle) / Math.sin(angle);
  const s1 = Math.sin(fraction * angle) / Math.sin(angle);
  return normalised(a.map((v, k) => s0 * v + s1 * b[k]));
}

/** An attitude [w, x, y, z] (BODY -> NED) in the order three.js takes a
 * quaternion, [x, y, z, w]; normalised (a recording rounds it). Set on the
 * drone's object, it turns the drone's body-axes points into the room's: a
 * point b of the body goes to R b, R the rotation BODY -> NED. */
export function threeQuaternion(q) {
  const [w, x, y, z] = normalised(q);
  return [x, y, z, w];
}

/** How many of the sorted `times` are <= t (binary search). */
export function countUpTo(times, t) {
  let lo = 0;
  let hi = times.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= t) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}

/** Where a recorded lap's drone is at lap clock t [s]: its position (NED [m]),
 * attitude (quaternion [w, x, y, z]) and how many samples of its path are
 * behind it (`before`: those earlier than t; all of them once it has ended).
 * Before the start, the first sample; after the end, the last (where the lap
 * ended). `lap` holds times, positions and attitudes, as a recording file. */
export function poseAt(lap, t) {
  const { times, positions, attitudes } = lap;
  const last = times.length - 1;
  if (t >= times[last]) {
    return {
      position: positions[last].slice(),
      attitude: normalised(attitudes[last]),
      before: times.length,
    };
  }
  if (t <= times[0]) {
    return {
      position: positions[0].slice(),
      attitude: normalised(attitudes[0]),
      before: 0,
    };
  }
  const j = countUpTo(times, t); // times[j - 1] <= t < times[j]
  const span = times[j] - times[j - 1];
  const fraction = span > 0 ? (t - times[j - 1]) / span : 1;
  const p0 = positions[j - 1];
  const p1 = positions[j];
  return {
    position: p0.map((v, k) => v + fraction * (p1[k] - v)),
    attitude: slerp(attitudes[j - 1], attitudes[j], fraction),
    before: times[j - 1] < t ? j : j - 1,
  };
}

// ---- the player ---------------------------------------------------------------

/** How visible the trail of the recording `age` places back is: from
 * NEWEST_OPACITY (the one just before) down to OLDEST_OPACITY (keep back), in
 * a straight line; 0 beyond. */
export function trailOpacity(age, keep = TRAILS) {
  if (age < 1 || age > keep) return 0;
  if (keep === 1) return NEWEST_OPACITY;
  const fraction = (age - 1) / (keep - 1);
  return NEWEST_OPACITY + fraction * (OLDEST_OPACITY - NEWEST_OPACITY);
}

/** The next playback speed up (step +1) or down (-1) the list, staying at its
 * ends. */
export function stepSpeed(speed, step) {
  let nearest = 0;
  SPEEDS.forEach((s, i) => {
    if (Math.abs(Math.log(speed / s)) < Math.abs(Math.log(speed / SPEEDS[nearest]))) {
      nearest = i;
    }
  });
  return SPEEDS[Math.min(Math.max(nearest + step, 0), SPEEDS.length - 1)];
}

function finitePoint(point) {
  return Array.isArray(point) && point.every((v) => Number.isFinite(v));
}

/** Plays recordings in order (see the top of the file). `entries` are the
 * recordings' summaries. */
export class Player {
  constructor(entries = [], pause = PAUSE) {
    this.entries = [...entries];
    this.index = 0;
    this.course = 0;
    this.t = 0;
    this.speed = 1;
    this.paused = false;
    this.pause = pause;
    this.waited = 0;
  }

  get current() {
    return this.entries.length ? this.entries[this.index] : null;
  }

  /** The current recording's laps on the course shown (summaries). */
  laps() {
    const entry = this.current;
    return entry ? entry.laps.filter((lap) => lap.course === this.course) : [];
  }

  /** How long the current recording plays: its longest lap on the course [s]
   * (0 when there is none). */
  duration() {
    return this.laps().reduce((longest, lap) => Math.max(longest, lap.duration), 0);
  }

  /** New recordings (a training still running), after the others. */
  add(entries) {
    this.entries.push(...entries);
  }

  /** Replace every recording (the run's order changed): the one being played
   * stays at the same number, clipped to those there are. */
  reset(entries) {
    this.entries = [...entries];
    this.index = Math.min(this.index, Math.max(this.entries.length - 1, 0));
    if (this.current && this.course >= this.current.courses.length) this.course = 0;
  }

  /** Move the playback on by dt seconds of wall time; nothing moves while the
   * current recording is not `ready` (its samples not yet arrived). */
  advance(dt, ready = true) {
    if (this.paused || !this.current || !ready) return;
    const end = this.duration();
    if (this.t < end) {
      this.t = Math.min(end, this.t + this.speed * dt);
      return;
    }
    this.waited += dt; // the pause after the end, in wall time
    if (this.waited >= this.pause) {
      if (this.index + 1 < this.entries.length) this.goTo(this.index + 1);
      else this.restart();
    }
  }

  // ---- what the keys do ----

  /** Play recording `index` (clipped to those there are) from its start; its
   * course is kept if it has one of that number. */
  goTo(index) {
    if (!this.entries.length) return;
    this.index = Math.min(Math.max(Math.round(index), 0), this.entries.length - 1);
    if (this.course >= this.current.courses.length) this.course = 0;
    this.restart();
  }

  restart() {
    this.t = 0;
    this.waited = 0;
  }

  next() {
    this.goTo(this.index + 1);
  }

  previous() {
    this.goTo(this.index - 1);
  }

  first() {
    this.goTo(0);
  }

  newest() {
    this.goTo(this.entries.length - 1);
  }

  /** Show the laps on the recording's next course (random courses have one
   * lap on each), from their start. */
  nextCourse() {
    if (!this.current) return;
    this.course = (this.course + 1) % this.current.courses.length;
    this.restart();
  }

  togglePause() {
    this.paused = !this.paused;
  }

  faster() {
    this.speed = stepSpeed(this.speed, +1);
  }

  slower() {
    this.speed = stepSpeed(this.speed, -1);
  }

  /** Jump to lap clock t (clipped to the recording). */
  seek(t) {
    this.t = Math.min(Math.max(t, 0), this.duration());
    this.waited = 0;
  }

  // ---- what the page draws ----

  /** The earlier recordings' first laps on the course shown (the most recent
   * first), up to `keep` recordings back: their path, how they ended, where
   * they touched something, how visible. */
  trails(keep = TRAILS) {
    const entry = this.current;
    if (!entry) return [];
    const id = entry.courses[this.course].id;
    const trails = [];
    for (let age = 1; age <= keep && this.index - age >= 0; age++) {
      const earlier = this.entries[this.index - age];
      const c = earlier.courses.findIndex((course) => course.id === id);
      if (c < 0 || !earlier.trails[c]) continue;
      const lap = earlier.laps.find((l) => l.course === c);
      const crashed = lap.status === "crashed" && finitePoint(lap.crash_point);
      trails.push({
        age,
        steps: earlier.steps,
        status: lap.status,
        points: earlier.trails[c],
        crashPoint: crashed ? lap.crash_point : null,
        opacity: trailOpacity(age, keep),
      });
    }
    return trails;
  }

  /** Where every drone of the current recording on the course shown is now.
   * `recording` is that recording whole (as its file holds it); the drones
   * are its laps on the course, in order, each with its pose, how many of its
   * samples are behind it, whether its lap has ended, how it ends and, once it
   * has touched something, where. */
  frame(recording) {
    const entry = this.current;
    if (!entry) return null;
    const drones = [];
    if (recording) {
      recording.laps.forEach((lap, k) => {
        if (lap.course !== this.course) return;
        const pose = poseAt(lap, this.t);
        const crashed = lap.status === "crashed" && finitePoint(lap.crash_point);
        const touched = crashed && this.t >= lap.crash_time;
        drones.push({
          lap: k,
          label: lap.label,
          status: lap.status,
          lapTime: lap.lap_time,
          position: pose.position,
          attitude: pose.attitude,
          before: pose.before,
          ended: this.t >= lap.times[lap.times.length - 1],
          crashPoint: touched ? lap.crash_point : null,
        });
      });
    }
    return {
      index: this.index,
      course: this.course,
      t: this.t,
      duration: this.duration(),
      drones,
    };
  }
}

// ---- words and numbers beside the scene ----------------------------------------

function thousands(n) {
  return Math.round(n)
    .toString()
    .replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

/** How a lap of the current recording reads: "..." while it flies, its lap
 * time once it has finished, or how it ended. */
export function result(drone) {
  if (!drone.ended) return "...";
  if (Number.isFinite(drone.lapTime)) return `${drone.lapTime.toFixed(2)} s`;
  return drone.status;
}

/** The lines in the corner of the window: which recording, the course, the
 * time. */
export function describe(player, total, stage = "") {
  const entry = player.current;
  if (!entry) return [];
  const iteration =
    entry.info && entry.info.iteration !== undefined
      ? ` (iteration ${entry.info.iteration})`
      : "";
  const where = stage ? `, stage ${stage}` : "";
  return [
    `recording ${player.index + 1} of ${total}: ${thousands(entry.steps)} steps${iteration}${where}`,
    `course ${player.course + 1} of ${entry.courses.length}: ${entry.courses[player.course].name}`,
    `t = ${player.t.toFixed(2)} s of ${player.duration().toFixed(2)} s, x${player.speed}` +
      (player.paused ? "  (paused)" : ""),
  ];
}

/** For the lap time chart: each recording with laps on course `course`, at its
 * place on the run's axis (`positions`), with its first lap's (the mean
 * action's) lap time (null unless it finished) and the share of its laps that
 * finished. */
export function progress(entries, positions, course = 0) {
  const points = [];
  entries.forEach((entry, i) => {
    const laps = entry.laps.filter((lap) => lap.course === course);
    if (!laps.length) return;
    const finished = laps.filter((lap) => lap.status === "finished").length;
    points.push({
      index: i,
      steps: positions[i] ?? entry.steps,
      firstLapTime: Number.isFinite(laps[0].lap_time) ? laps[0].lap_time : null,
      finished: finished / laps.length,
    });
  });
  return points;
}

/** Whether the first `known` keys are still the same (otherwise the run's
 * order changed and the page reads every summary again). */
export function samePrefix(mine, theirs) {
  return mine.every((key, i) => key === theirs[i]); // a key not there: undefined
}
