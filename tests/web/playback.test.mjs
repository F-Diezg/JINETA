// Tests for the viewer's playback (phase 13, tribuna/web/playback.js), run with
// node: `node --test tests/web/` (tests/test_tribuna_web.py runs them inside
// pytest when node is installed). The same cases the Python player had:
// recordings played in order, new ones added while playing, the keys, where
// every drone is at the player's moment, the trails.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  NEWEST_OPACITY,
  OLDEST_OPACITY,
  PAUSE,
  Player,
  SPEEDS,
  countUpTo,
  describe,
  poseAt,
  progress,
  result,
  samePrefix,
  slerp,
  stepSpeed,
  threeQuaternion,
  trailOpacity,
} from "../../tribuna/web/playback.js";
import { Quaternion, Vector3 } from "../../tribuna/web/vendor/three.module.js";

const close = (a, b, tol = 1e-12) =>
  assert.ok(Math.abs(a - b) <= tol, `${a} is not within ${tol} of ${b}`);
const allClose = (a, b, tol = 1e-12) => {
  assert.equal(a.length, b.length);
  a.forEach((v, k) => close(v, b[k], tol));
};

// ---- quaternions [w, x, y, z], BODY -> NED (as jineta.core.quaternion) ----

function fromEuler(roll, pitch, yaw) {
  const [cr, sr] = [Math.cos(roll / 2), Math.sin(roll / 2)];
  const [cp, sp] = [Math.cos(pitch / 2), Math.sin(pitch / 2)];
  const [cy, sy] = [Math.cos(yaw / 2), Math.sin(yaw / 2)];
  return [
    cr * cp * cy + sr * sp * sy,
    sr * cp * cy - cr * sp * sy,
    cr * sp * cy + sr * cp * sy,
    cr * cp * sy - sr * sp * cy,
  ];
}
const turned = (deg) => fromEuler(0, 0, (deg * Math.PI) / 180);
const heading = (q) =>
  (Math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)) * 180) /
  Math.PI;
const norm = (q) => Math.hypot(...q);
const conjugate = (q) => [q[0], -q[1], -q[2], -q[3]];
function multiply(a, b) {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}
function rotationVector(q) {
  const s = q[0] < 0 ? -1 : 1; // the shorter way
  const v = [s * q[1], s * q[2], s * q[3]];
  const n = Math.hypot(...v);
  const angle = 2 * Math.atan2(n, s * q[0]);
  return n > 0 ? v.map((x) => (x * angle) / n) : [0, 0, 0];
}

// ---- recordings as the server sends them ----

/** A straight lap North at 1 m/s, sampled every 0.02 s: the whole lap (as a
 * recording file holds it) and its summary (as /api/poll sends it). */
function flown(duration, { status = "timeout", label = "mean action", course = 0, ...change } = {}) {
  const n = Math.round(duration / 0.02) + 1;
  const times = Array.from({ length: n }, (_, k) => k * 0.02);
  const lap = {
    course,
    label,
    status,
    lap_time: status === "finished" ? duration : null,
    crash_time: null,
    crash_point: [null, null, null],
    times,
    positions: times.map((t) => [t, 5, -1]),
    attitudes: times.map(() => [1, 0, 0, 0]),
    ...change,
  };
  return lap;
}

function lapSummary(lap) {
  const { times, positions, attitudes, ...rest } = lap;
  return { ...rest, duration: times[times.length - 1] };
}

const COURSES = { first: { id: "c1", name: "First course" } };

/** A recording: its summary (with each course's first lap as its trail) and
 * the whole of it. */
function recording(steps, laps, courses = [COURSES.first]) {
  const trails = courses.map((_, c) => {
    const first = laps.find((lap) => lap.course === c);
    return first ? first.positions : null;
  });
  return {
    entry: {
      steps,
      info: { iteration: Math.floor(steps / 100) },
      courses,
      shape: "s",
      laps: laps.map(lapSummary),
      trails,
    },
    whole: { steps, laps },
  };
}
const entries = (...recordings) => recordings.map((r) => r.entry);

// ---- between the samples ----

test("slerp turns steadily about one axis", () => {
  const [q0, q1] = [turned(0), turned(90)];
  for (const fraction of [0, 0.25, 0.5, 0.9, 1]) {
    const q = slerp(q0, q1, fraction);
    close(norm(q), 1, 1e-15);
    close(heading(q), 90 * fraction, 1e-9);
  }
  // -q1 is the same attitude: the same way round, not 270 degrees back.
  allClose(slerp(q0, q1.map((v) => -v), 0.5), slerp(q0, q1, 0.5), 1e-15);
  // A tilt and a turn together: half-way is half the rotation between them.
  const a = fromEuler(0.3, 0.1, 0);
  const b = fromEuler(-0.2, 0.4, 1);
  const step = multiply(conjugate(a), slerp(a, b, 0.5));
  const whole = multiply(conjugate(a), b);
  allClose(rotationVector(step).map((v) => 2 * v), rotationVector(whole), 1e-12);
  const near = slerp(q0, turned(1e-3), 0.5); // almost equal: still a unit quaternion
  close(norm(near), 1, 1e-15);
  close(heading(near), 5e-4, 1e-9);
});

/** The rotation matrix BODY -> NED of a quaternion [w, x, y, z] (as
 * jineta.core.quaternion.to_matrix). */
function matrix([w, x, y, z]) {
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

test("three.js places the body in NED as the simulator does", () => {
  const q = fromEuler(0.2, -0.3, Math.PI / 2);
  const placed = new Quaternion().fromArray(threeQuaternion(q));
  const R = matrix(q);
  for (const b of [[0.13, 0.22, -0.023], [1, 0, 0], [0, 0, 1], [-0.2, 0.5, 0.1]]) {
    const v = new Vector3(...b).applyQuaternion(placed);
    allClose([v.x, v.y, v.z], R.map((row) => row[0] * b[0] + row[1] * b[1] + row[2] * b[2]), 1e-12);
  }
  // Nose East at a heading of 90 degrees, level; belly down (+z).
  const east = new Quaternion().fromArray(threeQuaternion(turned(90)));
  const nose = new Vector3(1, 0, 0).applyQuaternion(east);
  allClose([nose.x, nose.y, nose.z], [0, 1, 0], 1e-15);
  const belly = new Vector3(0, 0, 1).applyQuaternion(east);
  allClose([belly.x, belly.y, belly.z], [0, 0, 1], 1e-15);
  // A rounded quaternion (as in a recording) still gives a rotation.
  const rounded = threeQuaternion(q.map((v) => Math.round(v * 1000) / 1000));
  close(Math.hypot(...rounded), 1, 1e-15);
});

function threeSamples() {
  return {
    times: [0, 0.02, 0.04],
    positions: [
      [0, 0, -1],
      [0.2, 0, -1],
      [0.2, 0.4, -1.2],
    ],
    attitudes: [turned(0), turned(10), turned(30)],
  };
}

test("a pose between samples", () => {
  const lap = threeSamples();
  [0, 10, 30].forEach((deg, k) => {
    const pose = poseAt(lap, lap.times[k]); // at a sample, the sample
    allClose(pose.position, lap.positions[k], 1e-15);
    close(heading(pose.attitude), deg, 1e-9);
  });
  let pose = poseAt(lap, 0.03);
  allClose(pose.position, [0.2, 0.2, -1.1]);
  close(heading(pose.attitude), 20, 1e-9);
  pose = poseAt(lap, 0.005);
  allClose(pose.position, [0.05, 0, -1]);
  close(heading(pose.attitude), 2.5, 1e-9);
  // Before the start and after the end: where it stood and where it ended.
  allClose(poseAt(lap, -1).position, lap.positions[0]);
  allClose(poseAt(lap, 9).position, lap.positions[2]);
  // Rounded attitudes come back as unit quaternions.
  const rough = { ...lap, attitudes: lap.attitudes.map((q) => q.map((v) => Math.round(v * 100) / 100)) };
  close(norm(poseAt(rough, 0.01).attitude), 1, 1e-15);
  // The pose does not change the lap.
  assert.deepEqual(lap, threeSamples());
});

test("the path flown so far: the samples behind the drone", () => {
  const lap = threeSamples();
  assert.equal(poseAt(lap, 0).before, 0); // only where it is
  assert.equal(poseAt(lap, -1).before, 0);
  assert.equal(poseAt(lap, 0.02).before, 1); // a sample at t is where it is
  assert.equal(poseAt(lap, 0.03).before, 2);
  assert.equal(poseAt(lap, 0.04).before, 3); // ended: the whole path
  assert.equal(poseAt(lap, 5).before, 3);
  assert.equal(countUpTo([0, 1, 1, 2], 1), 3);
  assert.equal(countUpTo([0, 1, 2], -1), 0);
  assert.equal(countUpTo([0, 1, 2], 7), 3);
});

// ---- the player ----

test("recordings play in the order given, then the last again", () => {
  // The order given (the run's) is kept: a later stage can bring fewer steps.
  const player = new Player(entries(recording(100, [flown(0.5)]), recording(90, [flown(1)])));
  assert.deepEqual(player.entries.map((e) => e.steps), [100, 90]);
  close(player.duration(), 0.5);
  player.advance(0.2);
  close(player.t, 0.2);
  player.advance(1);
  close(player.t, 0.5); // held at the end of the laps
  player.advance(PAUSE - 0.01);
  assert.equal(player.index, 0); // still the pause after the end
  player.advance(0.02);
  assert.deepEqual([player.index, player.t], [1, 0]);
  player.advance(1);
  player.advance(PAUSE);
  assert.deepEqual([player.index, player.t], [1, 0]); // the last one again
});

test("the pause between recordings is wall time at any speed", () => {
  const player = new Player(entries(recording(100, [flown(0.4)]), recording(200, [flown(0.4)])));
  for (let k = 0; k < 3; k++) player.faster(); // x8
  player.advance(0.05); // 0.4 s of lap clock: the end
  close(player.t, 0.4);
  player.advance(PAUSE / 2);
  assert.equal(player.index, 0);
  player.advance(PAUSE / 2 + 1e-9);
  assert.equal(player.index, 1);
  // And after a jump (a key), a whole pause again at the end.
  player.advance(0.05);
  player.advance(PAUSE * 0.9);
  player.previous();
  player.advance(0.05);
  player.advance(PAUSE * 0.5);
  assert.equal(player.index, 0);
});

test("nothing moves until the recording's samples have arrived", () => {
  const player = new Player(entries(recording(100, [flown(0.4)]), recording(200, [flown(0.4)])));
  player.advance(0.3, false);
  assert.equal(player.t, 0);
  player.advance(0.4);
  player.advance(PAUSE * 2, false); // nor the pause after the end
  assert.equal(player.index, 0);
});

test("new recordings are played when the player gets there", () => {
  const player = new Player(entries(recording(100, [flown(0.5)])));
  player.advance(0.3);
  player.add(entries(recording(200, [flown(0.4)]), recording(150, [flown(0.2)])));
  // After the others, in the order given; the one being played goes on.
  assert.deepEqual(player.entries.map((e) => e.steps), [100, 200, 150]);
  assert.equal(player.index, 0);
  close(player.t, 0.3);
  player.advance(0.2);
  player.advance(PAUSE);
  assert.equal(player.current.steps, 200);
  player.add([]);
  assert.equal(player.current.steps, 200);
});

test("when the run's order changes, the same number is played", () => {
  const player = new Player(entries(...[100, 200, 300].map((s) => recording(s, [flown(0.5)]))));
  player.newest();
  player.advance(0.2);
  player.reset(entries(recording(50, [flown(0.5)]), recording(60, [flown(0.5)])));
  assert.equal(player.index, 1); // clipped to those there are
  close(player.t, 0.2);
  player.reset([]);
  assert.equal(player.index, 0);
  assert.equal(player.current, null);
});

test("the keys", () => {
  const player = new Player(entries(...[100, 200, 300].map((s) => recording(s, [flown(0.5)]))));
  player.advance(0.1);
  player.next();
  assert.deepEqual([player.index, player.t], [1, 0]);
  player.newest();
  player.next(); // nothing after the newest
  assert.equal(player.index, 2);
  player.first();
  player.previous();
  assert.equal(player.index, 0);
  player.togglePause();
  player.advance(0.3);
  assert.ok(player.t === 0 && player.paused);
  player.togglePause();
  assert.equal(player.speed, 1);
  player.faster();
  player.advance(0.1);
  assert.equal(player.speed, 2);
  close(player.t, 0.2);
  for (let k = 0; k < 10; k++) player.faster();
  assert.equal(player.speed, SPEEDS[SPEEDS.length - 1]);
  for (let k = 0; k < 10; k++) player.slower();
  assert.equal(player.speed, SPEEDS[0]);
  // From a speed off the list: the nearest on it, then one step.
  assert.equal(stepSpeed(3, +1), 8);
  assert.equal(stepSpeed(1.1, +1), 2);
  player.seek(9);
  close(player.t, 0.5);
  player.seek(-1);
  assert.equal(player.t, 0);
  player.goTo(7.6);
  assert.equal(player.index, 2);
  const empty = new Player();
  empty.advance(1);
  empty.next();
  empty.nextCourse();
  assert.equal(empty.frame(null), null);
  assert.equal(empty.duration(), 0);
  assert.deepEqual(empty.trails(), []);
  assert.deepEqual(describe(empty, 0), []);
});

test("one course at a time", () => {
  const three = [COURSES.first, { id: "c2", name: "Second" }, { id: "c3", name: "Third" }];
  const both = recording(
    100,
    [
      flown(0.5, { label: "First course" }),
      flown(0.3, { label: "Second", course: 1 }),
      flown(0.2, { label: "Third", course: 2 }),
    ],
    three,
  );
  const one = recording(200, [flown(0.4)]);
  const player = new Player(entries(both, one));
  assert.deepEqual(player.laps().map((l) => l.label), ["First course"]);
  player.advance(0.2);
  player.nextCourse();
  assert.deepEqual([player.course, player.t], [1, 0]);
  assert.deepEqual(player.laps().map((l) => l.label), ["Second"]);
  close(player.duration(), 0.3);
  player.nextCourse();
  assert.equal(player.course, 2);
  close(player.duration(), 0.2);
  player.nextCourse();
  assert.equal(player.course, 0);
  player.nextCourse();
  player.next(); // the next recording has one course: back to the first
  assert.equal(player.course, 0);
  assert.equal(player.current, one.entry);
  // Its whole recording: only the laps on the course shown fly.
  player.previous();
  player.nextCourse();
  assert.deepEqual(player.frame(both.whole).drones.map((d) => [d.label, d.lap]), [["Second", 1]]);
});

test("a frame says where every drone is", () => {
  const crash = flown(0.4, {
    status: "crashed",
    label: "noisy 1",
    crash_time: 0.33,
    crash_point: [0.33, 5, -1],
  });
  const mean = flown(0.6, { status: "finished" });
  const both = recording(100, [mean, crash]);
  const player = new Player(entries(both));
  player.seek(0.25);
  let frame = player.frame(both.whole);
  assert.deepEqual([frame.index, frame.course], [0, 0]);
  close(frame.duration, 0.6);
  assert.deepEqual(frame.drones.map((d) => d.label), ["mean action", "noisy 1"]);
  frame.drones.forEach((drone, k) => {
    const pose = poseAt([mean, crash][k], 0.25);
    allClose(drone.position, pose.position);
    allClose(drone.attitude, pose.attitude);
    assert.equal(drone.before, pose.before);
    assert.ok(!drone.ended);
  });
  assert.equal(frame.drones[1].crashPoint, null); // not there yet
  player.seek(0.35);
  allClose(player.frame(both.whole).drones[1].crashPoint, [0.33, 5, -1]);
  player.seek(0.5);
  frame = player.frame(both.whole);
  assert.ok(frame.drones[1].ended && !frame.drones[0].ended);
  assert.equal(frame.drones[1].before, crash.times.length);
  assert.deepEqual(frame.drones.map(result), ["...", "crashed"]);
  player.faster();
  player.togglePause();
  const text = describe(player, 7, "fly").join("\n");
  assert.ok(text.includes("recording 1 of 7: 100 steps (iteration 1), stage fly"));
  assert.ok(text.includes("x2  (paused)") && text.includes("First course"));
  player.seek(0.59); // between its last two samples: not ended yet
  assert.ok(!player.frame(both.whole).drones[0].ended);
  player.seek(0.6);
  assert.ok(player.frame(both.whole).drones[0].ended);
  assert.equal(result(player.frame(both.whole).drones[0]), "0.60 s");
  // Before its samples arrive, the recording plays with no drones.
  assert.deepEqual(player.frame(null).drones, []);
  // Thousands with commas, no iteration when there is none.
  const big = recording(2293760, [flown(0.2)]);
  delete big.entry.info.iteration;
  assert.equal(describe(new Player([big.entry]), 1)[0], "recording 1 of 1: 2,293,760 steps");
});

test("trails fade with their age", () => {
  const opacities = Array.from({ length: 10 }, (_, k) => trailOpacity(k + 1, 10));
  assert.equal(opacities[0], NEWEST_OPACITY);
  close(opacities[9], OLDEST_OPACITY);
  opacities.slice(1).forEach((o, k) => assert.ok(o < opacities[k]));
  assert.equal(trailOpacity(0, 10), 0);
  assert.equal(trailOpacity(11, 10), 0);
  assert.equal(trailOpacity(1, 1), NEWEST_OPACITY);
});

test("the trails are the earlier recordings' first laps", () => {
  const crash = flown(0.2, { status: "crashed", crash_time: 0.15, crash_point: [0.15, 5, -1] });
  const recordings = [
    recording(100, [crash, flown(0.3, { label: "noisy 1" })]),
    recording(200, [flown(0.4)]),
    recording(300, [flown(0.5, { status: "finished" })]),
    recording(400, [flown(0.6, { status: "finished" })]),
  ];
  const player = new Player(entries(...recordings));
  assert.deepEqual(player.trails(), []); // nothing before the first
  player.newest();
  const trails = player.trails(2);
  assert.deepEqual(trails.map((t) => t.steps), [300, 200]); // the most recent first
  assert.deepEqual(trails.map((t) => t.opacity), [trailOpacity(1, 2), trailOpacity(2, 2)]);
  assert.deepEqual(trails.map((t) => t.status), ["finished", "timeout"]);
  assert.deepEqual(trails[0].points, recordings[2].whole.laps[0].positions);
  const all = player.trails(10);
  const oldest = all[all.length - 1];
  assert.equal(oldest.steps, 100);
  assert.equal(oldest.status, "crashed");
  allClose(oldest.crashPoint, [0.15, 5, -1]);
  assert.ok(all.slice(0, -1).every((t) => t.crashPoint === null));
});

test("trails only of the same course", () => {
  const other = { id: "other", name: "Other" };
  const moved = { id: "moved", name: "First course" }; // same name, a gate moved
  // Random courses: the same validation courses, but listed in another order.
  const first = recording(
    100,
    [
      flown(0.3, { label: "Other", course: 0, status: "crashed" }),
      flown(0.2, { label: "First", course: 1, status: "finished" }),
    ],
    [other, COURSES.first],
  );
  const elsewhere = recording(200, [flown(0.4)], [moved]);
  const now = recording(300, [flown(0.5)]);
  const player = new Player(entries(first, elsewhere, now));
  player.newest();
  const trails = player.trails();
  assert.deepEqual(trails.map((t) => t.steps), [100]); // not the moved course's lap
  assert.deepEqual(trails[0].points, first.whole.laps[1].positions);
  assert.equal(trails[0].status, "finished"); // that lap's ending, not the other's
});

test("what the recordings say about learning", () => {
  const lap = (status, lapTime, course = 0) =>
    flown(0.2, { status, lap_time: lapTime, course });
  const two = [COURSES.first, { id: "c2", name: "Second" }];
  const list = entries(
    recording(100, [lap("crashed", null), lap("timeout", null)]),
    recording(200, [lap("finished", 5), lap("timeout", null)]),
    recording(300, [lap("finished", 4), lap("finished", 4.5)]),
    recording(400, [lap("finished", 7, 1), lap("finished", 6, 0)], two),
  );
  const points = progress(list, [10, 20, 30, 40]);
  assert.deepEqual(points.map((p) => p.steps), [10, 20, 30, 40]); // on the run's axis
  assert.deepEqual(points.map((p) => p.finished), [0, 0.5, 1, 1]);
  assert.deepEqual(points.map((p) => p.firstLapTime), [null, 5, 4, 6]); // each course's own
  assert.deepEqual(progress(list, [10, 20, 30, 40], 1).map((p) => [p.index, p.firstLapTime]), [[3, 7]]);
  assert.deepEqual(progress(list, []).map((p) => p.steps), [100, 200, 300, 400]);
});

test("the page reads every summary again when the run's order changed", () => {
  assert.ok(samePrefix(["fly:1", "fly:2"], ["fly:1", "fly:2", "fly:3"]));
  assert.ok(samePrefix([], ["fly:1"]));
  assert.ok(!samePrefix(["fly:1", "fly:2"], ["fly:1", "margin:2", "fly:2"]));
  assert.ok(!samePrefix(["fly:1", "fly:2"], ["fly:1"]));
});
