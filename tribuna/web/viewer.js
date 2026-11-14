// The viewer's page (phase 13): the scene drawn with three.js, the panel, the
// charts and the keys. What is shown at each moment comes from playback.js
// (tested with node); this file only draws it.
//
// The scene is drawn in NED, as the simulator computes: x North, y East, z
// down, so the camera's "up" is -z and the floor is z = 0, with the room above
// it (negative z). The shapes come from the server (`tribuna.geometry`: the same
// surfaces the collision model uses) and the recordings too (`tribuna.server`).
// A light from above casts the drone's shadow on the floor: its height shows.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { Line2 } from "three/addons/lines/Line2.js";
import { LineGeometry } from "three/addons/lines/LineGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";

import {
  Player,
  TRAILS,
  describe,
  progress,
  result,
  samePrefix,
  threeQuaternion,
} from "./playback.js";

const COLORS = {
  background: 0x0b0f15,
  floor: 0x151b24,
  walls: 0x1b2433,
  grid: 0x253042,
  edges: 0x5d6b80,
  gate: 0xff9f1c,
  obstacle: 0x6f8fb8,
  start: 0xe8edf5,
  frame: 0xdfe5ee,
  ccw: 0xff7a45,
  cw: 0x45b7ff,
  hitbox: 0xf3c623,
  flying: 0x4fb3ff,
  finished: 0x3ddc84,
  crashed: 0xff5c5c,
  timeout: 0x9aa4b2,
};
const CSS = (hex) => `#${hex.toString(16).padStart(6, "0")}`;
const POLL = 1000; // [ms] between asking the server for new recordings
const KEEP = 16; // whole recordings kept in memory (the others asked again)

const $ = (id) => document.getElementById(id);

// ---- talking to the server --------------------------------------------------------

async function get(path) {
  const response = await fetch(path, { cache: "no-store" });
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response.json();
}

const state = {
  player: new Player(),
  keys: [], // each recording's key (stage and steps), as the server sent them
  positions: [], // each recording's place on the run's axis of steps
  stages: [],
  log: null,
  errors: [],
  lost: false,
  wholes: new Map(), // index -> whole recording (or null while it comes)
  courses: new Map(), // course id -> drawn course (or null while it comes)
  drones: new Map(), // drone id -> its meshes' data (or null while it comes)
  version: 0, // goes up when the recordings or the log change (the charts redraw)
};

async function poll() {
  try {
    const known = state.keys.length;
    let data = await get(`/api/poll?known=${known}`);
    if (!samePrefix(state.keys, data.keys)) {
      // The run's order changed (a stage appeared before others): read all again.
      data = await get("/api/poll?known=0");
      state.wholes.clear();
      state.player.reset(data.recordings);
    } else if (data.recordings.length) {
      state.player.add(data.recordings);
    }
    if (data.keys.length !== known) state.version++;
    state.keys = data.keys;
    state.positions = data.positions;
    state.stages = data.stages;
    state.errors = data.errors;
    state.log = await get("/api/log");
    state.lost = false;
  } catch (error) {
    state.lost = true; // the viewer was closed, or the server is busy
  }
  state.version++;
  setTimeout(poll, POLL);
}

/** The whole recording `index` if it has arrived (else asks for it, and null). */
function whole(index) {
  if (index < 0 || index >= state.player.entries.length) return null;
  if (!state.wholes.has(index)) {
    state.wholes.set(index, null);
    get(`/api/recording?index=${index}`)
      .then((data) => state.wholes.set(index, data))
      .catch(() => state.wholes.delete(index));
    // Keep the ones near the recording being played.
    const here = state.player.index;
    const far = [...state.wholes.keys()].sort((a, b) => Math.abs(b - here) - Math.abs(a - here));
    while (state.wholes.size > KEEP) state.wholes.delete(far.shift());
  }
  return state.wholes.get(index);
}

/** The drawn course with this id (asked for from recording `index`). */
function course(id, index, number) {
  if (!state.courses.has(id)) {
    state.courses.set(id, null);
    get(`/api/course?index=${index}&course=${number}`)
      .then((data) => state.courses.set(id, buildCourse(data)))
      .catch(() => state.courses.delete(id));
  }
  return state.courses.get(id);
}

function droneData(id, index) {
  if (!state.drones.has(id)) {
    state.drones.set(id, null);
    get(`/api/drone?index=${index}`)
      .then((data) => state.drones.set(id, data))
      .catch(() => state.drones.delete(id));
  }
  return state.drones.get(id);
}

// ---- the scene ----------------------------------------------------------------------

const canvas = $("scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.toneMapping = THREE.NeutralToneMapping;

const scene = new THREE.Scene();
scene.background = new THREE.Color(COLORS.background);

const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 1000);
camera.up.set(0, 0, -1); // NED: up is -z (before the controls read it)
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.15;

const sky = new THREE.HemisphereLight(0xc9d8ff, 0x1a1e26, 1.4);
sky.position.set(0, 0, -1); // the sky is up: -z
scene.add(sky);
const sun = new THREE.DirectionalLight(0xffffff, 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.bias = -0.0004;
sun.shadow.normalBias = 0.02;
scene.add(sun, sun.target);

function surface(data, smooth = false) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(data.positions, 3));
  geometry.setIndex(data.index);
  if (smooth) geometry.computeVertexNormals();
  return geometry;
}

function segments(flat, color, width, opacity = 1) {
  const geometry = new LineSegmentsGeometry().setPositions(flat);
  const material = new LineMaterial({
    color,
    linewidth: width,
    transparent: opacity < 1,
    opacity,
  });
  return new LineSegments2(geometry, material);
}

function label(text, color) {
  const size = 128;
  const paper = document.createElement("canvas");
  paper.width = paper.height = size;
  const pen = paper.getContext("2d");
  pen.fillStyle = "rgba(11, 15, 21, 0.75)";
  pen.beginPath();
  pen.arc(size / 2, size / 2, size / 2 - 6, 0, 2 * Math.PI);
  pen.fill();
  pen.lineWidth = 6;
  pen.strokeStyle = color;
  pen.stroke();
  pen.fillStyle = color;
  pen.font = `bold ${size * 0.5}px Segoe UI, sans-serif`;
  pen.textAlign = "center";
  pen.textBaseline = "middle";
  pen.fillText(text, size / 2, size / 2 + 3);
  const texture = new THREE.CanvasTexture(paper);
  texture.colorSpace = THREE.SRGBColorSpace;
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthWrite: false }));
  sprite.scale.set(0.45, 0.45, 1);
  return sprite;
}

/** A course from /api/course: floor, walls, edges and grid, gates (numbered in
 * the order they are flown), obstacles, the start. */
function buildCourse(data) {
  const group = new THREE.Group();
  const [length, width, height] = data.room;

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(length, width),
    new THREE.MeshStandardMaterial({ color: COLORS.floor, roughness: 1 }),
  );
  floor.rotation.x = Math.PI; // its face up: -z
  floor.position.set(length / 2, width / 2, 0.004);
  floor.receiveShadow = true;
  group.add(floor);

  // The walls seen from inside: those far from the camera show, the near ones
  // (whose outside faces it) do not, as in a doll's house.
  const wall = new THREE.MeshStandardMaterial({
    color: COLORS.walls,
    side: THREE.BackSide,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
  });
  const none = new THREE.MeshBasicMaterial({ visible: false });
  const walls = new THREE.Mesh(new THREE.BoxGeometry(length, width, height), [
    wall,
    wall,
    wall,
    wall,
    none, // +z: the floor
    none, // -z: the ceiling
  ]);
  walls.position.set(length / 2, width / 2, -height / 2);
  walls.receiveShadow = true;
  group.add(walls);

  group.add(segments(data.grid, COLORS.grid, 1));
  group.add(segments(data.edges, COLORS.edges, 1.5));

  data.gates.forEach((gate, k) => {
    const mesh = new THREE.Mesh(
      surface(gate.surface, gate.round),
      new THREE.MeshStandardMaterial({
        color: COLORS.gate,
        emissive: COLORS.gate,
        emissiveIntensity: 0.18,
        roughness: 0.45,
        metalness: 0.1,
        flatShading: !gate.round,
      }),
    );
    mesh.castShadow = true;
    group.add(mesh);
    const box = new THREE.Box3().setFromBufferAttribute(mesh.geometry.attributes.position);
    const number = label(String(k + 1), CSS(COLORS.gate));
    number.position.set((box.min.x + box.max.x) / 2, (box.min.y + box.max.y) / 2, box.min.z - 0.35);
    group.add(number);
  });

  for (const obstacle of data.obstacles) {
    const mesh = new THREE.Mesh(
      surface(obstacle),
      new THREE.MeshStandardMaterial({ color: COLORS.obstacle, roughness: 0.8, flatShading: true }),
    );
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    group.add(mesh);
  }

  const start = new THREE.Mesh(
    new THREE.RingGeometry(0.22, 0.3, 40),
    new THREE.MeshBasicMaterial({ color: COLORS.start, side: THREE.DoubleSide }),
  );
  start.position.set(data.start[0], data.start[1], -0.006);
  group.add(start);

  return { data, group };
}

let shownCourse = null; // the drawn course on screen
let shownRoom = ""; // its room's size (the camera is placed again when it changes)
let follow = false;
let showHitbox = false;
let panels = true; // the text and the charts over the scene

/** The part of the window the panels leave free [px]. */
function freeArea() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  if (!panels) return { left: 0, right: width, top: 0, bottom: height };
  return {
    left: $("hud").getBoundingClientRect().right + 12,
    right: $("charts").getBoundingClientRect().left - 12,
    top: $("top").getBoundingClientRect().bottom + 12,
    bottom: height - 36,
  };
}

const corner = new THREE.Vector3();

/** The whole room in view: from the server's overview camera (its direction
 * and focus), moved nearer or farther until the room's eight corners just fit
 * in the free part of the window. */
function overview() {
  if (!shownCourse) return;
  const view = shownCourse.data.camera;
  camera.position.fromArray(view.position);
  controls.target.fromArray(view.focus);
  const [length, width, height] = shownCourse.data.room;
  const free = freeArea();
  const middle = [(free.left + free.right) / 2, (free.top + free.bottom) / 2];
  const half = [(free.right - free.left) / 2, (free.bottom - free.top) / 2];
  for (let pass = 0; pass < 6; pass++) {
    camera.lookAt(controls.target);
    camera.updateMatrixWorld();
    let worst = 0;
    for (const x of [0, length])
      for (const y of [0, width])
        for (const z of [0, -height]) {
          corner.set(x, y, z).project(camera);
          const px = ((corner.x + 1) / 2) * window.innerWidth;
          const py = ((1 - corner.y) / 2) * window.innerHeight;
          worst = Math.max(worst, Math.abs(px - middle[0]) / half[0], Math.abs(py - middle[1]) / half[1]);
        }
    const away = camera.position.clone().sub(controls.target);
    camera.position.copy(controls.target).add(away.multiplyScalar(worst));
  }
  controls.update();
}

function showCourse(drawn) {
  if (drawn === shownCourse) return;
  if (shownCourse) scene.remove(shownCourse.group);
  shownCourse = drawn;
  scene.add(drawn.group);
  const [length, width, height] = drawn.data.room;
  const reach = Math.max(length, width);
  sun.position.set(length * 0.35, width * 0.4, -height * 4);
  sun.target.position.set(length / 2, width / 2, 0);
  Object.assign(sun.shadow.camera, {
    left: -reach,
    right: reach,
    top: reach,
    bottom: -reach,
    near: 0.1,
    far: height * 8,
  });
  sun.shadow.camera.updateProjectionMatrix();
  const room = drawn.data.room.join(",");
  if (room !== shownRoom) {
    shownRoom = room;
    follow = false;
    overview();
  }
}

// ---- the drones and their paths ---------------------------------------------------

const material = {
  frame: new THREE.MeshStandardMaterial({ color: COLORS.frame, roughness: 0.5, flatShading: true }),
  ccw: new THREE.MeshStandardMaterial({
    color: COLORS.ccw,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  }),
  cw: new THREE.MeshStandardMaterial({
    color: COLORS.cw,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide,
  }),
  hitbox: new THREE.MeshStandardMaterial({
    color: COLORS.hitbox,
    transparent: true,
    opacity: 0.28,
    depthWrite: false,
  }),
  crash: new THREE.MeshStandardMaterial({
    color: COLORS.crashed,
    emissive: COLORS.crashed,
    emissiveIntensity: 0.6,
  }),
};
const crashBall = new THREE.SphereGeometry(0.09, 16, 12);

/** One drone on screen: its body, rotors and spheres, its path behind it (the
 * samples flown, and a last piece to where it is now) and its crash mark. */
function makeDrone(data) {
  const body = new THREE.Group();
  for (const [part, smooth] of [
    ["frame", false],
    ["ccw", false],
    ["cw", false],
  ]) {
    const mesh = new THREE.Mesh(surface(data[part], smooth), material[part]);
    mesh.castShadow = true;
    body.add(mesh);
  }
  const hitbox = new THREE.Mesh(surface(data.hitbox, true), material.hitbox);
  body.add(hitbox);
  const pathMaterial = new LineMaterial({ color: COLORS.flying, linewidth: 2.5, transparent: true });
  const path = new Line2(new LineGeometry(), pathMaterial);
  const tip = new Line2(new LineGeometry().setPositions([0, 0, 0, 0, 0, 0]), pathMaterial);
  tip.frustumCulled = false;
  const crash = new THREE.Mesh(crashBall, material.crash);
  scene.add(body, path, tip, crash);
  return { body, hitbox, path, tip, crash, pathMaterial, points: null };
}

function removeDrone(drone) {
  scene.remove(drone.body, drone.path, drone.tip, drone.crash);
  drone.path.geometry.dispose();
  drone.tip.geometry.dispose();
}

/** A path's points as the flat list a line takes (a line needs two). */
function flat(points) {
  const list = points.length > 1 ? points : [points[0], points[0]];
  return list.flat();
}

let drones = []; // on screen
let droneShape = ""; // their drone's id
let trails = []; // on screen
let shown = ""; // what the drones and trails were drawn for

function clearTrails() {
  for (const trail of trails) {
    scene.remove(trail.line, trail.crash);
    trail.line.geometry.dispose();
    trail.line.material.dispose();
    trail.crash.material.dispose();
  }
  trails = [];
}

function drawTrails() {
  clearTrails();
  for (const trail of state.player.trails(TRAILS)) {
    const color = COLORS[trail.status] ?? COLORS.timeout;
    const line = new Line2(
      new LineGeometry().setPositions(flat(trail.points)),
      new LineMaterial({
        color,
        linewidth: 1.6,
        transparent: true,
        opacity: trail.opacity,
        depthWrite: false,
      }),
    );
    const crash = new THREE.Mesh(
      crashBall,
      new THREE.MeshBasicMaterial({ color, transparent: true, opacity: trail.opacity }),
    );
    crash.visible = trail.crashPoint !== null;
    if (trail.crashPoint) crash.position.fromArray(trail.crashPoint);
    crash.scale.setScalar(0.7);
    scene.add(line, crash);
    trails.push({ line, crash });
  }
}

/** The drones for the recording being played, once it and its drone have
 * arrived: one per lap on the course, each path laid out whole (only the part
 * flown is drawn). */
function layOut(recording, frame) {
  const entry = state.player.current;
  const data = droneData(entry.shape, state.player.index);
  if (!data) return false;
  if (droneShape !== entry.shape) {
    drones.forEach(removeDrone);
    drones = [];
    droneShape = entry.shape;
  }
  while (drones.length < frame.drones.length) drones.push(makeDrone(data));
  while (drones.length > frame.drones.length) removeDrone(drones.pop());
  frame.drones.forEach((drone, k) => {
    const lap = recording.laps[drone.lap];
    const view = drones[k];
    view.points = lap.positions;
    view.path.geometry.dispose();
    view.path.geometry = new LineGeometry().setPositions(flat(lap.positions));
    // The first lap (the mean action: the policy as it would fly) stands out.
    view.pathMaterial.linewidth = k === 0 ? 3 : 1.6;
    view.pathMaterial.opacity = k === 0 ? 1 : 0.6;
  });
  drawTrails();
  return true;
}

const tipStart = new THREE.Vector3();

function place(frame) {
  frame.drones.forEach((drone, k) => {
    const view = drones[k];
    if (!view) return;
    view.body.position.fromArray(drone.position);
    view.body.quaternion.fromArray(threeQuaternion(drone.attitude));
    view.hitbox.visible = showHitbox && k === 0;
    // The path: the samples behind the drone, and a last piece to it.
    view.path.geometry.instanceCount = Math.max(drone.before - 1, 0);
    view.tip.visible = !drone.ended && drone.before > 0;
    if (view.tip.visible) {
      tipStart.fromArray(view.points[drone.before - 1]);
      const ends = view.tip.geometry.attributes.instanceStart.data;
      ends.array.set([tipStart.x, tipStart.y, tipStart.z, ...drone.position]);
      ends.needsUpdate = true;
    }
    const color = drone.ended ? (COLORS[drone.status] ?? COLORS.timeout) : COLORS.flying;
    view.pathMaterial.color.setHex(color);
    view.crash.visible = drone.crashPoint !== null;
    if (drone.crashPoint) view.crash.position.fromArray(drone.crashPoint);
  });
}

// ---- following a drone ---------------------------------------------------------------

const chase = new THREE.Vector3();

function toggleFollow() {
  follow = !follow;
  if (!follow) {
    overview();
    return;
  }
  const first = drones[0];
  if (!first) {
    follow = false;
    return;
  }
  // Keep the direction the camera looks from; come to 3.5 m of the drone.
  const away = camera.position.clone().sub(controls.target).setLength(3.5);
  controls.target.copy(first.body.position);
  camera.position.copy(first.body.position).add(away);
}

function chaseCamera() {
  if (!follow || !drones[0]) return;
  chase.copy(drones[0].body.position).sub(controls.target);
  controls.target.add(chase);
  camera.position.add(chase);
}

// ---- the panels ------------------------------------------------------------------------

const hud = { lines: $("lines"), laps: $("laps"), errors: $("errors") };
let hudText = "";

function writeHud(frame) {
  const player = state.player;
  const lines = describe(player, player.entries.length, state.stages[player.index] ?? "");
  const laps = frame
    ? frame.drones.map((drone, k) => {
        const status = drone.ended ? drone.status : "flying";
        return `<li class="${k === 0 ? "first" : ""}"><i class="dot ${status}"></i><span class="label">${drone.label}</span><span>${result(drone)}</span></li>`;
      })
    : [];
  const waiting = frame && !frame.drones.length ? ["<li>loading the laps&hellip;</li>"] : [];
  const errors = [
    ...(state.lost ? ["the server does not answer"] : []),
    ...state.errors.map((e) => `skipped: ${e}`),
  ];
  const text = [lines.join("\n"), laps.join(""), waiting.join(""), errors.join("\n")].join("|");
  if (text === hudText) return;
  hudText = text;
  hud.lines.innerHTML = lines.map((line) => `<div>${line}</div>`).join("");
  hud.laps.innerHTML = laps.join("") + waiting.join("");
  hud.errors.textContent = errors.join("\n");
}

const controlsUi = {
  recording: $("recording"),
  recordingLabel: $("recording-label"),
  clock: $("clock"),
  clockLabel: $("clock-label"),
  play: $("play"),
  speed: $("speed"),
};
let dragging = false;

function writeControls() {
  const player = state.player;
  const total = player.entries.length;
  const ui = controlsUi;
  ui.recording.max = Math.max(total - 1, 0);
  if (document.activeElement !== ui.recording) ui.recording.value = player.index;
  ui.recordingLabel.textContent = total
    ? `${player.index + 1} / ${total} · ${Math.round(player.current.steps).toLocaleString("en")} steps`
    : "–";
  const duration = player.duration();
  ui.clock.max = Math.max(duration, 0.01);
  if (!dragging) ui.clock.value = player.t;
  ui.clockLabel.textContent = `${player.t.toFixed(2)} / ${duration.toFixed(2)} s`;
  ui.play.innerHTML = player.paused ? "&#x23F5;" : "&#x23F8;";
  ui.speed.textContent = `x${player.speed}`;
}

// ---- the charts ------------------------------------------------------------------------

function niceStep(range, ticks) {
  const raw = range / ticks;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = raw / power;
  return power * (unit < 1.5 ? 1 : unit < 3.5 ? 2 : unit < 7.5 ? 5 : 10);
}

function short(steps) {
  if (steps >= 1e6) return `${+(steps / 1e6).toFixed(1)}M`;
  if (steps >= 1e3) return `${+(steps / 1e3).toFixed(0)}k`;
  return `${steps}`;
}

/** A chart on a canvas: lines and dots against steps, and a dashed line at the
 * recording being played. Returns how to turn a click into steps. */
function chart(canvasElement, { right, top, lines, dots, now, ticks }) {
  const ratio = window.devicePixelRatio || 1;
  const width = canvasElement.clientWidth;
  const height = canvasElement.clientHeight;
  if (canvasElement.width !== Math.round(width * ratio)) canvasElement.width = Math.round(width * ratio);
  if (canvasElement.height !== Math.round(height * ratio)) canvasElement.height = Math.round(height * ratio);
  const pen = canvasElement.getContext("2d");
  pen.setTransform(ratio, 0, 0, ratio, 0, 0);
  pen.clearRect(0, 0, width, height);
  const box = { left: 36, right: width - 8, top: 8, bottom: height - 20 };
  const x = (v) => box.left + (v / right) * (box.right - box.left);
  const y = (v) => box.bottom - (v / top) * (box.bottom - box.top);

  pen.font = "10px Consolas, monospace";
  pen.fillStyle = "#8b96a8";
  pen.strokeStyle = "rgba(255,255,255,0.07)";
  pen.lineWidth = 1;
  for (const v of ticks) {
    pen.beginPath();
    pen.moveTo(box.left, y(v));
    pen.lineTo(box.right, y(v));
    pen.stroke();
    pen.textAlign = "right";
    pen.fillText(String(v), box.left - 5, y(v) + 3);
  }
  const step = niceStep(right, 4);
  for (let v = 0; v <= right; v += step) {
    pen.textAlign = "center";
    pen.fillText(short(v), x(v), height - 5);
  }

  for (const line of lines) {
    pen.strokeStyle = line.color;
    pen.lineWidth = 1.8;
    pen.beginPath();
    let drawing = false;
    for (const [px, py] of line.points) {
      if (py === null || !Number.isFinite(py)) {
        drawing = false;
        continue;
      }
      if (drawing) pen.lineTo(x(px), y(py));
      else pen.moveTo(x(px), y(py));
      drawing = true;
    }
    pen.stroke();
  }
  for (const dot of dots ?? []) {
    pen.fillStyle = dot.color;
    pen.beginPath();
    pen.arc(x(dot.x), y(dot.y), dot.size, 0, 2 * Math.PI);
    pen.fill();
  }
  if (now !== null) {
    pen.strokeStyle = "rgba(255,255,255,0.75)";
    pen.setLineDash([4, 4]);
    pen.beginPath();
    pen.moveTo(x(now), box.top);
    pen.lineTo(x(now), box.bottom);
    pen.stroke();
    pen.setLineDash([]);
  }
  return (clientX) => ((clientX - box.left) / (box.right - box.left)) * right;
}

let charted = "";
let clickToSteps = null;

function drawCharts() {
  const player = state.player;
  const size = `${$("ends").clientWidth}x${$("ends").clientHeight}`;
  const key = `${state.version}|${player.index}|${player.course}|${size}`;
  if (key === charted) return;
  charted = key;
  const log = state.log ?? { steps: [], finished: [], crashed: [], timeout: [], mean_lap_time: [] };
  const points = progress(player.entries, state.positions, player.course);
  const now = player.entries.length ? (state.positions[player.index] ?? player.current.steps) : null;
  const right =
    Math.max(1, now ?? 0, ...log.steps, ...points.map((p) => p.steps)) * 1.02;
  const pairs = (key) => log.steps.map((s, i) => [s, log[key][i]]);
  clickToSteps = chart($("ends"), {
    right,
    top: 1,
    ticks: [0, 0.5, 1],
    lines: [
      { color: CSS(COLORS.timeout), points: pairs("timeout") },
      { color: CSS(COLORS.crashed), points: pairs("crashed") },
      { color: CSS(COLORS.finished), points: pairs("finished") },
    ],
    dots: points.map((p) => ({ x: p.steps, y: p.finished, color: "#ffffff", size: 2.5 })),
    now,
  });
  const times = [...log.mean_lap_time, ...points.map((p) => p.firstLapTime)].filter((v) =>
    Number.isFinite(v),
  );
  const top = times.length ? Math.ceil(Math.max(...times) * 1.15) : 10;
  const tick = niceStep(top, 3);
  const ticks = [];
  for (let v = 0; v <= top + 1e-9; v += tick) ticks.push(+v.toFixed(6));
  chart($("times"), {
    right,
    top,
    ticks,
    lines: [{ color: CSS(COLORS.flying), points: pairs("mean_lap_time") }],
    dots: points
      .filter((p) => p.firstLapTime !== null)
      .map((p) => ({ x: p.steps, y: p.firstLapTime, color: "#ffffff", size: 3 })),
    now,
  });
}

function jumpTo(event) {
  if (!clickToSteps || !state.player.entries.length) return;
  const rect = event.currentTarget.getBoundingClientRect();
  const steps = clickToSteps(event.clientX - rect.left);
  let best = 0;
  state.player.entries.forEach((entry, i) => {
    const here = state.positions[i] ?? entry.steps;
    const there = state.positions[best] ?? state.player.entries[best].steps;
    if (Math.abs(here - steps) < Math.abs(there - steps)) best = i;
  });
  state.player.goTo(best);
}

// ---- keys and buttons ---------------------------------------------------------------

function hitbox() {
  showHitbox = !showHitbox;
}

const actions = {
  " ": () => state.player.togglePause(),
  ArrowRight: () => state.player.next(),
  ArrowLeft: () => state.player.previous(),
  Home: () => state.player.first(),
  End: () => state.player.newest(),
  l: () => state.player.faster(),
  "+": () => state.player.faster(),
  k: () => state.player.slower(),
  "-": () => state.player.slower(),
  c: () => state.player.nextCourse(),
  g: toggleFollow,
  h: hitbox,
  p: togglePanels,
  r: () => {
    follow = false;
    overview();
  },
};

window.addEventListener("keydown", (event) => {
  const action = actions[event.key] ?? actions[event.key.toLowerCase()];
  if (!action || event.ctrlKey || event.altKey || event.metaKey) return;
  event.preventDefault();
  document.activeElement?.blur?.(); // a slider would take the arrows too
  action();
});

const buttons = {
  first: () => state.player.first(),
  previous: () => state.player.previous(),
  play: () => state.player.togglePause(),
  next: () => state.player.next(),
  newest: () => state.player.newest(),
  slower: () => state.player.slower(),
  faster: () => state.player.faster(),
};
for (const [id, action] of Object.entries(buttons)) {
  $(id).addEventListener("click", (event) => {
    action();
    event.currentTarget.blur();
  });
}
controlsUi.recording.addEventListener("input", () => state.player.goTo(Number(controlsUi.recording.value)));
controlsUi.recording.addEventListener("change", () => controlsUi.recording.blur());
controlsUi.clock.addEventListener("pointerdown", () => (dragging = true));
window.addEventListener("pointerup", () => (dragging = false));
controlsUi.clock.addEventListener("input", () => state.player.seek(Number(controlsUi.clock.value)));
$("ends").addEventListener("click", jumpTo);
$("times").addEventListener("click", jumpTo);

/** Fit the drawing to the window, with the middle of the view in the space the
 * panels leave free (the camera's view is shifted, not cut). */
function resize() {
  const width = window.innerWidth;
  const height = window.innerHeight;
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  const free = freeArea();
  if (panels && free.right > free.left) {
    const middle = [(free.left + free.right) / 2, (free.top + free.bottom) / 2];
    camera.setViewOffset(width, height, width / 2 - middle[0], height / 2 - middle[1], width, height);
  } else {
    camera.clearViewOffset();
  }
  camera.updateProjectionMatrix();
  charted = "";
}

function togglePanels() {
  panels = !panels;
  for (const id of ["hud", "charts", "help"]) $(id).classList.toggle("hidden", !panels);
  resize();
  if (!follow) overview();
}
window.addEventListener("resize", resize);
resize();

// ---- every frame -------------------------------------------------------------------------

let last = performance.now();

function update(now) {
  const dt = Math.min((now - last) / 1000, 0.2); // a stall is not a jump
  last = now;
  const player = state.player;
  $("waiting").classList.toggle("hidden", player.entries.length > 0);
  if (player.entries.length) {
    const recording = whole(player.index);
    whole(player.index + 1); // the next one, ready when it is its turn
    const entry = player.current;
    const id = entry.courses[player.course].id;
    const drawn = course(id, player.index, player.course);
    if (drawn) showCourse(drawn);
    let frame = player.frame(recording);
    const key = `${player.index}|${player.course}|${recording ? 1 : 0}|${state.keys[player.index]}`;
    let ready = false;
    if (recording && drawn) {
      ready = key === shown || layOut(recording, frame);
      if (ready) shown = key;
    }
    if (ready) {
      player.advance(dt, true);
      // On to the next recording (or course): drawn once its laps are here.
      const same = `${player.index}|${player.course}` === key.split("|", 2).join("|");
      frame = player.frame(same ? recording : null);
      if (same) place(frame);
      else ready = false;
    }
    for (const drone of drones) drone.body.visible = ready;
    writeHud(frame);
    writeControls();
    drawCharts();
  }
  chaseCamera();
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(update);
}

// For the tests (tests/test_tribuna_web.py and the checks in a browser).
window.tribuna = { state, camera, controls, renderer, scene };

poll();
requestAnimationFrame(update);
