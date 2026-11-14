"""Tests for the viewer's server and start (phase 13, `tribuna.server`,
`tribuna.__main__`): what the page is sent, read from a real training's folder
as it grows; and the page's playback (tribuna/web/playback.js), whose tests run
with node when it is installed."""

import dataclasses
import json
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from jineta.campo.gates import CircularGate
from jineta.doma.policy import Agent
from jineta.jinete.profiles import MULTIROTOR_PROFILES
from jineta.picadero.recording import (
    Recording,
    drone_shape,
    recording_files,
    recording_to_dict,
)
from jineta.picadero.training import LAPS_FOLDER, save_recording
from tests.test_recording import COURSE, lap
from tests.test_training import SHORT
from tests.test_training_record import short_training
from tribuna.geometry import (
    Mesh,
    course_meshes,
    overview_camera,
    segments,
    triangles,
)
from tribuna.run import Run
from tribuna.server import (
    CONTENT_TYPES,
    WEB,
    Feed,
    course_id,
    handler_for,
    shape_id,
    start,
    summary,
)

SHAPE = drone_shape(MULTIROTOR_PROFILES["iris"])
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def serve():
    """Start servers for a test, and stop them after it."""
    servers = []

    def run(path):
        server, url = start(path)
        servers.append(server)

        def get(address):
            with urllib.request.urlopen(url + address) as response:
                body = response.read()
                return response.headers["Content-Type"], body

        def get_json(address):
            kind, body = get(address)
            assert kind == "application/json"
            return json.loads(body)

        return url, get, get_json

    yield run
    for server in servers:
        server.shutdown()
        server.server_close()


# ---- shapes for three.js -------------------------------------------------------------


def test_faces_cut_into_triangles_and_lines_into_segments():
    square = Mesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]),
        faces=np.array([4, 0, 1, 2, 3]),
    )
    tris = triangles(square)
    np.testing.assert_array_equal(tris, [[0, 1, 2], [0, 2, 3]])  # a fan
    p = square.points
    area = sum(
        0.5 * np.linalg.norm(np.cross(p[b] - p[a], p[c] - p[a])) for a, b, c in tris
    )
    assert area == pytest.approx(1.0)  # the whole square, once
    for a, b, c in tris:  # the same side out as the square (+z)
        assert np.cross(p[b] - p[a], p[c] - p[a])[2] > 0.0
    path = Mesh(
        np.array([[0.0, 0, 0], [1, 0, 0], [1, 2, 0]]), lines=np.array([3, 0, 1, 2])
    )
    np.testing.assert_array_equal(
        segments(path), [[[0, 0, 0], [1, 0, 0]], [[1, 0, 0], [1, 2, 0]]]
    )
    assert segments(square).shape == (0, 2, 3)
    assert triangles(path).shape == (0, 3)


def test_course_and_drone_ids():
    assert course_id(COURSE) == course_id(COURSE.__class__(**COURSE.__dict__))
    renamed = COURSE.__class__(**{**COURSE.__dict__, "name": "Other"})
    gate = COURSE.gates[0]
    moved_gate = gate.__class__(
        **{**gate.__dict__, "center": gate.center + np.array([0.0, 0.1, 0.0])}
    )
    moved = COURSE.__class__(
        **{**COURSE.__dict__, "gates": (moved_gate, *COURSE.gates[1:])}
    )
    assert len({course_id(COURSE), course_id(renamed), course_id(moved)}) == 3
    other = dataclasses.replace(SHAPE, propeller_radius=SHAPE.propeller_radius + 0.01)
    assert shape_id(SHAPE) == shape_id(drone_shape(MULTIROTOR_PROFILES["iris"]))
    assert shape_id(SHAPE) != shape_id(other)


# ---- the server ----------------------------------------------------------------------


def test_the_page_and_its_files(serve, tmp_path):
    _, get, _ = serve(tmp_path)
    kind, body = get("")
    assert kind == CONTENT_TYPES[".html"] and b"viewer.js" in body
    # Module scripts must go out as JavaScript, whatever Windows' registry says.
    scripts = (
        "viewer.js",
        "playback.js",
        "vendor/three.module.js",
        "vendor/three.core.js",
        "vendor/addons/controls/OrbitControls.js",
        "vendor/addons/lines/Line2.js",
    )
    for name in scripts:
        kind, body = get(name)
        assert kind == "text/javascript; charset=utf-8", name
        assert body == (WEB / name).read_bytes()
    assert get("style.css")[0] == "text/css; charset=utf-8"
    # Nothing outside the page's folder, and nothing it does not use.
    outside = (
        "../server.py",
        "vendor/../../server.py",
        "%2e%2e/server.py",
        "nothing.js",
        "vendor",
        "../../tests/test_tribuna_web.py",
        "../../requirements.txt",  # a kind of file it serves, but outside
        "vendor/../../../requirements.txt",
    )
    for name in outside:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(name)
        assert error.value.code == 404, name


def test_only_the_kinds_of_file_the_page_uses(tmp_path):
    """A file in the page's folder goes out only if it is a kind the page uses
    (and with its kind said)."""
    web = tmp_path / "web"
    web.mkdir()
    (web / "index.html").write_text("<p>page</p>", encoding="utf-8")
    (web / "notes.py").write_text("secret = 1", encoding="utf-8")
    (web / "data.json").write_text("{}", encoding="utf-8")
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(Feed(tmp_path), web))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    try:
        with urllib.request.urlopen(url) as response:
            assert response.read() == b"<p>page</p>"
        with urllib.request.urlopen(url + "data.json") as response:
            assert response.headers["Content-Type"] == "application/json"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "notes.py")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()


def test_every_file_the_page_imports_is_there():
    """The page loads three.js from vendor/ (no internet): each import of the
    page and of the files it brings must be a file the server has."""
    page = (WEB / "index.html").read_text(encoding="utf-8")
    imports = json.loads(page.split('type="importmap">')[1].split("</script>")[0])[
        "imports"
    ]
    assert (WEB / imports["three"]).is_file()
    for script in [WEB / "viewer.js", *(WEB / "vendor").rglob("*.js")]:
        for line in script.read_text(encoding="utf-8").splitlines():
            if (
                not (line.startswith("import") or line.startswith("} from"))
                or "from" not in line
            ):
                continue
            name = line.split("from")[-1].strip().strip(";").strip("'\"")
            if name == "three":
                continue
            if name.startswith("three/addons/"):
                target = (
                    WEB / imports["three/addons/"] / name.removeprefix("three/addons/")
                )
            else:
                target = script.parent / name
            assert target.resolve().is_file(), f"{script.name}: {name}"
    assert (WEB / "vendor" / "LICENSE-three.txt").is_file()


def test_a_run_is_sent_as_it_grows(serve, tmp_path):
    short_training(tmp_path / "run")
    _, _, get = serve(tmp_path / "run")
    data = get("api/poll")
    assert data["keys"] == ["run:256", "run:512", "run:768"]
    assert data["positions"] == [256.0, 512.0, 768.0]
    assert data["stages"] == ["run"] * 3 and data["errors"] == []
    assert [r["steps"] for r in data["recordings"]] == [256, 512, 768]
    assert get("api/poll?known=3")["recordings"] == []  # nothing new
    # A recording written while the page watches, between two others: the
    # page sees its first keys changed and reads every summary again.
    run = Run(tmp_path / "run")
    first = run.poll()
    agent = Agent.load(tmp_path / "run" / "agents" / "000000000512.npz")
    agent.info = {**agent.info, "steps": 640}
    save_recording(tmp_path / "run", agent, (SHORT,), first[0].laps, SHAPE)
    data = get("api/poll?known=3")
    assert data["keys"] == ["run:256", "run:512", "run:640", "run:768"]
    assert [r["steps"] for r in data["recordings"]] == [768]  # those after the third
    assert [r["steps"] for r in get("api/poll")["recordings"]] == [256, 512, 640, 768]
    # A file that cannot be read is said, not sent.
    files = recording_files(tmp_path / "run" / LAPS_FOLDER)
    broken = tmp_path / "run" / LAPS_FOLDER / "000000000900.json"
    broken.write_text("{ not json", encoding="utf-8")
    data = get("api/poll?known=4")
    assert (
        len(data["keys"]) == 4
        and len(data["errors"]) == 1
        and "000000000900" in data["errors"][0]
    )
    assert len(files) == 4
    # The log, on the run's axis.
    log = get("api/log")
    assert list(log) == ["steps", "finished", "crashed", "timeout", "mean_lap_time"]
    assert log["steps"] == [256.0, 512.0, 768.0]
    np.testing.assert_allclose(
        np.add(log["finished"], log["crashed"]) + np.array(log["timeout"]), 1.0
    )


def test_what_a_summary_holds(serve, tmp_path):
    short_training(tmp_path)
    recording = Run(tmp_path).poll()[0]
    _, _, get = serve(tmp_path)
    sent = get("api/poll")["recordings"][0]
    assert sent == json.loads(json.dumps(summary(recording)))
    assert sent["courses"] == [{"id": course_id(SHORT), "name": SHORT.name}]
    assert sent["shape"] == shape_id(recording.shape)
    assert len(sent["laps"]) == len(recording.laps)
    for sent_lap, flown in zip(sent["laps"], recording.laps, strict=True):
        assert (sent_lap["label"], sent_lap["status"], sent_lap["course"]) == (
            flown.label,
            flown.status,
            0,
        )
        assert sent_lap["duration"] == flown.duration
        if flown.status != "finished":
            assert sent_lap["lap_time"] is None  # nan goes out as null
        if flown.status != "crashed":
            assert sent_lap["crash_point"] == [None, None, None]
    # The trail: the first lap's path, to the millimetre.
    np.testing.assert_allclose(
        sent["trails"][0], recording.laps[0].positions, atol=5e-4
    )
    # On a set of courses, each course's trail is its own first lap.
    second = COURSE.__class__(**{**COURSE.__dict__, "name": "Second"})
    laps = (
        lap(course=1, label="b", positions=np.ones((2, 3))),
        lap(course=0, label="a", positions=np.full((2, 3), 2.0)),
        lap(course=1, label="c", positions=np.full((2, 3), 3.0)),
    )
    both = summary(Recording(5, (COURSE, second), SHAPE, laps))
    assert both["trails"] == [np.full((2, 3), 2.0).tolist(), np.ones((2, 3)).tolist()]
    assert [c["name"] for c in both["courses"]] == [COURSE.name, "Second"]
    assert (
        summary(Recording(5, (COURSE, second), SHAPE, laps[1:2]))["trails"][1] is None
    )
    # The whole recording, as its file holds it.
    assert get("api/recording?index=0") == json.loads(
        json.dumps(recording_to_dict(recording))
    )


def test_a_course_and_a_drone_for_the_page(serve, tmp_path):
    short_training(tmp_path)
    recording = Run(tmp_path).poll()[0]
    _, _, get = serve(tmp_path)
    course = get("api/course?index=0&course=0")
    meshes = course_meshes(SHORT)
    assert course["id"] == course_id(SHORT) and course["room"] == list(SHORT.room.size)
    assert len(course["gates"]) == len(SHORT.gates)
    for sent, mesh, gate in zip(
        course["gates"], meshes.gates, SHORT.gates, strict=True
    ):
        surface = sent["surface"]
        np.testing.assert_allclose(
            np.reshape(surface["positions"], (-1, 3)), mesh.points, atol=1e-5
        )
        np.testing.assert_array_equal(
            np.reshape(surface["index"], (-1, 3)), triangles(mesh)
        )
        assert sent["round"] == isinstance(gate, CircularGate)
    assert len(course["obstacles"]) == len(SHORT.obstacles)
    np.testing.assert_allclose(
        np.reshape(course["edges"], (-1, 2, 3)), segments(meshes.edges), atol=1e-5
    )
    assert len(course["grid"]) == segments(meshes.grid).size
    position, focus, _ = overview_camera(SHORT.room)
    np.testing.assert_allclose(course["camera"]["position"], position)
    np.testing.assert_allclose(course["camera"]["focus"], focus)
    assert course["camera"]["up"] == [0.0, 0.0, -1.0]
    drone = get("api/drone?index=0")
    assert drone["id"] == shape_id(recording.shape)
    for part in ("frame", "ccw", "cw", "hitbox"):
        points = np.reshape(drone[part]["positions"], (-1, 3))
        index = np.array(drone[part]["index"])
        assert index.size % 3 == 0 and index.min() >= 0 and index.max() < len(points)


def test_bad_requests(serve, tmp_path):
    short_training(tmp_path)
    _, get, _ = serve(tmp_path)
    bad = (
        "api/recording",
        "api/recording?index=9",
        "api/course?index=0",
        "api/course?index=0&course=3",
        "api/course?index=0&course=-1",
        "api/recording?index=-1",
        "api/drone?index=x",
        "api/poll?known=a",
    )
    for address in bad:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(address)
        assert error.value.code == 400, address


def test_one_recording_file_and_a_folder_not_there_yet(serve, tmp_path):
    short_training(tmp_path / "run")
    file = recording_files(tmp_path / "run" / LAPS_FOLDER)[1]
    _, _, get = serve(file)
    data = get("api/poll")
    assert data["keys"] == [":512"] and data["positions"] == [512.0]
    assert get("api/log")["steps"] == []
    assert get("api/course?index=0&course=0")["name"] == SHORT.name
    # A training that has not made its folder yet: nothing, until it does.
    _, _, later = serve(tmp_path / "later")
    assert later("api/poll")["keys"] == []
    shutil.copytree(tmp_path / "run", tmp_path / "later")
    assert len(later("api/poll")["keys"]) == 3


# ---- starting it ---------------------------------------------------------------------


def test_the_viewer_opens_in_the_browser_and_stops(tmp_path, monkeypatch, capsys):
    import tribuna.__main__ as viewer

    opened = []
    stopped = []
    real_start = viewer.start

    def spy_start(path, port):
        server, url = real_start(path, port)
        original = server.shutdown
        server.shutdown = lambda: (stopped.append(True), original())
        return server, url

    def interrupted(_seconds):
        raise KeyboardInterrupt  # Ctrl+C

    windows = []
    monkeypatch.setattr(viewer, "start", spy_start)
    monkeypatch.setattr(viewer.webbrowser, "open", opened.append)
    monkeypatch.setattr(viewer.time, "sleep", interrupted)
    monkeypatch.setattr(viewer, "window", lambda url: windows.append(url) or True)
    viewer.main([str(tmp_path), "--browser"])
    assert len(opened) == 1 and opened[0].startswith("http://127.0.0.1:")
    assert stopped == [True] and windows == []  # no window asked for
    viewer.main([str(tmp_path)])  # a window, closed: no browser
    assert len(windows) == 1 and len(opened) == 1 and stopped == [True, True]
    monkeypatch.undo()
    monkeypatch.setattr(viewer, "start", spy_start)
    monkeypatch.setattr(viewer.webbrowser, "open", opened.append)
    monkeypatch.setattr(viewer.time, "sleep", interrupted)
    stopped.clear()
    opened.clear()
    # Without pywebview, the browser too.
    monkeypatch.setitem(sys.modules, "webview", None)
    assert viewer.window("http://127.0.0.1:1/") is False
    viewer.main([str(tmp_path / "not yet")])
    assert len(opened) == 1 and stopped == [True]
    out = capsys.readouterr().out
    assert "does not exist yet" in out and "No pywebview" in out
    with pytest.raises(SystemExit):
        viewer.main([str(tmp_path / "missing.json")])


def test_a_window_that_cannot_open_falls_back(monkeypatch):
    import types

    import tribuna.__main__ as viewer

    made = []

    def create_window(*args, **kwargs):
        made.append((args, kwargs))

    def fail():
        raise RuntimeError("no web engine")

    fake = types.SimpleNamespace(create_window=create_window, start=fail)
    monkeypatch.setitem(sys.modules, "webview", fake)
    assert viewer.window("http://127.0.0.1:1/") is False
    assert made[0][0] == (viewer.TITLE, "http://127.0.0.1:1/")
    fake.start = lambda: None  # the window opens, and is closed
    assert viewer.window("http://127.0.0.1:1/") is True


# ---- the page's playback, in JavaScript ----------------------------------------------


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node (nodejs.org)")
def test_the_playback_in_javascript():
    files = sorted(str(p) for p in (ROOT / "tests" / "web").glob("*.test.mjs"))
    assert files
    done = subprocess.run(
        ["node", "--test", *files],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    assert "# fail 0" in done.stdout
