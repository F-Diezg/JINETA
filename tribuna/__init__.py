"""Tribuna: the viewer of recorded flights (phase 13).

The *tribuna* is the grandstand, where the public watches the riding. It sits
outside `jineta/`: it never runs the physics, it only reads what a training or a
replay wrote (`jineta.picadero.recording`: laps, and the training's log) and
shows it in 3D.

It is a web page shown in a desktop window (pywebview) or a browser, served by
a small server on this computer: the Python reads the files and makes the
shapes, the page plays and draws them (three.js):

    run.py       the run folder read as it grows: recordings and the log
    geometry.py  the course and the drone as meshes (points, faces, lines), NED
    server.py    the HTTP server the page asks (recordings, shapes, the log)
    __main__.py  python -m tribuna <folder>: the server and the window
    web/         the page: viewer.js draws, playback.js says what is where
                 (tested with node), vendor/ is three.js (MIT licence)

Dependency rule: tribuna imports jineta (to read recordings and courses, and for
quaternions); nothing in jineta imports tribuna.
"""
