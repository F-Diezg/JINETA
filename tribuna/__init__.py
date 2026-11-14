"""Tribuna: the viewer of recorded flights (phase 13).

The *tribuna* is the grandstand, where the public watches the riding. It sits
outside `jineta/`: it never runs the physics, it only reads what a training or a
replay wrote (`jineta.picadero.recording`: laps, and the training's log) and
shows it in 3D.

It is split so that almost all of it is plain NumPy, tested everywhere, and only
the drawing needs PyVista:

    run.py       the run folder read as it grows: recordings and the log
    geometry.py  the course and the drone as meshes (points, faces, lines), NED
    motion.py    where a recorded drone is at any moment (interpolated)
    scene.py     the PyVista window (phase 13, step 4)

Dependency rule: tribuna imports jineta (to read recordings and courses, and for
quaternions); nothing in jineta imports tribuna.
"""
