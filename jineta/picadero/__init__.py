"""The training layer: the environment a policy learns in (observation, action,
reward, end of an episode) and the classical baseline it is measured against.

It sits on top of the engine (`escuadron`, `campo`, `jinete`), which never imports
it. A picadero is the riding school where horse and rider train.
"""
