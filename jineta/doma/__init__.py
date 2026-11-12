"""Doma: the learning algorithm (where a horse is trained).

Neural networks, a stochastic policy and PPO, in NumPy. Generic: it knows nothing
about drones; it trains on any vectorised environment with the shape of
`picadero.env.CourseEnv` (the shape of Gymnasium's interface). It imports only
NumPy, and nothing of the engine imports it.
"""
