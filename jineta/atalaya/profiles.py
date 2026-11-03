"""Sensor profiles: parameter sets from open sources, one constant per sensor.

As with the vehicles (`jinete/profiles.py`), a profile is only data and cites where
its numbers come from; the scenario files name the profiles (the table at the end).
The profiles are generic classes, not copies of a specific product.
"""

import math

from jineta.atalaya.radar import RadarParams

# Generic short-range 3D surveillance radar of the kind used to detect small drones
# around a site. Its numbers sit inside two public datasheets:
#   - Echodyne EchoGuard (K band, electronically scanned): track accuracy < 1 deg in
#     azimuth and < 1.5 deg in elevation, 10 Hz track updates, 120 x 80 deg field
#     of view, small quadcopter (Phantom 4 class) detected beyond 1 km.
#     https://www.echodyne.com/media/mksbimrj/ts-echoguard-echodyne_5jl.pdf
#   - Kongsberg DR 100 for C-UAS: range accuracy < 0.5 m, azimuth < 0.5 deg,
#     elevation < 1.0 deg, update rate 0.1-13 Hz, tracking range 2 km for a
#     "small" NATO class 1 target (0.5 m^2 radar cross-section).
#     https://www.kongsberg.com/globalassets/kongsberg/1.-what-we-do/2.-ocean-space/8.-surveillance--monitoring/dr/110-0089315e_datasheet_dr100_c-uas_may26-1.pdf
# Taken as one standard deviation per measurement (the datasheets do not say if
# their "accuracy" is 1 sigma, RMS or a bound, nor if it is per measurement or per
# track): range 1 m (twice the better datasheet), azimuth 0.5 deg, elevation
# 1.0 deg, 10 Hz, 2 km. The field of view is taken as the whole sky (a rotating or
# multi-panel installation); detection probability 1 and no false alarms.
SHORT_RANGE_RADAR = RadarParams(
    range_noise=1.0,
    azimuth_noise=math.radians(0.5),
    elevation_noise=math.radians(1.0),
    rate=10.0,
    max_range=2000.0,
)

# The profiles by the names the scenario files use.
RADAR_PROFILES = {"short_range": SHORT_RANGE_RADAR}
