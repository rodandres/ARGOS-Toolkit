from argos.frames.frames_base import Frame

ECI_frame = Frame(
    name="ECI",
    inertial=True,
    origin="Earth"
)

BODY_frame = Frame(
    name="BODY",
    inertial=False,
    origin="Spacecraft CG"
)

SENSOR_frame = Frame(
    name="SENSOR",
    inertial=False,
    origin="Local Sensor Position"
)

CR3BP_ROTATING_frame = Frame(
    name="CR3BP_ROTATING",
    inertial=False,
    origin="Earth-Moon Barycenter",
)

NONE_frame = Frame(
    name="NONE",
    inertial=False,
    origin="None"
)