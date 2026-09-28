from argos.frames.frames_base import Frame

ECI = Frame(name="ECI",
                inertial=True,
                origin="Earth"
                )

BODY = Frame(name="BODY",
                 inertial=False,
                 origin="Spacecraft CG"
                 )

SENSOR = Frame(name="SENSOR",
                   inertial=False,
                   origin="Local Sensor Position"
                   )