from py.modules.frames.frames_base import FrameBase

ECI = FrameBase(name="ECI",
                inertial=True,
                origin="Earth"
                )

BODY = FrameBase(name="BODY",
                 inertial=False,
                 origin="Spacecraft CG"
                 )

SENSOR = FrameBase(name="SENSOR",
                   inertial=False,
                   origin="Local Sensor Position"
                   )