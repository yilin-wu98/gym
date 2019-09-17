# import random
# from mujoco_py import load_model_from_path, MjSim, MjViewer
# import os
#
# model = load_model_from_path("cloth_v0.xml")
# sim = MjSim(model)
# viewer = MjViewer(sim)
# sim_state = sim.get_state()
#
# while True:
#     sim.set_state(sim_state)
#     for i in range(500): # number of iteration steps
#         if i < 250:
#             sim.data.ctrl[:] = 0.1
#         else:
#             sim.data.ctrl[:] = -0.1
#         sim.step()
#         viewer.render()
#     if os.getenv('TESTING') is not None:
#         break
from dm_control import mujoco

# Load a model from an MJCF XML string.
xml_string = """
<mujoco>
  <worldbody>
    # <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1" directional="true" exponent="1" pos="0 0 0.0" specular=".1 .1 .1"/>
 <body name="B3_5" pos="0 0 1">
    <freejoint name="r1"/>
    <composite type="cloth" count="9 9 1" spacing="0.05" flatinertia="0.01">
        <joint kind="main" damping="0.001"/>
        <skin rgba="0.6 1 0.6 1" texcoord="true" inflate="0.005" subgrid="2"/>
        <geom type="capsule" size="0.015 0.01" rgba=".8 .2 .1 1"/>
    </composite>
</body>
  </worldbody>
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(xml_string)

# Render the default camera view as a numpy array of pixels.
pixels = physics.render()
print(pixels)