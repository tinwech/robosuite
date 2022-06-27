from turtle import width
from robosuite.models import MujocoWorldBase
from robosuite.models.tasks import Task
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject
from mujoco_py import MjSim, MjViewer
from robosuite.devices import Keyboard
from robosuite.renderers.mujoco.mujoco_py_renderer import MujocoPyRenderer
from robosuite.utils.input_utils import input2action
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import xml_path_completion
from robosuite.models.objects.xml_objects import BreadObject
from robosuite.models.objects.xml_objects import BreadVisualObject
from robosuite.utils.placement_samplers import SequentialCompositeSampler, UniformRandomSampler
from robosuite.models.objects import * 
#import cv2

# mods = dir()                                                              # lists current modules
# visualObjs  = [ item for item in mods if 'VisualObject' in item ]         # keep XXXVisualObject
# objs        = [ item.replace('Visual','') for item in visualObjs ]        # remove Visual
# print(visualObjs)
# print(objs)
# world = MujocoWorldBase()
class SpoonObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__("object.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

robot = Panda()
gripper = gripper_factory('PandaGripper')
robot.add_gripper(gripper)
robot.set_base_xpos([0, 0, 0])
# world.merge(robot)

table = TableArena()
table.set_origin([0.8, 0, 0])
# world.merge(table)
world = Task(table, robot)

bread = BreadObject('bread')
world.merge_assets(bread)
bread = bread.get_obj()
bread.set('pos', '1.0 0 2.0')
world.worldbody.append(bread)
# world = Task(table, robot, [bread])

# spoon = SpoonObject('spoon')
# world.merge_assets(spoon)
# spoon = spoon.get_obj()
# spoon.set('pos', '1.0 0 1.0')
# world.worldbody.append(spoon)

# sphere = BallObject(
#     name="sphere",
#     size=[0.04],
#     rgba=[0, 0.5, 0.5, 1]).get_obj()
# sphere.set('pos', '1.0 0 1.0')
# world.worldbody.append(sphere)

# world = Task(table, robot, [bread, sphere])

# print(world.get_xml())
# print(world.mujoco_objects)
# print(world.mujoco_arena)
# print(world.mujoco_robots)

# print(world.get_element_names(world.worldbody, 'geom'))

model = world.get_model(mode="mujoco_py")
sim = MjSim(model)
# print(dir(sim))
# print(dir(sim.model))
# print(sim.model.site_pos)
# print(sim.model.site_names)
# print(sim.data.get_joint_qpos('bread_joint0'))


# for obj in world.mujoco_objects:
#   obj.set_sites_visibility(sim=sim, visible=True)

# viewer = MjViewer(sim)
viewer = MujocoPyRenderer(sim)
viewer.viewer.vopt.geomgroup[0] = 1 
viewer.viewer.vopt.geomgroup[1] = 1 
# viewer.vopt.geomgroup[0] = 0 # disable visualization of collision mesh

for i in range(10000):
  sim.data.ctrl[:] = 0
  sim.step()
  # res, d = sim.render(255, 255, camera_name='frontview', depth=True)
  viewer.render()