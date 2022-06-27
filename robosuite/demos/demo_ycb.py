from turtle import width
import robosuite
from robosuite.models import MujocoWorldBase
from robosuite.models.tasks import Task
from robosuite.models.robots import Panda
from robosuite.models.grippers import gripper_factory
from robosuite.models.arenas import TableArena
from robosuite.models.arenas import BinsArena
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
from robosuite import load_controller_config
from robosuite.robots import SingleArm
from robosuite.wrappers import VisualizationWrapper
from robosuite.models.robots import create_robot
from robosuite.environments.base import MujocoEnv

controller_config = load_controller_config(default_controller='OSC_POSE')

class SpoonObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__("spoon.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class MugObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__("mug.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class ForkObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__("fork.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class BananaObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__("banana.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)

class LegoObject(MujocoXMLObject):
    def __init__(self, name):
        super().__init__("lego.xml",
                         name=name, joints=[dict(type="free", damping="0.0005")],
                         obj_type="all", duplicate_collision_geoms=True)



# robot = SingleArm('Pandas', controller_config=controller_config)
# print(robot)
robot = Panda()
gripper = gripper_factory('PandaGripper')
robot.add_gripper(gripper)
robot.set_base_xpos([0, 0, 0])

table = TableArena()
table.set_origin([0.8, 0, 0])
world = Task(table, robot)

bread = BreadObject('bread')
world.merge_assets(bread)
bread = bread.get_obj()
bread.set('pos', '1.0 0.3 2.0')
world.worldbody.append(bread)
# world = Task(table, robot, [bread])

spoon = SpoonObject('spoon')
world.merge_assets(spoon)
spoon = spoon.get_obj()
spoon.set('pos', '0.8 -0.2 2')
world.worldbody.append(spoon)

fork = ForkObject('fork')
world.merge_assets(fork)
fork = fork.get_obj()
fork.set('pos', '0.8 -0.25 2')
world.worldbody.append(fork)

banana = BananaObject('banana')
world.merge_assets(banana)
banana = banana.get_obj()
banana.set('pos', '1 -0.2 2')
world.worldbody.append(banana)

mug = MugObject('mug')
world.merge_assets(mug)
mug = mug.get_obj()
mug.set('pos', '0.8 0.15 2')
world.worldbody.append(mug)

lego = LegoObject('lego')
world.merge_assets(lego)
lego = lego.get_obj()
lego.set('pos', '1.1 -0.1 3')
world.worldbody.append(lego)

model = world.get_model(mode="mujoco_py")
sim = MjSim(model)

# viewer = MjViewer(sim)
viewer = MujocoPyRenderer(sim)
viewer.viewer.vopt.geomgroup[0] = 0 

device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
viewer.add_keypress_callback("any", device.on_press)
viewer.add_keyup_callback("any", device.on_release)
viewer.add_keyrepeat_callback("any", device.on_press)


while True:
    # action, grasp = input2action(
    #     device=device, robot=robot, active_arm='right', env_configuration=None
    # )

    # if action is None:
    #     break
    # action = action[: sim.action_dim]
    # obs, reward, done, info = sim.step(action)

    sim.data.ctrl[:] = 0
    sim.step()
    viewer.render()