from statistics import mode
import numpy as np
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from mujoco_py import MjViewer
from robosuite.devices import Keyboard

import argparse
import colorsys
import imageio
import matplotlib.cm as cm
from PIL import Image
import robosuite.utils.macros as macros
from robosuite import make

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="control_record.mp4")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--skip_frame", type=int, default=4)
    parser.add_argument("--depth", type=bool, default=True)
    parser.add_argument("--segmentation-level", type=str, default="class", help="instance, class, or element")
    parser.add_argument("--random-colors", action="store_true", help="Radnomize segmentation colors")

    args = parser.parse_args()

    # Import controller config for EE IK or OSC (pos/ori)
    controller_name = "OSC_POSE"
    #controller_name = "IK_POSE"

    # Get controller config
    controller_config = load_controller_config(default_controller=controller_name)

    # Create argument configuration
    config = {
        "env_name": 'Lift',
        "robots": 'Panda',
        "controller_configs": controller_config,
    }

    # Create environment
    env = make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        render_camera='frontview',
        ignore_done=True,
        use_camera_obs=False,
        reward_shaping=True,
        control_freq=20,
        hard_reset=False,
    )

    # Wrap this environment in a visualization wrapper
    env = VisualizationWrapper(env, indicator_configs=None)

    # Setup printing options for numbers
    np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

    # initialize device
    device = Keyboard(pos_sensitivity=1, rot_sensitivity=1)
    env.viewer.add_keypress_callback("any", device.on_press)
    env.viewer.add_keyup_callback("any", device.on_release)
    env.viewer.add_keyrepeat_callback("any", device.on_press)
    #viewer = MjViewer(env.sim)

    print(env.sim.model.cam_pos)

    # Reset the environment
    obs = env.reset()
    
    # Setup rendering
    cam_id = 0
    num_cam = len(env.sim.model.camera_names)
    #viewer.render()
    env.render()

    # Initialize device control
    device.start_control()

    action_record = []
    #num_step = args.timesteps
    num_step = 0
    while 1:
        # Set active robot
        active_robot = env.robots[0]
        # Get the newest action
        action, grasp = input2action(
            device=device, robot=active_robot, active_arm='right', env_configuration=None
        )
        # If action is none, then this a reset so we should break
        if action is None:
            break
        
        num_step += 1
        action = action[: env.action_dim]
        action_record.append(action)
        # Step through the simulation and render
        obs, reward, done, info = env.step(action)
        #viewer.render()
        env.render()
        #print(i)

    #print(action_record)

    camera = "frontview"

    env = make(
        **config,
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=camera,
        camera_heights=512,
        camera_widths=512,
        hard_reset=False,
    )

    # initialize an environment with offscreen renderer
    obs = env.reset()

    # create a video writer with imageio
    writer = imageio.get_writer(args.video_path, fps=20)

    frames = []
    for i in range(num_step):

        # run a uniformly random agent
        action = action_record[i]
        obs, reward, done, info = env.step(action)
        # print(obs)

        # dump a frame from every K frames
        if i % args.skip_frame == 0:
            frame = obs[camera + "_image"]
            img = Image.fromarray(frame)
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            frame = np.array(img)
            writer.append_data(frame)
            #print("Saving frame #{}".format(i))

        if done:
            break


    writer.close()

    path = 'spoon_pos.txt'
    f = open(path, 'w')
    print('Random', file=f)
    f.close()

