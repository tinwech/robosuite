import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.wrappers import VisualizationWrapper
from mujoco_py import MjViewer
from robosuite.devices import Keyboard

if __name__ == "__main__":

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
    env = suite.make(
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
    viewer = MjViewer(env.sim)

    print(env.sim.model.cam_pos)

    while True:
        # Reset the environment
        obs = env.reset()

        # Setup rendering
        cam_id = 0
        num_cam = len(env.sim.model.camera_names)
        viewer.render()
        env.render()

        # Initialize device control
        device.start_control()

        while True:
            # Set active robot
            active_robot = env.robots[0] 
            # Get the newest action
            action, grasp = input2action(
                device=device, robot=active_robot, active_arm='right', env_configuration=None
            )
            # If action is none, then this a reset so we should break
            if action is None:
                break

            action = action[: env.action_dim]
            # Step through the simulation and render
            obs, reward, done, info = env.step(action)
            viewer.render()
            env.render()
