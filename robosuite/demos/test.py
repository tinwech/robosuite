"""
Record video of agent episodes with the imageio library.
This script uses offscreen rendering.

Example:
    $ python demo_video_recording.py --environment Lift --robots Panda
"""

import argparse
import colorsys
import imageio
import numpy as np
import matplotlib.cm as cm

import robosuite.utils.macros as macros
from robosuite import make


def randomize_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.5
    hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=20)
    np.random.shuffle(colors)
    return colors


def segmentation_to_rgb(seg_im, random_colors=False):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    seg_im = np.mod(seg_im, 256)

    if random_colors:
        colors = randomize_colors(N=256, bright=True)
        return (255.0 * colors[seg_im]).astype(np.uint8)
    else:
        # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
        rstate = np.random.RandomState(seed=8)
        inds = np.arange(256)
        rstate.shuffle(inds)

        # use @inds to map each geom ID to a color
        return (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]

# Set the image convention to opencv so that the images are automatically rendered "right side up" when using imageio
# (which uses opencv convention)
macros.IMAGE_CONVENTION = "opencv"

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="frontview_seg_instance_video.mp4")
    parser.add_argument("--video_path2", type=str, default="agentview_seg_instance_video.mp4")
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--skip_frame", type=int, default=1)
    parser.add_argument("--depth", type=bool, default=True)
    parser.add_argument("--segmentation-level", type=str, default="class", help="instance, class, or element")
    parser.add_argument("--random-colors", action="store_true", help="Radnomize segmentation colors")

    args = parser.parse_args()

    segmentation_level = args.segmentation_level  # Options are {instance, class, element}
    
    

    # Choose camera
    camera = ["frontview", "agentview"]

    # initialize an environment with offscreen renderer
    env = make(
        "Stack",
        "Panda",
        has_renderer=False,
        ignore_done=True,
        use_camera_obs=True,
        use_object_obs=False,
        camera_names=camera,
        #camera_depths=True,
        camera_segmentations=segmentation_level,
        camera_heights=512,
        camera_widths=512,
    )


    obs = env.reset()
    ndim = env.action_dim

    # create a video writer with imageio
    writer = imageio.get_writer(args.video_path, fps=20)
    writer2 = imageio.get_writer(args.video_path2, fps=20)

    frames = []
    for i in range(args.timesteps):

        # run a uniformly random agent
        action = 0.5 * np.random.randn(ndim)
        # print(action)
        obs, reward, done, info = env.step(action)
        # print(obs)

        # dump a frame from every K frames
        '''if i % args.skip_frame == 0:
            frame = obs[args.camera + "_image"]
            writer.append_data(frame)
            print("Saving frame #{}".format(i))'''

        
        for cam in camera:
            video_img = obs[f"{cam}_segmentation_{segmentation_level}"].squeeze(-1)[::-1]
            video_img = segmentation_to_rgb(video_img, args.random_colors)
            if cam == "frontview":
                writer.append_data(video_img)
            else:
                writer2.append_data(video_img)
        print("Saving frame #{}".format(i))

        if done:
            break

    writer.close()
