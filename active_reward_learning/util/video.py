import os
import subprocess

import cv2
import numpy as np


def save_video(rgbs, filename, fps=20.0):
    """
    Writes out a series of RGB arrays as a video with a specified framerate.

    Args:
        ims (list): numpy arrays if the same size
        filename (str): name of the output file (should end on .mp4)
        fps (float): frames per second
    """
    if filename[-4:] == ".gif":
        gif_output = True
        filename_gif = filename
        filename = filename_gif + ".mp4"
    else:
        gif_output = False

    if filename[-4:] != ".mp4":
        print("Warning: filename should end on .mp4, not '{}'".format(filename))

    # write .mp4
    ims = [cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) for rgb in rgbs]
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    (height, width, _) = ims[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    for im in ims:
        writer.write(im)
    writer.release()

    if gif_output:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                filename,
                filename_gif,
            ]
        )
        os.remove(filename)


def record_gym_video(env, policy, filename):
    """
    Rollout a policy in a gym environment and record a video of its performance.

    Args:
        env (gym.Env): an environment that supports rgb_array rendering
        policy: a policy object that implements a get_action method
    """
    obs = env.reset()
    rgb = env.render("rgb_array")
    rgb = np.array(rgb, dtype=np.uint8)
    images = [rgb]
    done = False
    while not done:  # np all for vectorized envs
        a = policy.get_action(obs)
        obs, rew, done, info = env.step(a)
        rgb = env.render("rgb_array")
        rgb = np.array(rgb, dtype=np.uint8)
        images.append(rgb)
    save_video(images, filename)
