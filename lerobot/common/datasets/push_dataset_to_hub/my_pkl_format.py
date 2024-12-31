import gc
import tqdm
import numpy as np
from numpy.typing import ArrayLike
import argparse
import pickle
from pathlib import Path
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
import torch
import einops
import shutil
from PIL import Image as PILImage
from datasets import Dataset, Sequence, Features, Image, Value, Features, Array3D
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.common.datasets.utils import hf_transform_to_torch
from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    calculate_episode_data_index,
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)

XELA_FREQ = 100
RGB_FREQ = 10
FRANKA_FREQ = 10
ALLEGRO_FREQ = 10


def read_xela_data(
    xela_array: np.ndarray,
    resampling_timestamps: np.ndarray,
    nominal_freq: int,
    smooth_data: bool = False,
    xela_force_data: ArrayLike | None = None,
):
    num_frames = int((resampling_timestamps[-1] - resampling_timestamps[0]) * nominal_freq)
    num_of_sensors = xela_array.shape[1]
    xela_processed_array = np.zeros((num_frames, num_of_sensors, 4))
    xela_timestamps = xela_array[:, 0, 0]
    xela_mag = xela_array[:, :, 1:4]
    xela_mag_interpolated = CubicSpline(xela_timestamps, xela_mag, axis=0)(resampling_timestamps)
    if smooth_data:
        xela_mag_interpolated = savgol_filter(xela_mag_interpolated, nominal_freq // 3, 3, axis=0)
    xela_processed_array[:, :, 0] = resampling_timestamps[:, None]
    xela_processed_array[:, :, 1:4] = xela_mag_interpolated

    if xela_force_data is not None:
        xela_force_array = np.zeros((num_frames, xela_force_data.shape[-2], 7))
        xela_force_mag = xela_force_data[:, :, 1:4]
        xela_force_mag_interpolated = CubicSpline(xela_timestamps, xela_force_mag, axis=0)(
            resampling_timestamps
        )
        if smooth_data:
            xela_force_mag_interpolated = savgol_filter(
                xela_force_mag_interpolated, nominal_freq // 3, 3, axis=0
            )
        xela_force_array[:, :, 0] = resampling_timestamps[:, None]
        xela_force_array[:, :, 1:4] = xela_force_mag_interpolated
        return xela_processed_array, xela_force_array

    return xela_processed_array


def read_allegro_joint_data(
    joint_data: np.ndarray,
    resampling_timestamps: np.ndarray,
    nominal_freq: int,
    smooth_data: bool = False,
):
    joint_timestamps = joint_data[:, 0]
    joint_angles = joint_data[:, 1:17]
    joint_effort = joint_data[:, 17:]

    joint_angles_interpolated = CubicSpline(joint_timestamps, joint_angles, axis=0)(resampling_timestamps)
    joint_effort_interpolated = CubicSpline(joint_timestamps, joint_effort, axis=0)(resampling_timestamps)

    if smooth_data:
        joint_angles_interpolated = savgol_filter(joint_angles_interpolated, nominal_freq // 3, 3, axis=0)
        joint_effort_interpolated = savgol_filter(joint_effort_interpolated, nominal_freq // 3, 3, axis=0)

    return joint_angles_interpolated, joint_effort_interpolated


def read_franka_joint_data(
    joint_data: np.ndarray,
    resampling_timestamps: np.ndarray,
    nominal_freq: int,
    smooth_data: bool = False,
):
    joint_timestamps = joint_data[:, 0]
    joint_angles = joint_data[:, 1:8]
    joint_effort = joint_data[:, 8:]

    joint_angles_interpolated = CubicSpline(joint_timestamps, joint_angles, axis=0)(resampling_timestamps)
    joint_effort_interpolated = CubicSpline(joint_timestamps, joint_effort, axis=0)(resampling_timestamps)

    if smooth_data:
        joint_angles_interpolated = savgol_filter(joint_angles_interpolated, nominal_freq // 3, 3, axis=0)
        joint_effort_interpolated = savgol_filter(joint_effort_interpolated, nominal_freq // 3, 3, axis=0)

    return joint_angles_interpolated, joint_effort_interpolated


def load_from_raw(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    episode_folders = list(raw_dir.glob("*"))
    episode_ids = np.arange(len(episode_folders))

    if episodes is not None:
        episode_ids = episode_ids[episodes]

    ep_dicts = []
    for episode_idx in tqdm.tqdm(episode_ids):
        episode_folder = episode_folders[episode_idx]

        xela_observations = pickle.load(open(episode_folder / "xela/data.pkl", "rb"))
        xela_observations = np.asarray(xela_observations)

        allegro_joint_positions = pickle.load(open(episode_folder / "allegro/data.pkl", "rb"))
        allegro_joint_positions = np.asarray(allegro_joint_positions["joint_states"])

        franka_joint_positions = pickle.load(open(episode_folder / "franka/data.pkl", "rb"))
        franka_joint_positions = np.asarray(franka_joint_positions["joint_states"])

        timestamps = []
        timestamps.append(xela_observations[:, 0, 0])
        timestamps.append(allegro_joint_positions[:, 0])
        timestamps.append(franka_joint_positions[:, 0])

        rgb_observation_keys = ["left/color", "right/color", "top/color"]
        for rgb_observation_key in rgb_observation_keys:
            rgb_timestamp = np.loadtxt(episode_folder / rgb_observation_key / "timestamps.txt")
            timestamps.append(rgb_timestamp)

        start_timestamp = max([timestamp[0] for timestamp in timestamps])
        end_timestamp = min([timestamp[-1] for timestamp in timestamps])

        xela_interp_timestamps = np.linspace(
            start_timestamp, end_timestamp, int((end_timestamp - start_timestamp) * XELA_FREQ)
        )
        allegro_interp_timestamps = np.linspace(
            start_timestamp, end_timestamp, int((end_timestamp - start_timestamp) * ALLEGRO_FREQ)
        )
        franka_interp_timestamps = np.linspace(
            start_timestamp, end_timestamp, int((end_timestamp - start_timestamp) * FRANKA_FREQ)
        )
        rgb_interp_timestamps = np.linspace(
            start_timestamp, end_timestamp, int((end_timestamp - start_timestamp) * RGB_FREQ)
        )

        rgb_idxs = {}
        max_episode_length = np.inf
        for rgb_observation_key in rgb_observation_keys:
            rgb_timestamp = np.loadtxt(episode_folder / rgb_observation_key / "timestamps.txt")
            idx = np.searchsorted(rgb_timestamp, rgb_interp_timestamps)
            rgb_idxs[rgb_observation_key] = idx
            max_episode_length = min(max_episode_length, idx.shape[0] * (XELA_FREQ / RGB_FREQ))

        xela_observations = read_xela_data(xela_observations, xela_interp_timestamps, XELA_FREQ)
        max_episode_length = min(max_episode_length, xela_observations.shape[0])

        allegro_joint_positions, allegro_joint_efforts = read_allegro_joint_data(
            allegro_joint_positions, allegro_interp_timestamps, ALLEGRO_FREQ
        )
        max_episode_length = min(
            max_episode_length, allegro_joint_positions.shape[0] * (XELA_FREQ / ALLEGRO_FREQ)
        )

        franka_joint_positions, franka_joint_efforts = read_franka_joint_data(
            franka_joint_positions, franka_interp_timestamps, FRANKA_FREQ
        )
        max_episode_length = min(
            max_episode_length, franka_joint_positions.shape[0] * (XELA_FREQ / FRANKA_FREQ)
        )
        max_episode_length = int(max_episode_length)

        xela_observations = xela_observations[:max_episode_length]
        xela_observations = einops.rearrange(
            xela_observations, "(b t) n c -> b t n c", t=int(XELA_FREQ / RGB_FREQ)
        )
        max_episode_length = xela_observations.shape[0]
        allegro_joint_positions = allegro_joint_positions[: int(max_episode_length)]
        allegro_joint_efforts = allegro_joint_efforts[: int(max_episode_length)]
        franka_joint_positions = franka_joint_positions[: int(max_episode_length)]
        franka_joint_efforts = franka_joint_efforts[: int(max_episode_length)]

        rgb_data = {}
        for rgb_observation_key in rgb_observation_keys:
            rgb_idxs[rgb_observation_key] = rgb_idxs[rgb_observation_key][: int(max_episode_length)]

            rgb_frames = []
            for idx in rgb_idxs[rgb_observation_key]:
                # assert idx < len(
                #     list(episode_folder.glob(f"{rgb_observation_key}/*.jpg"))
                # ), f"idx: {idx}, len: {len(list(episode_folder.glob(f'{rgb_observation_key}/*.jpg')))}"
                image = PILImage.open(episode_folder / rgb_observation_key / f"{idx:06d}.jpg")
                rgb_frames.append(np.asarray(image))
                image.close()

            if video:
                rgb_frames = np.stack(rgb_frames)
                num_frames = rgb_frames.shape[0]
                # save png images in temporary directory
                tmp_imgs_dir = videos_dir / "tmp_images"
                save_images_concurrently(rgb_frames, tmp_imgs_dir)

                # encode images to a mp4 video
                fname = f"{rgb_observation_key}_episode_{episode_idx:06d}.mp4"
                video_path = videos_dir / fname
                encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))

                # clean temporary images directory
                shutil.rmtree(tmp_imgs_dir)

                # store the reference to the video frame
                rgb_data[rgb_observation_key] = [
                    {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
                ]

        episode_data = {
            "observation.tactile": torch.from_numpy(xela_observations[..., 1:]).float(),
            "allegro_joint_positions": torch.from_numpy(allegro_joint_positions).float(),
            "allegro_joint_efforts": torch.from_numpy(allegro_joint_efforts).float(),
            "franka_joint_positions": torch.from_numpy(franka_joint_positions).float(),
            "franka_joint_efforts": torch.from_numpy(franka_joint_efforts).float(),
            "observation.images.top": rgb_data["top/color"],
            "observation.images.left": rgb_data["left/color"],
            "observation.images.right": rgb_data["right/color"],
            "frame_index": torch.arange(0, max_episode_length),
            "timestamp": torch.arange(0, max_episode_length) / RGB_FREQ,
            "episode_index": [int(episode_idx)] * max_episode_length,
            "episode_name": [episode_folder.name] * max_episode_length,
        }
        ep_dicts.append(episode_data)

        gc.collect()

    data_dict = concatenate_episodes(ep_dicts)

    return data_dict


def to_hf_dataset(data_dict, video):
    features = {}
    keys = [key for key in data_dict if "observation.images." in key]
    for key in keys:
        if video:
            features[key] = VideoFrame()
        else:
            features[key] = Image()
    features["observation.tactile"] = Array3D(shape=(10, 368, 3), dtype="float32", id=None)
    for key in [
        "allegro_joint_positions",
        "allegro_joint_efforts",
        "franka_joint_positions",
        "franka_joint_efforts",
    ]:
        print(f"data_dict[{key}].shape: {data_dict[key].shape}")
        features[key] = Sequence(length=data_dict[key].shape[1], feature=Value(dtype="float32", id=None))

    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["index"] = Value(dtype="int64", id=None)
    features["episode_name"] = Value(dtype="string", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int | None = None,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    if fps is None:
        fps = RGB_FREQ
    data_dict = load_from_raw(raw_dir, videos_dir, fps, video, [0, 1], encoding)

    hf_dataset = to_hf_dataset(data_dict, video)
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, required=True)
    args = parser.parse_args()
    main(args)
