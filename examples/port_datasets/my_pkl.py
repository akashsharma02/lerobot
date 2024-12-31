from typing import Dict
import tqdm
import einops
import shutil
import pickle
import gc
from pathlib import Path

import numpy as np
import torch
import PIL.Image as PILImage
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline

from datasets import Dataset, Features, Array3D, Sequence, Value
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    save_images_concurrently,
)

XELA_FREQ = 100
RGB_FREQ = 10
FRANKA_FREQ = 10
ALLEGRO_FREQ = 10

PLUG_INSERTION_TASK = "Insert a charger plug into a socket"
PLUG_INSERTION_FEATURES = {
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "joint_states",
        ],
    },
    "allegro.joint_state": {
        "dtype": "float32",
        "shape": (16,),
        "names": [
            "joint_states",
        ],
    },
    "allegro.joint_effort": {
        "dtype": "float32",
        "shape": (16,),
        "names": [
            "joint_efforts",
        ],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "joint_action",
        ],
    },
    "observation.image.top": {
        "dtype": "video",
        "shape": (3, 320, 240),
        "names": [
            "channel",
            "height",
            "width",
        ],
    },
    "observation.image.left": {
        "dtype": "video",
        "shape": (3, 320, 240),
        "names": [
            "channel",
            "height",
            "width",
        ],
    },
    "observation.image.right": {
        "dtype": "video",
        "shape": (3, 320, 240),
        "names": [
            "channel",
            "height",
            "width",
        ],
    },
    "observation.tactile": {
        "dtype": "float32",
        "shape": (10, 368, 3),
        "names": [
            "time",
            "num_sensors",
            "channel",
        ],
    },
}


def build_features(mode: str) -> dict:
    features = PLUG_INSERTION_FEATURES
    if mode == "keypoints":
        features.pop("observation.image")
    else:
        features.pop("observation.environment_state")
        features["observation.image"]["dtype"] = mode

    return features


def read_xela_data(
    xela_array: np.ndarray,
    resampling_timestamps: np.ndarray,
    nominal_freq: int,
    smooth_data: bool = False,
    xela_force_data: np.ndarray | None = None,
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


def load_raw_dataset(
    raw_dir: Path,
    fps: int,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    videos_dir = raw_dir / "videos"
    episode_folders = list(raw_dir.glob("*"))
    episode_ids = np.arange(len(episode_folders))

    if episodes is not None:
        episode_ids = episode_ids[episodes]

    ep_dicts = []
    for episode_idx in tqdm.tqdm(episode_ids):
        episode_folder = episode_folders[episode_idx]

        with open(episode_folder / "xela/data.pkl", "rb") as f:
            xela_observations = pickle.load(f)
        xela_observations = np.asarray(xela_observations)

        with open(episode_folder / "allegro/data.pkl", "rb") as f:
            allegro_joint_positions = pickle.load(f)
        allegro_joint_positions = np.asarray(allegro_joint_positions["joint_states"])

        with open(episode_folder / "franka/data.pkl", "rb") as f:
            franka_joint_positions = pickle.load(f)
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

            rgb_frames_ = np.stack(rgb_frames)
            # num_frames = rgb_frames_.shape[0]
            # print(f"num_frames: {num_frames}")
            # fname = f"{rgb_observation_key}_episode_{episode_idx:06d}.mp4"
            # video_path = videos_dir / fname
            # if not video_path.exists():
            #     # save png images in temporary directory
            #     tmp_imgs_dir = videos_dir / "tmp_images"
            #     save_images_concurrently(rgb_frames_, tmp_imgs_dir)
            #
            #     # encode images to a mp4 video
            #     encode_video_frames(tmp_imgs_dir, video_path, fps, **(encoding or {}))
            #
            #     # clean temporary images directory
            #     shutil.rmtree(tmp_imgs_dir)

            # store the reference to the video frame
            rgb_data[rgb_observation_key] = rgb_frames_
            # rgb_data[rgb_observation_key] = [
            #     {"path": f"videos/{fname}", "timestamp": i / fps} for i in range(num_frames)
            # ]

        actions = np.zeros_like(franka_joint_positions)
        actions[:-1] = franka_joint_positions[1:]

        episode_data = {
            "observation.tactile": torch.from_numpy(xela_observations[..., 1:]).float(),
            "allegro_joint_positions": torch.from_numpy(allegro_joint_positions).float(),
            "allegro_joint_efforts": torch.from_numpy(allegro_joint_efforts).float(),
            "actions": torch.from_numpy(actions).float(),
            "franka_joint_positions": torch.from_numpy(franka_joint_positions).float(),
            "franka_joint_efforts": torch.from_numpy(franka_joint_efforts).float(),
            "observation.images.top": rgb_data["top/color"],
            "observation.images.left": rgb_data["left/color"],
            "observation.images.right": rgb_data["right/color"],
            "frame_index": torch.arange(0, max_episode_length),
            "timestamp": torch.arange(0, max_episode_length) / RGB_FREQ,
            "episode_index": [episode_idx] * max_episode_length,
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
        "actions",
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
    return hf_dataset, features


def calculate_coverage(zarr_data):
    try:
        import pymunk
        from gym_pusht.envs.pusht import PushTEnv, pymunk_to_shapely
    except ModuleNotFoundError as e:
        print("`gym_pusht` is not installed. Please install it with `pip install 'lerobot[gym_pusht]'`")
        raise e

    block_pos = zarr_data["state"][:, 2:4]
    block_angle = zarr_data["state"][:, 4]

    num_frames = len(block_pos)

    coverage = np.zeros((num_frames,))
    # 8 keypoints with 2 coords each
    keypoints = np.zeros((num_frames, 16))

    # Set x, y, theta (in radians)
    goal_pos_angle = np.array([256, 256, np.pi / 4])
    goal_body = PushTEnv.get_goal_pose_body(goal_pos_angle)

    for i in range(num_frames):
        space = pymunk.Space()
        space.gravity = 0, 0
        space.damping = 0

        # Add walls.
        walls = [
            PushTEnv.add_segment(space, (5, 506), (5, 5), 2),
            PushTEnv.add_segment(space, (5, 5), (506, 5), 2),
            PushTEnv.add_segment(space, (506, 5), (506, 506), 2),
            PushTEnv.add_segment(space, (5, 506), (506, 506), 2),
        ]
        space.add(*walls)

        block_body, block_shapes = PushTEnv.add_tee(space, block_pos[i].tolist(), block_angle[i].item())
        goal_geom = pymunk_to_shapely(goal_body, block_body.shapes)
        block_geom = pymunk_to_shapely(block_body, block_body.shapes)
        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage[i] = intersection_area / goal_area
        keypoints[i] = torch.from_numpy(PushTEnv.get_keypoints(block_shapes).flatten())

    return coverage, keypoints


def calculate_success(coverage: float, success_threshold: float):
    return coverage > success_threshold


def calculate_reward(coverage: float, success_threshold: float):
    return np.clip(coverage / success_threshold, 0, 1)


def calculate_episode_data_index(data_dict: Dict) -> Dict[str, torch.Tensor]:
    episode_indices = data_dict["episode_index"]

    current_episode = None
    episode_data_index = {
        "from": [],
        "to": [],
    }
    for idx, ep_idx in enumerate(episode_indices):
        if ep_idx != current_episode:
            episode_data_index["from"].append(idx)
            if current_episode is not None:
                episode_data_index["to"].append(idx)
            current_episode = ep_idx
        else:
            pass
    episode_data_index["to"].append(idx + 1)

    for k in ["from", "to"]:
        episode_data_index[k] = torch.tensor(episode_data_index[k])
    return episode_data_index


def main(raw_dir: Path, repo_id: str, mode: str = "video", push_to_hub: bool = True):
    if mode not in ["video", "image", "keypoints"]:
        raise ValueError(mode)

    if (LEROBOT_HOME / repo_id).exists():
        shutil.rmtree(LEROBOT_HOME / repo_id)

    if not raw_dir.exists():
        print(f"Raw data not found in {raw_dir}.")

    data_dict = load_raw_dataset(raw_dir, 10, [0, 1])

    episode_data_index = calculate_episode_data_index(data_dict)

    # Calculate success and reward based on the overlapping area
    # of the T-object and the T-area.
    # coverage, keypoints = calculate_coverage(zarr_data)
    # success = calculate_success(coverage, success_threshold=0.95)
    # reward = calculate_reward(coverage, success_threshold=0.95)

    # features = build_features(mode)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=10,
        robot_type="franka panda w allegro",
        features=PLUG_INSERTION_FEATURES,
        image_writer_threads=4,
    )
    episodes = range(len(episode_data_index["from"]))
    for ep_idx in episodes:
        from_idx = episode_data_index["from"][ep_idx]
        to_idx = episode_data_index["to"][ep_idx]
        num_frames = to_idx - from_idx

        for frame_idx in range(num_frames):
            i = from_idx + frame_idx
            print(f"data_dict['observation.tactile'][i].shape: {data_dict['observation.tactile'][i].shape}")
            frame = {
                "observation.tactile": data_dict["observation.tactile"][i],
                "allegro.joint_state": data_dict["allegro_joint_positions"][i],
                "allegro.joint_effort": data_dict["allegro_joint_efforts"][i],
                "action": data_dict["actions"][i],
                "observation.state": data_dict["franka_joint_positions"][i],
                "observation.image.top": data_dict["observation.images.top"][i],
                "observation.image.left": data_dict["observation.images.left"][i],
                "observation.image.right": data_dict["observation.images.right"][i],
                "frame_index": data_dict["frame_index"][i],
                "timestamp": data_dict["timestamp"][i],
            }
            dataset.add_frame(frame)

        dataset.save_episode(task=PLUG_INSERTION_TASK)

    dataset.consolidate()

    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    # To try this script, modify the repo id with your own HuggingFace user (e.g cadene/pusht)
    repo_id = "akashsharma02/xela_plug_insertion"

    raw_dir = Path(
        "/home/akashsharma/workspace/datasets/xela/downstream_tasks/insertion_policy/20241219_pkl/"
    )

    # download and load raw dataset, create LeRobotDataset, populate it, push to hub
    main(raw_dir, repo_id=repo_id, mode="video", push_to_hub=False)

    # Uncomment if you want to load the local dataset and explore it
    # dataset = LeRobotDataset(repo_id=repo_id, local_files_only=True)
    # breakpoint()
