"""
## Information

- File Name: preprocess.py
- Author: Selene
- Date of Creation: 2023.03.20
- Date of Last Modification: 2023.05.17 (TODO: Update this)
- Python Version: 3.9.13
- License: GNU GPL v3.0
"""

import os
import random
from typing import Tuple

import numpy as np
from GeoPrivacy import mechanism


class Preprocess:
    """_summary_

    Returns:
        _type_: _description_
    """

    @staticmethod
    def _normalize_trajectory(
        lat: np.ndarray, lng: np.ndarray, scale: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        traj_range = np.max([np.ptp(lat), np.ptp(lng)])
        # assert traj_range > 0
        lat = scale * (lat - np.min(lat)) / traj_range
        lng = scale * (lng - np.min(lng)) / traj_range
        return lat, lng

    @staticmethod
    def _normalize(array: np.ndarray, scale: float = 1.0) -> np.ndarray:
        # assert np.ptp(array) > 0
        return scale * (array - np.min(array)) / np.ptp(array)

    @staticmethod
    def _load_trajectory(file: str) -> np.ndarray:
        data = np.loadtxt(
            file,
            dtype=str,
            delimiter=",",
            skiprows=6,
            usecols=(0, 1, 3, 4),
        )
        data = np.asarray(data, dtype=float)

        lat, lng = Preprocess._normalize_trajectory(data[:, 0], data[:, 1])
        alt = Preprocess._normalize(data[:, 2])
        time = Preprocess._normalize(data[:, 3])

        return np.column_stack((lat, lng, alt, time))

    @staticmethod
    def _save_trajectory(file: str, data: np.ndarray) -> None:
        np.savetxt(file, data, delimiter=",")

    @staticmethod
    def _preprocess(input_file: str, output_file: str) -> None:
        data = Preprocess._load_trajectory(input_file)
        Preprocess._save_trajectory(output_file, data)

    @staticmethod
    def _find_traj_files(path: str, end: str = ".csv") -> list:
        traj_files = []
        for file_name in os.listdir(path):
            file_name: str = file_name
            file_path = os.path.join(path, file_name)
            if os.path.isfile(file_path):
                if file_name.endswith(end):
                    traj_files.append(file_path)
            elif os.path.isdir(file_path):
                traj_files.extend(Preprocess._find_traj_files(file_path))
        return traj_files

    @staticmethod
    def run(src_path: str, dest_path: str) -> None:
        """_summary_

        Args:
            src_path (str): _description_
            dest_path (str): _description_
        """
        traj_files = Preprocess._find_traj_files(src_path)
        for idx, file_name in enumerate(traj_files):
            print(f"Processing file {idx + 1} of {len(traj_files)}")
            output_file = os.path.join(
                dest_path,
                os.path.relpath(file_name, src_path).replace(".plt", ".csv"),
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            Preprocess._preprocess(file_name, output_file)

    @staticmethod
    def _noise(batch_size: int):
        noise = np.zeros((batch_size, 2))
        for i in range(batch_size):
            noise[i, :] = mechanism.random_laplace_noise(
                6, seed=random.randint(0, 100000)
            )
        return noise

    @staticmethod
    def _preserve(input_file: str, output_file: str) -> None:
        data = np.loadtxt(
            input_file,
            dtype=str,
            delimiter=",",
            skiprows=0,
            usecols=(0, 1, 2, 3),
        )
        data = np.asarray(data, dtype=float)

        noise = Preprocess._noise(data.shape[0])
        lat = data[:, 0] + noise[:, 0]
        lng = data[:, 1] + noise[:, 1]
        alt = data[:, 2]
        time = data[:, 3]

        data = np.column_stack((lat, lng, alt, time))
        np.savetxt(output_file, data, delimiter=",")

    @staticmethod
    def preserve(src_path: str, dest_path: str) -> None:
        """_summary_

        Args:
            src_path (str): _description_
            dest_path (str): _description_
        """
        traj_files = Preprocess._find_traj_files(src_path, end=".csv")
        for idx, file_name in enumerate(traj_files):
            print(f"Processing file {idx + 1} of {len(traj_files)}")
            output_file = os.path.join(
                dest_path,
                os.path.relpath(file_name, src_path),
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            Preprocess._preserve(file_name, output_file)

    @staticmethod
    def info(src_path: str) -> None:
        """_summary_

        Args:
            src_path (str): _description_
        """
        traj_files = Preprocess._find_traj_files(src_path)
        print(f"Number of files: {len(traj_files)}")
        shapes = []
        for _, file_name in enumerate(traj_files):
            data = Preprocess._load_trajectory(file_name)
            shapes.append(data.shape[0])
        # get the chart of the number of points in each trajectory
        shapes = np.asarray(shapes)

        np.savetxt("shapes.csv", shapes, delimiter=",")
        print(f"min: {np.min(shapes)}")
        print(f"max: {np.max(shapes)}")
        print(f"mean: {np.mean(shapes)}")
        print(f"median: {np.median(shapes)}")
        print(f"std: {np.std(shapes)}")


if __name__ == "__main__":
    Preprocess.preserve(
        "/Users/selene/Desktop/project/data/Geolife/Data/",
        "/Users/selene/Desktop/project/data/Geoind2/Data/",
    )
