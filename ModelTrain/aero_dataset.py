# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from typing import Any, List, Union

import numpy as np
import torch

class AeroSDFDataset():
    """
    In-memory Aero SDF Dataset

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    batch_size : int, optional
        Batch size of simulations, by default 64
    invar_keys: List[str], optional
        The input node features to consider. Default includes 'x', 'y', 'sdf', 'Re', 'Ma'
    outvar_keys: List[str], optional
        The output features to consider. Default includes 'p', 'u', 'v'
    normalize_keys List[str], optional
        The features to normalize. Default includes 'p', 'u', 'v'
    force_reload: bool, optional
        If True, forces a reload of the data, by default False.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    verbose: bool, optional
        If True, enables verbose mode, by default False.
    normaliser : Union[Dict[str, Tuple[float, float]], None], optional
        Dictionary with keys 'x', 'y', 'sdf', 'Re', 'Ma', 'p', 'u', 'v'. The values for these keys are two floats corresponding to mean and std `(mean, std)`.
    device : Union[str, torch.device], optional
        Device for datapipe to run place data on, by default "cuda"
    """

    def __init__(
        self,
        data_dir,
        split="train",
        batch_size=64,
        invar_keys=["x", "y", "sdf", 'Re', 'Ma'],
        outvar_keys=["p", "u", "v"],
        normalize_keys=["p", "u", "v"],
        force_reload=False,
        name="dataset",
        verbose=False,
        normaliser: Union[Dict[str, Tuple[float, float]], None] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.split = split
        self.batch_size = batch_size
        self.data_dir = os.path.join(data_dir, self.split)
        self.input_keys = invar_keys
        self.output_keys = outvar_keys

        print(f"Preparing the {split} dataset...")

        all_entries = os.listdir(self.data_dir)

        data_list = [
            os.path.join(self.data_dir, entry)
            for entry in all_entries
            if os.path.isfile(os.path.join(self.data_dir, entry))
        ]

        numbers = []
        for directory in data_list:
            match = re.search(r"\d+", directory)
            if match:
                numbers.append(int(match.group()))

        numbers = [int(n) for n in numbers]

        # sort
        args = np.argsort(numbers)
        self.data_list = [data_list[index] for index in args]
        numbers = [numbers[index] for index in args]

        # create the graphs with edge features
        self.length = min(len(self.data_list), self.num_samples)

        if self.num_samples > self.length:
            raise ValueError(
                f"Number of available {self.split} dataset entries "
                f"({self.length}) is less than the number of samples "
                f"({self.num_samples})"
            )

        self.graphs = []
        for i in range(self.length):
            # create the dgl graph
            file_path = self.data_list[i]
            polydata = read_vtp_file(file_path)
            graph = self._create_dgl_graph(polydata, outvar_keys, dtype=torch.int32)
            self.graphs.append(graph)

        self.graphs = self.add_edge_features()

        if self.split == "train":
            self.node_stats = self._get_node_stats(keys=normalize_keys)
            self.edge_stats = self._get_edge_stats()
        else:
            self.node_stats = load_json("node_stats.json")
            self.edge_stats = load_json("edge_stats.json")

        self.graphs = self.normalize_node()
        self.graphs = self.normalize_edge()

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        return graph

    def __len__(self):
        return self.length

    def add_edge_features(self):
        """
        adds relative displacement & displacement norm as edge features
        """
        for i in range(len(self.graphs)):
            pos = self.graphs[i].ndata["pos"]
            row, col = self.graphs[i].edges()
            disp = torch.tensor(pos[row.long()] - pos[col.long()])
            disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
            self.graphs[i].edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        return self.graphs

    def normalize_node(self):
        """normalizes node features"""
        invar_keys = set(
            [
                key.replace("_mean", "").replace("_std", "")
                for key in self.node_stats.keys()
            ]
        )
        for i in range(len(self.graphs)):
            for key in invar_keys:
                self.graphs[i].ndata[key] = (
                    self.graphs[i].ndata[key] - self.node_stats[key + "_mean"]
                ) / self.node_stats[key + "_std"]

            self.graphs[i].ndata["x"] = torch.cat(
                [self.graphs[i].ndata[key] for key in self.input_keys], dim=-1
            )
            self.graphs[i].ndata["y"] = torch.cat(
                [self.graphs[i].ndata[key] for key in self.output_keys], dim=-1
            )
        return self.graphs

    def normalize_edge(self):
        """normalizes a tensor"""
        for i in range(len(self.graphs)):
            self.graphs[i].edata["x"] = (
                self.graphs[i].edata["x"] - self.edge_stats["edge_mean"]
            ) / self.edge_stats["edge_std"]

        return self.graphs

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar

    def _get_edge_stats(self):
        stats = {
            "edge_mean": 0,
            "edge_meansqr": 0,
        }
        for i in range(self.length):
            stats["edge_mean"] += (
                torch.mean(self.graphs[i].edata["x"], dim=0) / self.length
            )
            stats["edge_meansqr"] += (
                torch.mean(torch.square(self.graphs[i].edata["x"]), dim=0) / self.length
            )
        stats["edge_std"] = torch.sqrt(
            stats["edge_meansqr"] - torch.square(stats["edge_mean"])
        )
        stats.pop("edge_meansqr")

        # save to file
        save_json(stats, "edge_stats.json")
        return stats

    def _get_node_stats(self, keys):
        stats = {}
        for key in keys:
            stats[key + "_mean"] = 0
            stats[key + "_meansqr"] = 0

        for i in range(self.length):
            for key in keys:
                stats[key + "_mean"] += (
                    torch.mean(self.graphs[i].ndata[key], dim=0) / self.length
                )
                stats[key + "_meansqr"] += (
                    torch.mean(torch.square(self.graphs[i].ndata[key]), dim=0)
                    / self.length
                )

        for key in keys:
            stats[key + "_std"] = torch.sqrt(
                stats[key + "_meansqr"] - torch.square(stats[key + "_mean"])
            )
            stats.pop(key + "_meansqr")

        # save to file
        save_json(stats, "node_stats.json")
        return stats

    