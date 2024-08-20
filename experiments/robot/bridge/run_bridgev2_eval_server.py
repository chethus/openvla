"""
run_bridge_eval.py

Runs a model in a real-world Bridge V2 environment.

Usage:
    # OpenVLA:
    python experiments/robot/bridge/run_bridgev2_eval_server.py --pretrained_checkpoint openvla/openvla-7b
"""

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Union, Any
import draccus

# Append current directory so that interpreter can find experiments.robot
sys.path.append(".")
from experiments.robot.openvla_utils import get_processor
from experiments.robot.robot_utils import (
    get_action,
    get_image_resize_size,
    get_model,
)
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import json_numpy
json_numpy.patch()


@dataclass
class DeployConfig:
    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    host: str = "0.0.0.0"                                               # Host IP Address
    port: int = 8000                                                    # Host Port

    model_family: str = "openvla"                               # Model family
    pretrained_checkpoint: Union[str, Path] = ""                # Pretrained checkpoint path
    load_in_8bit: bool = False                                  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                                  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False                                   # Center crop? (if trained w/ random crop image aug)


class OpenVLAServer:
    def __init__(self, cfg: DeployConfig) -> None:
        assert cfg.pretrained_checkpoint is not None, "cfg.pretrained_checkpoint must not be None!"
        assert not cfg.center_crop, "`center_crop` should be disabled for Bridge evaluations!"

        # [OpenVLA] Set action un-normalization key
        cfg.unnorm_key = "bridge_orig"

        # Load model
        model = get_model(cfg)

        # [OpenVLA] Get Hugging Face processor
        processor = None
        if cfg.model_family == "openvla":
            processor = get_processor(cfg)

        self.cfg = cfg
        self.model = model
        self.processor = processor

    def predict_action(self, payload: Dict[str, Any]) -> str:
        obs, task_label = payload["obs"], payload["instruction"]
        # Query model to get action
        action = get_action(
            self.cfg,
            self.model,
            obs,
            task_label,
            processor=self.processor,
        )
        return JSONResponse(action)
    
    def run(self, host: str = "0.0.0.0", port: int = 8000) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.predict_action)
        uvicorn.run(self.app, host=host, port=port)

@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = OpenVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()