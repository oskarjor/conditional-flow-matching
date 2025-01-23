from matplotlib import rc

rc("animation", html="jshtml")
import matplotlib.pyplot as plt
from matplotlib import animation

import copy
import numpy as np
import torch
from torchdyn.core import NeuralODE
from torchvision.utils import save_image
from pathlib import Path
import os
from absl import flags

from torchcfm.models.unet.unet import UNetModelWrapper

FLAGS = flags.FLAGS

flags.DEFINE_string("model", "otcfm", help="flow matching model name")
flags.DEFINE_string(
    "model_path",
    Path.cwd()
    / "results"
    / FLAGS.model
    / f"{FLAGS.model}_cifar10_weights_step_400000.pt",
    help="flow matching model path",
)

flags.DEFINE_string("output_dir", Path.cwd() / 'assets' / FLAGS.model / 'images', help="output directory")
flags.DEFINE_integer("num_images", 16, help="number of images to generate")
