# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
Run all steps with default settings:
$ python3 train.py

Change the model used for each step:
$ python3 train.py --actor-model 350m --reward-model 1.3b

Change the ZeRO stage used for actor/reward models:
$ python3 train.py --actor-zero-stage 1 --reward-zero-stage 3

Run a subset of the steps:
$ python3 train.py --step 1 2

Note: Step 3 relies on models trained in Steps 1 & 2. If you have already
trained these models, you can run just Step 3 and select which models from
Steps 1 & 2 to use. For example, let's train models for Steps 1 & 2 using
125m and 350m models:
$ python3 train.py --step 1 2 --actor-model 125m --reward-model 125m
$ python3 train.py --step 1 2 --actor-model 350m --reward-model 350m

Now we can run Step 3 with any combination of these models:
$ python3 train.py --step 3 --actor-model 125m --reward-model 350m
$ python3 train.py --step 3 --actor-model 350m --reward-model 125m
"""

import argparse
import warnings
import subprocess
import os
import datetime
import time
import torch

from training.utils.data.data_utils import get_raw_dataset
from transformers import AutoTokenizer


def main():
    ds = get_raw_dataset('SJ/CNFAQ', './', 1, 1)
    train_dataset = ds.get_train_data()
    print(train_dataset)


if __name__ == "__main__":
    main()
