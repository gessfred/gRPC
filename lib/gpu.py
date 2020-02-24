#!/usr/bin/env python
from __future__ import print_function
import argparse
import torch

torch.zeros(32).to(torch.device('cuda'))