# Make it explicit that we do it the Python 3 way
from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import *

import sys
import torch
import re

from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from typing import Union

# Credit: Ryuichi Yamamoto (https://github.com/r9y9/wavenet_vocoder/blob/1717f145c8f8c0f3f85ccdf346b5209fa2e1c920/train.py#L599)
# Modified by: Ryan Butler (https://github.com/TheButlah)
# workaround for https://github.com/pytorch/pytorch/issues/15716
# the idea is to return outputs and replicas explicitly, so that making pytorch
# not to release the nodes (this is a pytorch bug though)

_output_ref = None
_replicas_ref = None

def data_parallel_workaround(model, *input):
    global _output_ref
    global _replicas_ref
    device_ids = list(range(torch.cuda.device_count()))
    output_device = device_ids[0]
    replicas = torch.nn.parallel.replicate(model, device_ids)
    # input.shape = (num_args, batch, ...)
    inputs = torch.nn.parallel.scatter(input, device_ids)
    # inputs.shape = (num_gpus, num_args, batch/num_gpus, ...)
    replicas = replicas[:len(inputs)]
    outputs = torch.nn.parallel.parallel_apply(replicas, inputs)
    y_hat = torch.nn.parallel.gather(outputs, output_device)
    _output_ref = outputs
    _replicas_ref = replicas
    return y_hat


###### Deal with hparams import that has to be configured at runtime ######
class __HParams:
    """Manages the hyperparams pseudo-module"""
    def __init__(self, path: Union[str, Path]=None):
        """Constructs the hyperparameters from a path to a python module. If
        `path` is None, will raise an AttributeError whenever its attributes
        are accessed. Otherwise, configures self based on `path`."""
        if path is None:
            self._configured = False
        else:
            self.configure(path)

    def __getattr__(self, item):
        if not self.is_configured():
            raise AttributeError("HParams not configured yet. Call self.configure()")
        else:
            return super().__getattr__(item)

    def configure(self, path: Union[str, Path]):
        """Configures hparams by copying over atrributes from a module with the
        given path. Raises an exception if already configured."""
        if self.is_configured():
            raise RuntimeError("Cannot reconfigure hparams!")

        ###### Check for proper path ######
        if not isinstance(path, Path):
            path = Path(path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Could not find hparams file {path}")
        elif path.suffix != ".py":
            raise ValueError("`path` must be a python file")

        ###### Load in attributes from module ######
        m = _import_from_file("hparams", path)

        reg = re.compile(r"^__.+__$")  # Matches magic methods
        for name, value in m.__dict__.items():
            if reg.match(name):
                # Skip builtins
                continue
            if name in self.__dict__:
                # Cannot overwrite already existing attributes
                raise AttributeError(
                    f"module at `path` cannot contain attribute {name} as it "
                    "overwrites an attribute of the same name in utils.hparams")
            # Fair game to copy over the attribute
            self.__setattr__(name, value)

        self._configured = True

    def is_configured(self):
        return self._configured

hparams = __HParams()


def _import_from_file(name, path: Path):
    """Programmatically returns a module object from a filepath"""
    if not Path(path).exists():
        raise FileNotFoundError('"%s" doesn\'t exist!' % path)
    spec = spec_from_file_location(name, path)
    if spec is None:
        raise ValueError('could not load module from "%s"' % path)
    m = module_from_spec(spec)
    spec.loader.exec_module(m)
    return m