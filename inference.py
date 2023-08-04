""" Useful utilities for inferencing but not needed for training. """
from typing import List, Tuple
import torch
import onnxruntime as ort


class OnnxEngine:
    """Wrapper class for using onnxruntime.InferenceSession. """
    def __init__(self, path: str) -> None:
        self.session = ort.InferenceSession(path)

    def __call__(self, *args: torch.Tensor) -> Tuple[torch.Tensor]:
        outputs = self.session.run(
            self.output_names,
            {name: arg.numpy() for name, arg in zip(self.input_names, args)}
        )
        return tuple(torch.Tensor(outp) for outp in outputs)

    @property
    def num_inputs(self) -> int:
        """ Number of input tensors. """
        return len(self.session.get_inputs())

    @property
    def num_outputs(self) -> int:
        """ Number of output tensors. """
        return len(self.session.get_outputs())

    @property
    def input_names(self) -> List[str]:
        """ Names of graph input nodes. """
        return [elem.name for elem in self.session.get_inputs()]

    @property
    def output_names(self) -> List[str]:
        """ Names of graph output nodes. """
        return [elem.name for elem in self.session.get_outputs()]
