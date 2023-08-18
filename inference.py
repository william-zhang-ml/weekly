""" Useful utilities for inferencing but not needed for training. """
import tempfile
from typing import List, Tuple
import torch
from torch import nn
import onnxruntime as ort


def assert_siso_onnx_output_same(
    model: nn.Module,
    example_inp: torch.Tensor,
    num_trials: int = 10
) -> None:
    """Test whether a model will produce same output if exported to ONNX.

    Args:
        model (nn.Module): model of interest in eval mode
        example_inp (torch.Tensor): example of valid model input
        num_trials (int): number of times to compare outputs
    """
    assert not model.training, 'must pass model in eval mode'

    # create onnx inference session by exporting to temp file
    with tempfile.TemporaryFile() as file:
        torch.onnx.export(
            model,
            example_inp,
            file,
            input_names=['input'],
            output_names=['output'],
        )
        file.seek(0)
        sess = ort.InferenceSession(file.read())

    # run random tensors through both models
    for _ in range(num_trials):
        inp = torch.rand_like(example_inp)
        with torch.no_grad():
            model_outp = model(inp)
        onnx_outp = sess.run(None, {'input': inp.numpy()})[0]
        torch.testing.assert_allclose(model_outp, onnx_outp)


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
