import onnxruntime
from typing import Any, Dict, List

class ONNXModule:
    """
    A class that encapsulates an ONNX model for inference.

    Attributes:
        weight (str): Path to the ONNX model file.
        session (onnxruntime.InferenceSession): The ONNX Runtime inference session for the model.

    Methods:
        __init__(self, weight: str): Initializes the EnhancedONNXModule instance.
        __init_engine(self): Initializes the ONNX Runtime inference engine.
        __call__(self, inputs: Dict[str, Any]): Performs inference on the given inputs.
    """

    def __init__(self, weight: str) -> None:
        """
        Initializes the EnhancedONNXModule with the given ONNX model.

        Parameters:
            weight (str): The path to the ONNX model file.
        """
        self.weight = weight
        self.session: onnxruntime.InferenceSession = self.__init_engine()

    def __init_engine(self) -> onnxruntime.InferenceSession:
        """
        Initializes the ONNX Runtime inference engine with the model.

        Returns:
            onnxruntime.InferenceSession: The initialized inference session.
        """
        try:
            session = onnxruntime.InferenceSession(self.weight, providers=['CPUExecutionProvider'])
            return session
        except onnxruntime.OnnxRuntimeException as e:
            raise RuntimeError(f"Failed to initialize ONNX Runtime session: {e}")

    def __call__(self, inputs: Dict[str, Any]) -> List[Any]:
        """
        Performs inference on the provided inputs using the ONNX model.

        Parameters:
            inputs (Dict[str, Any]): The inputs for the model inference. Keys are input names, and values are input tensors.

        Returns:
            List[Any]: The outputs from the model inference.
        """
        try:
            outputs = self.session.run(None, inputs)
            return outputs
        except Exception as e:
            raise RuntimeError(f"Inference failed: {e}")
