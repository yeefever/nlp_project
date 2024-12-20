from bitsandbytes.nn import Linear4bit
import torch.nn as nn
import torch
import bitsandbytes.functional as F



class Portable4BitLinear(Linear4bit):
    def __init__(self, existing_layer: nn.Linear, is_quantized="nf4"):
        assert is_quantized in [
            "fp4",
            "nf4",
        ], "Unsupported quantization type. Choose either 'fp4' or 'nf4'."

        in_features, out_features = (
            existing_layer.in_features,
            existing_layer.out_features,
        )
        has_bias = existing_layer.bias is not None
        Linear4bit.__init__(
            self,
            input_features=in_features,
            output_features=out_features,
            bias=has_bias,
            quant_type=is_quantized,
        )
        self.load_state_dict(existing_layer.state_dict(), strict=False)
        self.original_dtype = existing_layer.weight.dtype
        del existing_layer

        # Quantization type (fp4 or nf4)
        if is_quantized == "fp4":
            self.is_quantized = "fp4"
        else:
            self.is_quantized = "nf4"
        self.has_bias = has_bias

    def dequantize(self) -> nn.Linear:
        """
        Dequantizes the quantized tensor.

        Parameters:
        - A: torch.Tensor, the quantized tensor to be dequantized.
        """
        with torch.no_grad():
            if self.weight.quant_state is None:
                raise ValueError(
                    "Weight quantization state is not initialized. Please quantize before dequantizing."
                )

            # Use the existing quant_state for dequantization
            dequantized_w = F.dequantize_4bit(
                self.weight, quant_state=self.weight.quant_state
            )
            linear = nn.Linear(
                in_features=dequantized_w.shape[1],
                out_features=dequantized_w.shape[0],
                bias=self.has_bias,
            ).to(self.weight.device)
            linear.weight.data = dequantized_w
            if self.has_bias:
                linear.bias.data = self.bias.data

            return linear

    def dequantize_weight_tensor(self) -> torch.Tensor:
        return F.dequantize_4bit(self.weight, quant_state=self.weight.quant_state)