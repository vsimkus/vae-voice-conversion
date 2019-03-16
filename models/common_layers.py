import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormalizedGatedConv1d(nn.Module):
    """
    Gated convolution similar to WaveNet.
    """
    def __init__(self, input_shape, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(LayerNormalizedGatedConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.build_module(input_shape)

    def build_module(self, input_shape):
        x = torch.zeros(input_shape)

        self.conv = nn.Conv1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=self.padding)
        conv_out = self.conv(x)

        self.conv_ln = nn.LayerNorm(conv_out.shape[1:])

        self.gate = nn.Conv1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=self.padding)
        gate_out = self.gate(x)

        self.gate_ln = nn.LayerNorm(gate_out.shape[1:])

    def forward(self, input):
        conv_out = self.conv_ln(self.conv(input))
        gate_out = self.gate_ln(self.gate(input))
        return torch.tanh(conv_out) * torch.sigmoid(gate_out)

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.conv.reset_parameters()
        self.conv_ln.reset_parameters()
        self.gate.reset_parameters()
        self.gate_ln.reset_parameters()


class LayerNormalizedLReLUConv1d(nn.Module):
    """
    ReLU activated, normalized convolution.
    """
    def __init__(self, input_shape, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(LayerNormalizedLReLUConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.build_module(input_shape)
    
    def build_module(self, input_shape):
        x = torch.zeros(input_shape)

        self.conv = nn.Conv1d(in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                stride=self.stride,
                dilation=self.dilation,
                padding=self.padding)
        out = self.conv(x)

        self.ln = nn.LayerNorm(out.shape[1:])
    
    def forward(self, input):
        out = self.conv(input)
        conv_out = self.ln(out)
        return F.leaky_relu(out, negative_slope=0.02)

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.conv.reset_parameters()
        self.ln.reset_parameters()


class LayerNormalizedGatedTransposeConv1d(nn.Module):
    """
    Gated 1-dimensional convolution transpose.
    """
    def __init__(self, input_shape, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, out_padding=0):
        super(LayerNormalizedGatedTransposeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding=padding
        self.out_padding = out_padding
        self.dilation = dilation

        self.build_module(input_shape)

    def build_module(self, input_shape):
        x = torch.zeros(input_shape)

        self.conv = nn.ConvTranspose1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=self.padding,
                        output_padding=self.out_padding)
        conv_out = self.conv(x)

        self.conv_ln = nn.LayerNorm(conv_out.shape[1:])

        self.gate = nn.ConvTranspose1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=self.padding,
                        output_padding=self.out_padding)

        gate_out = self.conv(x)
        self.gate_ln = nn.LayerNorm(gate_out.shape[1:])

    def forward(self, input):
        conv_out = self.conv(input)
        conv_out = self.conv_ln(conv_out)

        gate_out = self.gate(input)
        gate_out = self.gate_ln(gate_out)

        return torch.tanh(conv_out) * torch.sigmoid(gate_out)

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.conv.reset_parameters()
        self.gate.reset_parameters()
        self.conv_ln.reset_parameters()
        self.gate_ln.reset_parameters()


class LayerNormalizedLReLUTransposeConv1d(nn.Module):
    """
    ReLU activated, 1-dimensional convolution transpose.
    """
    def __init__(self, input_shape, in_channels, out_channels, kernel_size, stride=1, dilation=1, padding=0, out_padding=0):
        super(LayerNormalizedLReLUTransposeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding=padding
        self.out_padding = out_padding
        self.dilation = dilation

        self.build_module(input_shape)

    def build_module(self, input_shape):
        x = torch.zeros(input_shape)

        self.conv = nn.ConvTranspose1d(in_channels=self.in_channels,
                        out_channels=self.out_channels,
                        kernel_size=self.kernel_size,
                        stride=self.stride,
                        dilation=self.dilation,
                        padding=self.padding,
                        output_padding=self.out_padding)
        conv_out = self.conv(x)

        self.ln = nn.LayerNorm(conv_out.shape[1:])

    def forward(self, input):
        conv_out = self.conv(input)
        conv_out = self.ln(conv_out)

        return F.leaky_relu(conv_out, negative_slope=0.02)

    def reset_parameters(self):
        """
        Re-initializes the networks parameters
        """
        self.conv.reset_parameters()
        self.ln.reset_parameters()


class Digital2Analog(nn.Module):
    """
    Converts digital (mu-law encoded) input to continuous input from -1 to 1
    """
    def __init__(self, num_input_quantization_channels):
        super(Digital2Analog, self).__init__()
        self.num_input_quantization_channels = num_input_quantization_channels
        # Dictionary for digital to analog conversion
        self.init_input_embeddings()

    def init_input_embeddings(self):
        self.input_embeddings = nn.Embedding(self.num_input_quantization_channels, 1)

        indices = torch.arange(self.num_input_quantization_channels)
        # Initialize values from -1 to +1 sequentially.
        analog_values = torch.arange(start=-1, end=1, step=1/(self.num_input_quantization_channels/2), requires_grad=False)

        self.input_embeddings.weight.data.put_(indices, analog_values)
        # The dictionary is non-trainable.
        self.input_embeddings.weight.requires_grad = False

    def forward(self, input):
        return self.input_embeddings(input).squeeze(-1)