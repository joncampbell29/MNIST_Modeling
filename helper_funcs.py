from math import floor

def conv2d_output_shape(h_w, kernel_size, stride=1, pad=0, dilation=1):
    num2tuple = lambda num: num if isinstance(num, tuple) else (num, num)
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    h = floor((h_w[0] + 2*pad[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = floor((h_w[1] + 2*pad[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    
    return h, w

def MaxPool2d_output_shape(h_w, kernel_size, stride=None, pad=0, dilation=1):
    if stride is None:
        stride = kernel_size
    num2tuple = lambda num: num if isinstance(num, tuple) else (num, num)
    h_w, kernel_size, stride, pad, dilation = num2tuple(h_w), \
        num2tuple(kernel_size), num2tuple(stride), num2tuple(pad), num2tuple(dilation)
    h = floor((h_w[0] + 2*pad[0] - dilation[0]*(kernel_size[0]-1) - 1) / stride[0] + 1)
    w = floor((h_w[1] + 2*pad[1] - dilation[1]*(kernel_size[1]-1) - 1) / stride[1] + 1)
    
    return h, w