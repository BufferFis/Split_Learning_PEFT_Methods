import torch
import numpy as np

def serialize_tensor(tensor):
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    shape = list(tensor.shape)
    dtype_str = str(tensor.dtype).split('.')[-1]
    tensor_bytes = tensor.numpy().tobytes()
    
    return shape, dtype_str, tensor_bytes

def deserialize_tensor(shape, dtype_str, tensor_bytes, requires_grad=False):
    dtype_map = {
        'float32': np.float32,
        'float64': np.float64,
        'int32': np.int32,
        'int64': np.int64,
        'uint8': np.uint8,
        'bool': np.bool_
    }
    
    np_dtype = dtype_map.get(dtype_str, np.float32)
    numpy_tensor = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(shape)
    torch_tensor = torch.from_numpy(numpy_tensor).clone()
    
    if requires_grad:
        torch_tensor.requires_grad_(True)
    
    return torch_tensor