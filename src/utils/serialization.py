from io import BytesIO
from typing import cast
import torch
import numpy as np
import logging 
from flwr.common.typing import NDArray, NDArrays, Parameters

def ndarrays_to_sparse_parameters(ndarrays: NDArrays) -> Parameters:
    """Convert NumPy ndarrays to parameters object."""
    tensors = [ndarray_to_sparse_bytes(ndarray) for ndarray in ndarrays]
    return Parameters(tensors=tensors, tensor_type="numpy.ndarray")


def sparse_parameters_to_ndarrays(parameters: Parameters) -> NDArrays:
    """Convert parameters object to NumPy ndarrays."""
    return [sparse_bytes_to_ndarray(tensor) for tensor in parameters.tensors]


def ndarray_to_sparse_bytes(ndarray: NDArray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    original_shape = ndarray.shape
    original_size = ndarray.size
    if len(ndarray.shape) > 1:
        # We convert our ndarray into a sparse matrix
        if len(ndarray.shape) > 2:
            ndarray = ndarray.reshape((ndarray.shape[0], -1))
        # if not all(len(row) == len(ndarray[0]) for row in ndarray):
        #     raise ValueError("All rows in ndarray must have the same number of columns")
        tensor = torch.tensor(ndarray).to_sparse_csr()
        #sparse_ndarray = csr_matrix(ndarray) #type: ignore
        #dense_tensor_size = tensor.to_dense().numpy().size
        #logging.info(f"Tensor Size: {original_size} CSR Size : {dense_tensor_size} Original Shape: {original_shape}")
        # And send it byutilizing the sparse matrix attributes
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.savez(
            bytes_io,  # type: ignore
            crow_indices=tensor.crow_indices(),
            col_indices=tensor.col_indices(),
            values=tensor.values(),
            reshape_shape = tensor.to_dense().numpy().shape,
            original_shape = original_shape,
            allow_pickle=False,
        )
    else:
        # WARNING: NEVER set allow_pickle to true.
        # Reason: loading pickled data can execute arbitrary code
        # Source: https://numpy.org/doc/stable/reference/generated/numpy.save.html
        np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def sparse_bytes_to_ndarray(tensor: bytes) -> NDArray:
    """Deserialize NumPy ndarray from bytes."""
    bytes_io = BytesIO(tensor)
    # WARNING: NEVER set allow_pickle to true.
    # Reason: loading pickled data can execute arbitrary code
    # Source: https://numpy.org/doc/stable/reference/generated/numpy.load.html
    loader = np.load(bytes_io, allow_pickle=False)  # type: ignore

    if "crow_indices" in loader:
        # We convert our sparse matrix back to a ndarray, using the attributes we sent
        ndarray_deserialized = torch.sparse_csr_tensor(
                crow_indices=torch.from_numpy(loader["crow_indices"]),
                col_indices=torch.from_numpy(loader["col_indices"]),
                values=torch.from_numpy(loader["values"]),
                size = tuple(loader["reshape_shape"]),
            )
        #logging.info(f"Tensor Size: {ndarray_deserialized.to_dense().numpy().size} Original Shape: {ndarray_deserialized.to_dense().numpy().shape}")
        ndarray_deserialized = ndarray_deserialized.to_dense().numpy().reshape(loader["original_shape"])
    else:
        ndarray_deserialized = loader
    return cast(NDArray, ndarray_deserialized)
    