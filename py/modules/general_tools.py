
import numpy as np

def _as_3d_array(value: float | np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)

    if arr.ndim == 0:
        return np.full(3, arr)

    if arr.shape == (3,):
        return arr.copy()

    raise ValueError(
        f"{name} Shall be a scalar or a 3D numpy vector, "
        f"but received {arr.shape}"
    )