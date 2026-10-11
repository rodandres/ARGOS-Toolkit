from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from argos.frames.frames_base import Frame


RotationMatrix = NDArray[np.float64]
RotationProvider = Callable[[], RotationMatrix]


class TransformsManager:
    """
    Manage rotation transformations between reference frames.

    The manager supports:
    - Static transformations, calculated once and cached.
    - Dynamic transformations, calculated on demand and cached.
    - Inverse transformations, obtained through matrix transposition.
    - Composite transformations, obtained by chaining registered transformations.
    - Explicit invalidation of dynamic transformations.

    Rotation matrices follow the convention:

        v_target = C_target_from_source @ v_source

    Parameters
    ----------
    atol : float, optional
        Absolute tolerance used to validate rotation matrices.
    """

    def __init__(self, atol: float = 1e-8):
        self.atol = atol

        # Registered transformations:
        # (source_id, target_id) -> (source, target, provider, is_static)
        self._transforms: dict[
            tuple[int, int],
            tuple[Frame, Frame, RotationProvider, bool],
        ] = {}

        # Cached rotation matrices.
        # Static matrices survive dynamic invalidation.
        self._static_cache: dict[
            tuple[int, int],
            RotationMatrix,
        ] = {}

        self._dynamic_cache: dict[
            tuple[int, int],
            RotationMatrix,
        ] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_static(
        self,
        source_frame: Frame,
        target_frame: Frame,
        matrix: RotationMatrix,
    ) -> None:
        """
        Register a static rotation transformation.

        Parameters
        ----------
        source_frame : Frame
            Frame in which the input vector is expressed.
        target_frame : Frame
            Frame in which the output vector will be expressed.
        matrix : ndarray, shape (3, 3)
            Rotation matrix satisfying:

                v_target = matrix @ v_source
        """
        matrix = self._validate_matrix(matrix)

        self._register(
            source_frame=source_frame,
            target_frame=target_frame,
            provider=lambda matrix=matrix: matrix,
            is_static=True,
        )

    def register_dynamic(
        self,
        source_frame: Frame,
        target_frame: Frame,
        provider: RotationProvider,
    ) -> None:
        """
        Register a dynamic rotation transformation.

        The provider must return the current rotation matrix when called.

        Dynamic matrices are cached until invalidate_dynamic() is called.
        """
        self._register(
            source_frame=source_frame,
            target_frame=target_frame,
            provider=provider,
            is_static=False,
        )

    def _register(
        self,
        source_frame: Frame,
        target_frame: Frame,
        provider: RotationProvider,
        is_static: bool,
    ) -> None:
        if source_frame is target_frame:
            raise ValueError(
                "Source and target frames must be different when "
                "registering a transformation."
            )

        key = (id(source_frame), id(target_frame))

        # Replace any previous registration for this ordered pair.
        self._transforms[key] = (
            source_frame,
            target_frame,
            provider,
            is_static,
        )

        # Remove previously cached matrices for this pair.
        self._static_cache.pop(key, None)
        self._dynamic_cache.pop(key, None)

        # A changed direct transformation can affect composite results.
        # Composite matrices are not stored separately in this implementation.

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transform_vector(
        self,
        vector: NDArray[np.float64],
        source_frame: Frame,
        target_frame: Frame,
    ) -> NDArray[np.float64]:
        """
        Transform a vector from source_frame to target_frame.

        Parameters
        ----------
        vector : ndarray, shape (3,)
            Vector expressed in source_frame.
        source_frame : Frame
            Frame in which the input vector is expressed.
        target_frame : Frame
            Frame in which the output vector will be expressed.

        Returns
        -------
        ndarray, shape (3,)
            Vector expressed in target_frame.
        """
        matrix = self.get_rotation_matrix(source_frame, target_frame)
        return matrix @ vector

    def get_rotation_matrix(
        self,
        source_frame: Frame,
        target_frame: Frame,
    ) -> RotationMatrix:
        """
        Obtain the rotation matrix from source_frame to target_frame.

        A direct transformation is preferred. If none is registered,
        the manager searches for a path through intermediate frames.

        Returns
        -------
        ndarray, shape (3, 3)
            Rotation matrix satisfying:

                v_target = C_target_from_source @ v_source

        Raises
        ------
        ValueError
            If no transformation path exists between the requested frames.
        """
        if source_frame is target_frame:
            return np.eye(3)

        path = self._find_path(source_frame, target_frame)

        if path is None:
            raise ValueError(
                f"No rotation transformation path exists from "
                f"'{source_frame.name}' to '{target_frame.name}'."
            )

        # If the path contains one edge, the matrix is obtained directly.
        # Otherwise, compose the matrices in path order.
        result = np.eye(3)

        for source, target in path:
            matrix = self._get_direct_matrix(source, target)
            result = matrix @ result

        return result

    def invalidate_dynamic(self) -> None:
        """
        Clear cached dynamic transformations.

        Call this after updating the state variables on which dynamic
        transformations depend, such as spacecraft attitude or time.

        Static transformations remain cached.
        """
        self._dynamic_cache.clear()

    def clear(self) -> None:
        """Remove all registered transformations and cached matrices."""
        self._transforms.clear()
        self._static_cache.clear()
        self._dynamic_cache.clear()

    # ------------------------------------------------------------------
    # Matrix retrieval
    # ------------------------------------------------------------------

    def _get_direct_matrix(
        self,
        source_frame: Frame,
        target_frame: Frame,
    ) -> RotationMatrix:
        """
        Retrieve a direct transformation or its inverse.
        """
        direct_key = (id(source_frame), id(target_frame))

        if direct_key in self._transforms:
            return self._evaluate_registered(direct_key)

        inverse_key = (id(target_frame), id(source_frame))

        if inverse_key in self._transforms:
            inverse_matrix = self._evaluate_registered(inverse_key)
            return inverse_matrix.T

        raise ValueError(
            f"No direct transformation exists from "
            f"'{source_frame.name}' to '{target_frame.name}'."
        )

    def _evaluate_registered(
        self,
        key: tuple[int, int],
    ) -> RotationMatrix:
        """
        Evaluate a registered transformation using its cache.
        """
        source, target, provider, is_static = self._transforms[key]

        cache = (
            self._static_cache
            if is_static
            else self._dynamic_cache
        )

        if key not in cache:
            matrix = self._validate_matrix(provider())
            cache[key] = matrix

        return cache[key]

    # ------------------------------------------------------------------
    # Path search
    # ------------------------------------------------------------------

    def _find_path(
        self,
        source_frame: Frame,
        target_frame: Frame,
    ) -> list[tuple[Frame, Frame]] | None:
        """
        Find a path between frames using breadth-first search.

        Registered transformations can be traversed in either direction.
        """
        queue = deque([(source_frame, [])])
        visited = {id(source_frame)}

        while queue:
            current_frame, path = queue.popleft()

            if current_frame is target_frame:
                return path

            current_id = id(current_frame)

            for key, (
                registered_source,
                registered_target,
                _,
                _,
            ) in self._transforms.items():

                if id(registered_source) == current_id:
                    next_frame = registered_target
                elif id(registered_target) == current_id:
                    next_frame = registered_source
                else:
                    continue

                next_id = id(next_frame)

                if next_id in visited:
                    continue

                visited.add(next_id)

                queue.append(
                    (
                        next_frame,
                        path + [(current_frame, next_frame)],
                    )
                )

        return None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_matrix(self, matrix: Any) -> RotationMatrix:
        """
        Validate a 3x3 proper orthogonal rotation matrix.
        """
        matrix = np.asarray(matrix, dtype=float)        

        if matrix.shape != (3, 3):
            raise ValueError(
                f"Rotation matrix must have shape (3, 3), "
                f"got {matrix.shape}."
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError(
                "Rotation matrix contains NaN or infinite values."
            )

        if not np.allclose(
            matrix.T @ matrix,
            np.eye(3),
            atol=self.atol,
            rtol=0.0,
        ):
            raise ValueError(
                "Rotation matrix is not orthogonal."
            )

        if not np.isclose(
            np.linalg.det(matrix),
            1.0,
            atol=self.atol,
            rtol=0.0,
        ):
            raise ValueError(
                "Rotation matrix determinant must be +1."
            )

        return matrix.copy()