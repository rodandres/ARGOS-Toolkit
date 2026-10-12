from collections import deque
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from argos.frames.frames_base import Frame


RotationMatrix = NDArray[np.float64]
TranslationVector = NDArray[np.float64]
TransformMatrix = NDArray[np.float64]

RotationProvider = Callable[[], RotationMatrix]
TranslationProvider = Callable[[], TranslationVector]


class TransformsManager:
    """
    Manage rigid transformations between reference frames.

    The manager supports:

    - Static transformations, calculated once and cached.
    - Dynamic transformations, calculated on demand and cached.
    - Inverse transformations, obtained analytically.
    - Composite transformations through intermediate frames.
    - Rotations and translations between frame origins.
    - Explicit invalidation of dynamic transformations.

    Transformation convention
    --------------------------
    A transformation from source_frame to target_frame satisfies:

        p_target = R_target_from_source @ p_source + t_target_from_source

    where:

        R_target_from_source:
            Rotation matrix transforming vector components.

        t_target_from_source:
            Position of the source origin expressed in the target frame.

    The homogeneous transformation matrix is:

        T = [[R, t],
             [0, 1]]

    Parameters
    ----------
    atol : float, optional
        Absolute tolerance used to validate rotation matrices.
    """

    def __init__(self, atol: float = 1e-8):

        self.atol = atol

        # Registered transformations:
        #
        # (source_id, target_id) ->
        # (source, target, rotation_provider,
        #  translation_provider, is_static)

        self._transforms: dict[
            tuple[int, int],
            tuple[
                Frame,
                Frame,
                RotationProvider,
                TranslationProvider,
                bool,
            ],
        ] = {}

        # Cached static transformations.
        self._static_cache: dict[
            tuple[int, int],
            TransformMatrix,
        ] = {}

        # Cached dynamic transformations.
        self._dynamic_cache: dict[
            tuple[int, int],
            TransformMatrix,
        ] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_static(
        self,
        source_frame: Frame,
        target_frame: Frame,
        matrix: RotationMatrix,
        translation: TranslationVector | None = None,
    ) -> None:
        """
        Register a static rigid transformation.

        Parameters
        ----------
        source_frame : Frame
            Frame in which the input coordinates are expressed.

        target_frame : Frame
            Frame in which the output coordinates will be expressed.

        matrix : ndarray, shape (3, 3)
            Rotation matrix satisfying:

                p_target = matrix @ p_source + translation

        translation : ndarray, shape (3,), optional
            Position of the source origin expressed in the target frame.

            Defaults to zero, meaning that both origins coincide.
        """

        rotation = self._validate_rotation(matrix)

        if translation is None:
            translation = np.zeros(3, dtype=float)

        translation = self._validate_translation(translation)

        # Capture independent copies of the registered data.
        rotation = rotation.copy()
        translation = translation.copy()

        self._register(
            source_frame=source_frame,
            target_frame=target_frame,
            rotation_provider=lambda rotation=rotation: rotation,
            translation_provider=lambda translation=translation: translation,
            is_static=True,
        )

    def register_dynamic(
        self,
        source_frame: Frame,
        target_frame: Frame,
        provider: RotationProvider,
        translation_provider: TranslationProvider | None = None,
    ) -> None:
        """
        Register a dynamic rigid transformation.

        Parameters
        ----------
        source_frame : Frame
            Frame in which the input coordinates are expressed.

        target_frame : Frame
            Frame in which the output coordinates will be expressed.

        provider : RotationProvider
            Callable returning the current rotation matrix.

        translation_provider : TranslationProvider, optional
            Callable returning the current position of the source origin
            expressed in the target frame.

            If omitted, the translation is assumed to be zero.
        """

        if translation_provider is None:
            translation_provider = (
                lambda: np.zeros(3, dtype=float)
            )

        self._register(
            source_frame=source_frame,
            target_frame=target_frame,
            rotation_provider=provider,
            translation_provider=translation_provider,
            is_static=False,
        )

    def _register(
        self,
        source_frame: Frame,
        target_frame: Frame,
        rotation_provider: RotationProvider,
        translation_provider: TranslationProvider,
        is_static: bool,
    ) -> None:
        """
        Register or update a transformation between two reference frames.

        If a transformation already exists in the same direction, it is
        replaced by the new definition.

        If a transformation already exists in the opposite direction,
        registration is rejected to avoid duplicate or conflicting edges.

        Parameters
        ----------
        source_frame : Frame
            Source reference frame.

        target_frame : Frame
            Target reference frame.

        rotation_provider : RotationProvider
            Callable returning the rotation matrix.

        translation_provider : TranslationProvider
            Callable returning the translation vector.

        is_static : bool
            Whether the transformation is static.

        Raises
        ------
        ValueError
            If source and target are the same frame, or if a transformation
            between the same frames already exists in the opposite direction.
        """

        if source_frame is target_frame:
            raise ValueError(
                "Source and target frames must be different when "
                "registering a transformation."
            )

        source_id = id(source_frame)
        target_id = id(target_frame)

        key = (source_id, target_id)
        inverse_key = (target_id, source_id)

        # Reject a second independently registered edge in the opposite
        # direction. The inverse is already computed automatically.
        if inverse_key in self._transforms:
            raise ValueError(
                f"A transformation between '{source_frame.name}' and "
                f"'{target_frame.name}' is already registered in the "
                "opposite direction. The inverse is computed automatically."
            )

        # Register or replace the transformation in the requested direction.
        self._transforms[key] = (
            source_frame,
            target_frame,
            rotation_provider,
            translation_provider,
            is_static,
        )

        # Invalidate cached transformations for this pair.
        self._static_cache.pop(key, None)
        self._dynamic_cache.pop(key, None)

        # Composite transformations are not cached separately.
    
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
        Transform a vector between reference frames.

        Translation is not applied because vectors do not depend on
        the position of the frame origins.
        """

        vector = np.asarray(vector, dtype=float)

        if vector.shape != (3,):
            raise ValueError(
                f"Vector must have shape (3,), got {vector.shape}."
            )

        rotation = self.get_rotation_matrix(
            source_frame,
            target_frame,
        )

        return rotation @ vector

    def transform_point(
        self,
        point: NDArray[np.float64],
        source_frame: Frame,
        target_frame: Frame,
    ) -> NDArray[np.float64]:
        """
        Transform a point between reference frames.

        Both rotation and translation are applied.
        """

        point = np.asarray(point, dtype=float)

        if point.shape != (3,):
            raise ValueError(
                f"Point must have shape (3,), got {point.shape}."
            )

        transform = self.get_transform_matrix(
            source_frame,
            target_frame,
        )

        return (
            transform[:3, :3] @ point
            + transform[:3, 3]
        )

    def get_rotation_matrix(
        self,
        source_frame: Frame,
        target_frame: Frame,
    ) -> RotationMatrix:
        """
        Obtain the rotation matrix from source_frame to target_frame.

        Direct transformations are preferred. If no direct transformation
        exists, the manager searches for a path through intermediate frames.

        Returns
        -------
        ndarray, shape (3, 3)
            Rotation matrix satisfying:

                v_target = R_target_from_source @ v_source
        """

        transform = self.get_transform_matrix(
            source_frame,
            target_frame,
        )

        return transform[:3, :3].copy()

    def get_transform_matrix(
        self,
        source_frame: Frame,
        target_frame: Frame,
    ) -> TransformMatrix:
        """
        Obtain the complete transformation between two reference frames.

        The transformation includes both rotation and translation.

        Returns
        -------
        ndarray, shape (4, 4)
            Homogeneous transformation satisfying:

                p_target = R @ p_source + t

        Raises
        ------
        ValueError
            If no transformation path exists between the requested frames.
        """

        if source_frame is target_frame:
            return np.eye(4, dtype=float)

        path = self._find_path(
            source_frame,
            target_frame,
        )

        if path is None:
            raise ValueError(
                f"No transformation path exists from "
                f"'{source_frame.name}' to '{target_frame.name}'."
            )

        # Composition convention:
        #
        # T_C_from_A = T_C_from_B @ T_B_from_A

        result = np.eye(4, dtype=float)

        for source, target in path:

            transform = self._get_direct_transform(
                source,
                target,
            )

            result = transform @ result

        return result

    def invalidate_dynamic(self) -> None:
        """
        Clear cached dynamic transformations.

        Call this after updating the state variables on which dynamic
        transformations depend, such as spacecraft attitude, position,
        or simulation time.

        Static transformations remain cached.
        """

        self._dynamic_cache.clear()

    def clear(self) -> None:
        """Remove all registered transformations and cached matrices."""

        self._transforms.clear()
        self._static_cache.clear()
        self._dynamic_cache.clear()

    # ------------------------------------------------------------------
    # Transformation retrieval
    # ------------------------------------------------------------------

    def _get_direct_transform(
        self,
        source_frame: Frame,
        target_frame: Frame,
    ) -> TransformMatrix:
        """
        Retrieve a direct transformation or its inverse.
        """

        direct_key = (
            id(source_frame),
            id(target_frame),
        )

        if direct_key in self._transforms:
            return self._evaluate_registered(direct_key)

        inverse_key = (
            id(target_frame),
            id(source_frame),
        )

        if inverse_key in self._transforms:

            transform = self._evaluate_registered(inverse_key)

            rotation = transform[:3, :3]
            translation = transform[:3, 3]

            inverse = np.eye(4, dtype=float)

            inverse_rotation = rotation.T

            inverse[:3, :3] = inverse_rotation
            inverse[:3, 3] = -inverse_rotation @ translation

            return inverse

        raise ValueError(
            f"No direct transformation exists from "
            f"'{source_frame.name}' to '{target_frame.name}'."
        )

    def _evaluate_registered(
        self,
        key: tuple[int, int],
    ) -> TransformMatrix:
        """
        Evaluate a registered transformation using its cache.
        """

        (
            source,
            target,
            rotation_provider,
            translation_provider,
            is_static,
        ) = self._transforms[key]

        cache = (
            self._static_cache
            if is_static
            else self._dynamic_cache
        )

        if key not in cache:

            rotation = self._validate_rotation(
                rotation_provider()
            )

            translation = self._validate_translation(
                translation_provider()
            )

            transform = np.eye(4, dtype=float)

            transform[:3, :3] = rotation
            transform[:3, 3] = translation

            cache[key] = transform

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

            for (
                registered_source,
                registered_target,
                _,
                _,
                _,
            ) in self._transforms.values():

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

    def _validate_rotation(
        self,
        matrix: Any,
    ) -> RotationMatrix:
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

    def _validate_translation(
        self,
        translation: Any,
    ) -> TranslationVector:
        """
        Validate a three-dimensional translation vector.
        """

        translation = np.asarray(
            translation,
            dtype=float,
        )

        if translation.shape != (3,):
            raise ValueError(
                f"Translation must have shape (3,), "
                f"got {translation.shape}."
            )

        if not np.all(np.isfinite(translation)):
            raise ValueError(
                "Translation contains NaN or infinite values."
            )

        return translation.copy()