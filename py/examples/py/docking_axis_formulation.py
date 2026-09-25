import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from py.modules.math import quaternion_from_euler, quaternion_to_DCM

# Información definida por el problema
target_docking_axis_wrt_target_frame = np.array([0.0, 0.0, 1.0])  # Docking axis of the target in the target frame
target_complementary_axis_wrt_target_frame = np.array([1.0, 0.0, 0.0])  # Complementary axis of the target in the target frame
chaser_docking_axis_wrt_chaser_frame = np.array([1.0, 0.0, 0.0])  # Docking axis of the chaser in the chaser frame
chaser_complementary_axis_wrt_chaser_frame = np.array([0.0, 0.0, 1.0])  # Complementary axis of the chaser in the chaser frame
distance_along_docking_axis = 10.0  # Distance along the docking axis in meters from the target's docking port
interception_point_wrt_target_frame = distance_along_docking_axis * target_docking_axis_wrt_target_frame  # Interception point in the target frame

# Información actual del objetivo y del perseguidor
target_current_position_wrt_inertal_frame = np.array([0.0, 0.0, 0.0])  # Current target position in meters wrt the inertial frame
target_current_attitude_wrt_inertal_frame = np.array([0.0, 0.0, 0.0])  # Target attitude in degrees wrt the inertial frame
target_current_attitude_wrt_inertal_frame = np.radians(target_current_attitude_wrt_inertal_frame)  # Convert target attitude to radians
target_current_quaternion_wrt_inertal_frame = quaternion_from_euler(*target_current_attitude_wrt_inertal_frame)  # Convert target attitude to quaternion

rotation_matrix_target_to_inertial = quaternion_to_DCM(target_current_quaternion_wrt_inertal_frame)  # Rotation matrix from target frame to inertial frame

chaser_current_position_wrt_inertial_frame = np.array([50.0, 50.0, 50.0])  # Chaser position in meters wrt the inertial frame
chaser_current_attitude_wrt_inertial_frame = np.array([45.0, 45.0, 0.0])  # Chaser attitude in degrees wrt the inertial frame
chaser_current_attitude_wrt_inertial_frame = np.radians(chaser_current_attitude_wrt_inertial_frame)  # Convert chaser attitude to radians
chaser_current_quaternion_wrt_inertial_frame = quaternion_from_euler(*chaser_current_attitude_wrt_inertial_frame)  # Convert chaser attitude to quaternion

rotation_matrix_chaser_to_inertial = quaternion_to_DCM(chaser_current_quaternion_wrt_inertial_frame)  # Rotation matrix from chaser frame to inertial frame

# Calcular la posición del punto de intercepción en el marco inercial
interception_point_wrt_inertial_frame = target_current_position_wrt_inertal_frame + rotation_matrix_target_to_inertial @ interception_point_wrt_target_frame  # Position of the docking distance in the inertial frame
chaser_relative_distance_to_interception_point = interception_point_wrt_inertial_frame - chaser_current_position_wrt_inertial_frame  # Relative distance between the chaser and the interception point

# Calcular la orientación deseada del perseguidor para alinear su eje de acoplamiento con el eje de acoplamiento del objetivo
target_docking_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ target_docking_axis_wrt_target_frame  # Target docking axis in the inertial frame
target_complementary_axis_wrt_inertial_frame = rotation_matrix_target_to_inertial @ target_complementary_axis_wrt_target_frame  # Target complementary axis in the inertial frame
chaser_docking_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ chaser_docking_axis_wrt_chaser_frame  # Chaser docking axis in the inertial frame
chaser_complementary_axis_wrt_inertial_frame = rotation_matrix_chaser_to_inertial @ chaser_complementary_axis_wrt_chaser_frame  # Chaser complementary axis in the inertial frame

# Se necesita que los dos ejes del docking esten enfrentados, es decir dC = -dT
desired_chaser_docking_axis_wrt_inertial_frame = -target_docking_axis_wrt_inertial_frame
docking_axis_alignment_error = desired_chaser_docking_axis_wrt_inertial_frame - chaser_docking_axis_wrt_inertial_frame

# Pero adicional, se necesita una orientación especifica, basicamente cerrar el roll sobre ese del docking una vez alineados.

desired_chaser_complementary_axis_wrt_inertial_frame = target_complementary_axis_wrt_inertial_frame
complementary_axis_alignment_error = desired_chaser_complementary_axis_wrt_inertial_frame - chaser_complementary_axis_wrt_inertial_frame



print(complementary_axis_alignment_error)
print(docking_axis_alignment_error)

def plot_rpod_geometry_3d(
    target_position_wrt_inertial,
    target_rotation_matrix_to_inertial,
    target_docking_axis_wrt_target,
    target_complementary_axis_wrt_target,
    chaser_position_wrt_inertial,
    chaser_rotation_matrix_to_inertial,
    chaser_docking_axis_wrt_chaser,
    chaser_complementary_axis_wrt_chaser,
    interception_point_wrt_inertial,
    docking_axis_extension=25.0,
    frame_scale=8.0,
    vehicle_scale=4.0,
):
    """
    Visualiza la geometría de un escenario RPOD en 3D.

    Convención:
      - Azul: Target
      - Rojo: Chaser
      - Verde: punto deseado de intercepción
      - Negro: eje de docking de cada vehículo
      - Morado: eje complementario de cada vehículo
      - Azul punteado: docking axis del target extendido más allá del
        punto de intercepción.

    Las matrices de rotación se interpretan como:
        vector_inertial = R_frame_to_inertial @ vector_frame
    """

    fig = plt.figure(figsize=(11, 9))
    ax = fig.add_subplot(111, projection="3d")

    # ------------------------------------------------------------------
    # Transformación de los ejes de cada vehículo al marco inercial
    # ------------------------------------------------------------------
    target_docking_axis = (
        target_rotation_matrix_to_inertial @ target_docking_axis_wrt_target
    )
    target_complementary_axis = (
        target_rotation_matrix_to_inertial
        @ target_complementary_axis_wrt_target
    )

    chaser_docking_axis = (
        chaser_rotation_matrix_to_inertial @ chaser_docking_axis_wrt_chaser
    )
    chaser_complementary_axis = (
        chaser_rotation_matrix_to_inertial
        @ chaser_complementary_axis_wrt_chaser
    )

    # Normalizar por seguridad
    target_docking_axis /= np.linalg.norm(target_docking_axis)
    target_complementary_axis /= np.linalg.norm(target_complementary_axis)
    chaser_docking_axis /= np.linalg.norm(chaser_docking_axis)
    chaser_complementary_axis /= np.linalg.norm(chaser_complementary_axis)

    # ------------------------------------------------------------------
    # Dibujar un vehículo como un punto grande
    # ------------------------------------------------------------------
    ax.scatter(
        *target_position_wrt_inertial,
        color="blue",
        s=250,
        marker="o",
        label="Target",
        depthshade=True,
    )

    ax.scatter(
        *chaser_position_wrt_inertial,
        color="red",
        s=250,
        marker="o",
        label="Chaser",
        depthshade=True,
    )

    # ------------------------------------------------------------------
    # Punto deseado de intercepción
    # ------------------------------------------------------------------
    ax.scatter(
        *interception_point_wrt_inertial,
        color="green",
        s=180,
        marker="X",
        label="Desired interception point",
        depthshade=True,
    )

    # ------------------------------------------------------------------
    # Ejes de referencia de cada vehículo
    #
    # X = rojo
    # Y = verde
    # Z = azul
    #
    # Se dibujan más pequeños que los ejes de docking para diferenciarlos.
    # ------------------------------------------------------------------
    inertial_axis_colors = ["red", "green", "blue"]

    for i, color in enumerate(inertial_axis_colors):
        axis = target_rotation_matrix_to_inertial[:, i]
        ax.quiver(
            *target_position_wrt_inertial,
            *axis,
            length=frame_scale,
            color=color,
            linewidth=1.5,
            alpha=0.65,
            arrow_length_ratio=0.15,
        )

        axis = chaser_rotation_matrix_to_inertial[:, i]
        ax.quiver(
            *chaser_position_wrt_inertial,
            *axis,
            length=frame_scale,
            color=color,
            linewidth=1.5,
            alpha=0.65,
            arrow_length_ratio=0.15,
        )

    # ------------------------------------------------------------------
    # Ejes de docking
    #
    # Target:
    #   negro sólido = eje principal
    #   azul punteado = extensión física/geométrica del docking axis
    #
    # Chaser:
    #   negro sólido = eje principal
    # ------------------------------------------------------------------
    ax.quiver(
        *target_position_wrt_inertial,
        *target_docking_axis,
        length=vehicle_scale,
        color="black",
        linewidth=3,
        arrow_length_ratio=0.18,
    )

    ax.quiver(
        *chaser_position_wrt_inertial,
        *chaser_docking_axis,
        length=vehicle_scale,
        color="black",
        linewidth=3,
        arrow_length_ratio=0.18,
    )

    # ------------------------------------------------------------------
    # Docking axis del target extendido hasta y más allá del punto deseado
    #
    # Se dibuja desde el target hasta:
    #     interception_point + extension * target_docking_axis
    # ------------------------------------------------------------------
    docking_axis_end = (
        interception_point_wrt_inertial
        + docking_axis_extension * target_docking_axis
    )

    ax.plot(
        [
            target_position_wrt_inertial[0],
            docking_axis_end[0],
        ],
        [
            target_position_wrt_inertial[1],
            docking_axis_end[1],
        ],
        [
            target_position_wrt_inertial[2],
            docking_axis_end[2],
        ],
        color="blue",
        linestyle="--",
        linewidth=2.0,
        label="Target docking axis",
    )

    # ------------------------------------------------------------------
    # Ejes complementarios
    # ------------------------------------------------------------------
    ax.quiver(
        *target_position_wrt_inertial,
        *target_complementary_axis,
        length=vehicle_scale,
        color="purple",
        linewidth=3,
        arrow_length_ratio=0.18,
    )

    ax.quiver(
        *chaser_position_wrt_inertial,
        *chaser_complementary_axis,
        length=vehicle_scale,
        color="purple",
        linewidth=3,
        arrow_length_ratio=0.18,
    )

    # ------------------------------------------------------------------
    # Línea entre el chaser y el punto de intercepción
    # ------------------------------------------------------------------
    ax.plot(
        [
            chaser_position_wrt_inertial[0],
            interception_point_wrt_inertial[0],
        ],
        [
            chaser_position_wrt_inertial[1],
            interception_point_wrt_inertial[1],
        ],
        [
            chaser_position_wrt_inertial[2],
            interception_point_wrt_inertial[2],
        ],
        color="gray",
        linestyle=":",
        linewidth=1.5,
        alpha=0.8,
    )

    # ------------------------------------------------------------------
    # Etiquetas
    # ------------------------------------------------------------------
    ax.text(
        *target_position_wrt_inertial,
        "  Target",
        color="blue",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        *chaser_position_wrt_inertial,
        "  Chaser",
        color="red",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        *interception_point_wrt_inertial,
        "  Desired point",
        color="green",
        fontsize=10,
        fontweight="bold",
    )

    # ------------------------------------------------------------------
    # Sistema de referencia inercial
    # ------------------------------------------------------------------
    origin = np.zeros(3)

    for i, color in enumerate(inertial_axis_colors):
        axis = np.zeros(3)
        axis[i] = 1.0

        ax.quiver(
            *origin,
            *axis,
            length=frame_scale * 1.2,
            color=color,
            linewidth=1.0,
            alpha=0.25,
            arrow_length_ratio=0.12,
        )

    ax.text(frame_scale * 1.2, 0, 0, "I-X", alpha=0.5)
    ax.text(0, frame_scale * 1.2, 0, "I-Y", alpha=0.5)
    ax.text(0, 0, frame_scale * 1.2, "I-Z", alpha=0.5)

    # ------------------------------------------------------------------
    # Aspecto y límites
    # ------------------------------------------------------------------
    points = np.vstack([
        target_position_wrt_inertial,
        chaser_position_wrt_inertial,
        interception_point_wrt_inertial,
        docking_axis_end,
    ])

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2
    span = np.max(maxs - mins)

    # Evitar que la gráfica quede demasiado pequeña si la geometría
    # tiene poca extensión.
    span = max(span, 2 * frame_scale)

    margin = 0.25 * span
    half_range = span / 2 + margin

    ax.set_xlim(center[0] - half_range, center[0] + half_range)
    ax.set_ylim(center[1] - half_range, center[1] + half_range)
    ax.set_zlim(center[2] - half_range, center[2] + half_range)

    ax.set_box_aspect([1, 1, 1])

    ax.set_xlabel("Inertial X [m]")
    ax.set_ylabel("Inertial Y [m]")
    ax.set_zlabel("Inertial Z [m]")

    ax.set_title("RPOD Geometry — Target / Chaser Docking Alignment")

    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    return fig, ax

fig, ax = plot_rpod_geometry_3d(
    target_position_wrt_inertial=target_current_position_wrt_inertal_frame,
    target_rotation_matrix_to_inertial=rotation_matrix_target_to_inertial,
    target_docking_axis_wrt_target=target_docking_axis_wrt_target_frame,
    target_complementary_axis_wrt_target=target_complementary_axis_wrt_target_frame,

    chaser_position_wrt_inertial=chaser_current_position_wrt_inertial_frame,
    chaser_rotation_matrix_to_inertial=rotation_matrix_chaser_to_inertial,
    chaser_docking_axis_wrt_chaser=chaser_docking_axis_wrt_chaser_frame,
    chaser_complementary_axis_wrt_chaser=chaser_complementary_axis_wrt_chaser_frame,

    interception_point_wrt_inertial=interception_point_wrt_inertial_frame,

    docking_axis_extension=25.0,        
)

plt.show()

