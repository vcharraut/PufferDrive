import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
import numpy as np


def plot_numpy_bounding_boxes(
    ax,
    bboxes: np.ndarray,
    color: np.ndarray,
    alpha=1.0,
    line_width_scale: float = 1.5,
    as_center_pts: bool = False,
    label=None,
) -> None:
    """Plots multiple bounding boxes.

    Args:
      ax: Fig handles.
      bboxes: Shape (num_bbox, 5), with last dimension as (x, y, length, width,
        yaw).
      color: Shape (3,), represents RGB color for drawing.
      alpha: Alpha value for drawing, i.e. 0 means fully transparent.
      as_center_pts: If set to True, bboxes will be drawn as center points,
        instead of full bboxes.
      label: String, represents the meaning of the color for different boxes.
    """
    if bboxes.ndim != 2 or bboxes.shape[1] != 5:
        raise ValueError(
            ("Expect bboxes rank 2, last dimension of bbox 5 got{}, {}, {} respectively").format(
                bboxes.ndim, bboxes.shape[1], color.shape
            )
        )

    if as_center_pts:
        ax.plot(
            bboxes[:, 0],
            bboxes[:, 1],
            "o",
            color=color,
            ms=2,
            alpha=alpha,
            linewidth=1.7 * line_width_scale,
            label=label,
        )
    else:
        c = np.cos(bboxes[:, 4])
        s = np.sin(bboxes[:, 4])
        pt = np.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
        length, width = bboxes[:, 2], bboxes[:, 3]
        u = np.array((c, s))
        ut = np.array((s, -c))

        # Compute box corner coordinates.
        tl = pt + length / 2 * u - width / 2 * ut
        tr = pt + length / 2 * u + width / 2 * ut
        br = pt - length / 2 * u + width / 2 * ut
        bl = pt - length / 2 * u - width / 2 * ut

        # Compute heading arrow using center left/right/front.
        cl = pt - width / 2 * ut
        cr = pt + width / 2 * ut
        cf = pt + length / 2 * u

        # Draw bboxes.
        ax.plot(
            [tl[0, :], tr[0, :], br[0, :], bl[0, :], tl[0, :]],
            [tl[1, :], tr[1, :], br[1, :], bl[1, :], tl[1, :]],
            color=color,
            zorder=4,
            linewidth=1.7 * line_width_scale,
            alpha=alpha,
            label=label,
        )

        # Draw heading arrow.
        ax.plot(
            [cl[0, :], cr[0, :], cf[0, :], cl[0, :]],
            [cl[1, :], cr[1, :], cf[1, :], cl[1, :]],
            color=color,
            zorder=6,
            alpha=alpha,
            linewidth=1.5 * line_width_scale,
            label=label,
        )


def img_from_fig(fig) -> np.ndarray:
    """Returns a [H, W, 3] uint8 np image from fig.canvas.tostring_argb()."""
    fig.subplots_adjust(left=0.08, bottom=0.08, right=0.98, top=0.98, wspace=0.0, hspace=0.0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    img = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, 1:]
    plt.close(fig)

    return img


def plot_entity(ax, entity, idx, active_agent_indices, static_car_indices):
    entity_type = entity["type"]

    # Vehicle
    if entity_type == 1:
        if entity["valid"] == 0:
            return

        x = np.array(entity["x"])
        y = np.array(entity["y"])
        length = np.array(entity["length"])
        width = np.array(entity["width"])
        heading = np.array(entity["heading"])
        goal_x = np.array(entity["goal_position_x"])
        goal_y = np.array(entity["goal_position_y"])

        if idx in active_agent_indices:
            obj_color = np.array([65, 105, 225]) / 255.0  # Royal Blue
        elif idx in static_car_indices:
            obj_color = np.array([139, 69, 19]) / 255.0  # Saddle Brown
        else:
            obj_color = np.array([0.3, 0.3, 0.3])  # Grey

        bbox = np.array((x, y, length, width, heading)).reshape(1, 5)
        plot_numpy_bounding_boxes(ax, bbox, color=obj_color, alpha=0.5)

        if idx in active_agent_indices:
            ax.scatter(goal_x, goal_y, color="red", marker="*", s=20)

    # Pedestrian
    if entity_type == 2:
        if entity["valid"] == 0:
            return

        x = np.array(entity["x"])
        y = np.array(entity["y"])
        length = np.array(entity["length"])
        width = np.array(entity["width"])
        heading = np.array(entity["heading"])
        goal_x = np.array(entity["goal_position_x"])
        goal_y = np.array(entity["goal_position_y"])

        obj_color = np.array([0.0, 1.0, 0.0])

        bbox = np.array((x, y, length, width, heading)).reshape(1, 5)
        plot_numpy_bounding_boxes(ax, bbox, color=obj_color, alpha=0.5)

    # Cyclist
    if entity_type == 3:
        if entity["valid"] == 0:
            return

        x = np.array(entity["x"])
        y = np.array(entity["y"])
        length = np.array(entity["length"])
        width = np.array(entity["width"])
        heading = np.array(entity["heading"])
        goal_x = np.array(entity["goal_position_x"])
        goal_y = np.array(entity["goal_position_y"])

        obj_color = np.array([1.0, 0.0, 1.0])

        bbox = np.array((x, y, length, width, heading)).reshape(1, 5)
        plot_numpy_bounding_boxes(ax, bbox, color=obj_color, alpha=0.5)

    # Road lane
    if entity_type == 4:
        ax.plot(entity["traj_x"], entity["traj_y"], color="lightgrey", linewidth=1)

    # Road line
    if entity_type == 5:
        ax.plot(entity["traj_x"], entity["traj_y"], color="grey", linewidth=1, linestyle="--")

    # Road edge
    if entity_type == 6:
        ax.plot(entity["traj_x"], entity["traj_y"], color="black", linewidth=2)

    # Stop sign
    if entity_type == 7:
        ax.plot(entity["traj_x"], entity["traj_y"], color="red", linewidth=3, linestyle="--")

    # Crosswalk
    if entity_type == 8:
        points = np.vstack((entity["traj_x"], entity["traj_y"])).T
        ax.add_patch(
            Polygon(
                xy=points,
                facecolor="none",
                edgecolor="xkcd:bluish grey",
                linewidth=2,
                alpha=0.4,
                hatch=r"//",
                zorder=1,
            )
        )

    # Speed bump
    if entity_type == 9:
        points = np.vstack((entity["traj_x"], entity["traj_y"])).T
        ax.add_patch(
            Polygon(
                xy=points,
                facecolor="xkcd:goldenrod",
                edgecolor="xkcd:black",
                linewidth=0,
                alpha=0.5,
                hatch=r"//",
                zorder=2,
            )
        )


def plot_simulator_state(scenario):
    fig, axs = plt.subplots(figsize=(20, 20))

    for idx, entity in enumerate(scenario["entities"]):
        plot_entity(axs, entity, idx, scenario["active_agent_indices"], scenario["static_car_indices"])

    axs.set_xlim(-75, 75)
    axs.set_ylim(-75, 75)
    axs.set_aspect("equal", adjustable="box")

    return img_from_fig(fig)
