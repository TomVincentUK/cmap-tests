"""largest_uniform_disc

This script attempts to find the biggest 2D circle that fits in the CAM-02
uniform colour space and also the rgb gamut.
I use this to generate a bunch of perceptually uniform colour maps that
will look good together.

It's hideously inefficient. I just generate random circles and reject them
if they're out of gamut, keeping the best ones.

With cache = True, you can run it a bunch of times to see the solution
getting better.
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy.constants import golden
from colorspacious import cspace_convert


def rgb_to_Jab(rgb):
    return cspace_convert(rgb, "sRGB1", "CAM02-UCS")


def Jab_to_rgb(rgb):
    return cspace_convert(rgb, "CAM02-UCS", "sRGB1")


def valid_rgb(rgb):
    out_of_range = (np.abs(np.asarray(rgb) - 0.5) > 0.5).any()
    all_finite = np.isfinite(rgb).all()
    return all_finite and not out_of_range


def normal_to_rotation(normal):
    normal = np.asarray(normal, dtype=float)

    normal /= np.linalg.norm(normal)
    v2 = np.array([1, 0, 0], dtype=float)
    v2 = np.array([1, 1, 0], dtype=float) if all(v2 == normal) else v2
    v2 -= normal * (v2 @ normal)
    v2 /= np.linalg.norm(v2)
    v3 = np.cross(normal, v2)
    return np.stack([normal, v2, v3]).T


def generate_circle(r, centre, normal, res=4096):
    centre = np.asarray(centre, dtype=float)
    normal = np.asarray(normal, dtype=float)

    # Unit circle
    theta = np.linspace(0, 2 * np.pi, res, endpoint=False)
    unit_circle = np.stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)])

    # Rotation matrix
    M = normal_to_rotation(normal)

    return (centre[..., np.newaxis] + r * (M @ unit_circle)).T


def generate_sunflower_disc(r, centre, normal, res=4096):
    centre = np.asarray(centre, dtype=float)
    normal = np.asarray(normal, dtype=float)

    # Generate sunflower seed pattern for unit disc
    n = np.arange(res)
    theta = n * 2 * np.pi / golden**2
    r_unit = np.sqrt(n) / np.sqrt(n).max()
    unit_disc = r_unit * np.stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)])

    # Rotation matrix
    M = normal_to_rotation(normal)

    return (centre[..., np.newaxis] + r * (M @ unit_disc)).T


def f_theta(theta, r, centre, normal, offset=0):
    theta = np.asarray(theta, dtype=float) - offset
    centre = np.asarray(centre, dtype=float)
    normal = np.asarray(normal, dtype=float)

    # Generate points on a unit circle from theta values
    points = np.stack([np.zeros_like(theta), np.cos(theta), np.sin(theta)])
    points = points[:, np.newaxis] if points.ndim == 1 else points

    # Rotation matrix
    M = normal_to_rotation(normal)

    return (centre[..., np.newaxis] + r * (M @ points)).T


cache = True
cache_path = pathlib.Path("best_circle.npz")
if cache and cache_path.exists():
    best_params = dict(np.load(cache_path))
else:
    best_params = dict(r=1, centre=[50, 0, 0], normal=[1, 0, 0])

n_iterations = 100
r_min = best_params["r"]
r_max = 50
rng = np.random.default_rng()
for i in tqdm(range(n_iterations)):
    params = dict(
        r=rng.triangular(r_min, r_min, r_max),
        centre=rng.triangular([0, -50, -50], [50, 0, 0], [100, 50, 50]),
        normal=rng.normal(size=3),
    )

    Jab_circle = generate_circle(**params)
    rgb_circle = Jab_to_rgb(Jab_circle)
    if valid_rgb(rgb_circle):
        best_params = params
        r_min = params["r"]

if cache:
    np.savez(cache_path, **best_params)

# Result as circle
best_Jab_circle = generate_circle(**best_params)
best_rgb_circle = Jab_to_rgb(best_Jab_circle)

fig, axes = plt.subplots(nrows=7, sharex=True)
axes[0].plot(best_Jab_circle[:, 0])
axes[0].set(ylabel="Brightness", ylim=[0, 100])

axes[1].plot(np.sqrt(best_Jab_circle[:, 1] ** 2 + best_Jab_circle[:, 2] ** 2))
axes[1].set(ylabel="Saturation", ylim=[0, 50])

axes[2].plot(np.arctan2(best_Jab_circle[:, 2], best_Jab_circle[:, 1]))
axes[2].set(ylabel="Hue", ylim=[-np.pi, np.pi])

axes[3].plot(best_rgb_circle[:, 0], c="r")
axes[3].set(ylabel="R", ylim=[0, 1])

axes[4].plot(best_rgb_circle[:, 1], c="g")
axes[4].set(ylabel="G", ylim=[0, 1])

axes[5].plot(best_rgb_circle[:, 2], c="b")
axes[5].set(ylabel="B", ylim=[0, 1])

axes[6].imshow(best_rgb_circle[np.newaxis], aspect="auto")
axes[6].set(ylabel="Colour", yticks=[])

fig.tight_layout()
plt.show(block=False)

# Result as disc
best_Jab_disc = generate_sunflower_disc(**best_params)
best_rgb_disc = Jab_to_rgb(best_Jab_disc)

scatter_points = 1024
if scatter_points > 0:
    wall_scatter = rng.normal(size=(scatter_points, 3))
    wall_scatter *= np.sqrt(3) / np.linalg.norm(wall_scatter, axis=-1)[..., np.newaxis]
    wall_scatter = np.clip(wall_scatter + 0.5, 0, 1)
    best_rgb_disc = np.vstack([best_rgb_disc, wall_scatter])
    best_Jab_disc = rgb_to_Jab(best_rgb_disc)

fig, (Jab_ax, rgb_ax) = plt.subplots(ncols=2, subplot_kw=dict(projection="3d"))

Jab_ax.scatter(*best_Jab_disc.T, c=best_rgb_disc, alpha=1)
Jab_ax.set(
    xlabel="J", ylabel="a", zlabel="b", xlim=[0, 100], ylim=[-50, 50], zlim=[-50, 50]
)

rgb_ax.scatter(*best_rgb_disc.T, c=best_rgb_disc, alpha=1)
rgb_ax.set(xlabel="R", ylabel="G", zlabel="B", xlim=[0, 1], ylim=[0, 1], zlim=[0, 1])

fig.tight_layout()
plt.show(block=False)

# Generated colormaps
cmap_res = 256
linear_A = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(np.pi, 0, cmap_res), **best_params)), "linear_A"
)
linear_B = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(-np.pi, 0, cmap_res), **best_params)), "linear_B"
)
linear_C = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(np.pi, np.pi / 2, cmap_res), **best_params)),
    "linear_C",
)
linear_D = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(np.pi, 3 * np.pi / 2, cmap_res), **best_params)),
    "linear_D",
)
linear_E = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(0, np.pi / 2, cmap_res), **best_params)), "linear_E"
)
linear_F = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(0, -np.pi / 2, cmap_res), **best_params)), "linear_F"
)
divergent_A = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(np.pi / 2, -np.pi / 2, cmap_res), **best_params)),
    "divergent_A",
)
divergent_B = ListedColormap(
    Jab_to_rgb(
        f_theta(np.pi + np.linspace(-np.pi / 2, np.pi / 2, cmap_res), **best_params)
    ),
    "divergent_B",
)
cyclic_A = ListedColormap(
    Jab_to_rgb(f_theta(np.linspace(0, 2 * np.pi, cmap_res), **best_params)), "cyclic_A"
)
cyclic_B = ListedColormap(
    Jab_to_rgb(f_theta(np.pi + np.linspace(0, -2 * np.pi, cmap_res), **best_params)),
    "cyclic_B",
)

cmap_list = (
    linear_A,
    linear_B,
    linear_C,
    linear_D,
    linear_E,
    linear_F,
    divergent_A,
    divergent_B,
    cyclic_A,
    cyclic_B,
)


gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

fig, axes = plt.subplots(nrows=len(cmap_list), sharex=True)

for ax, cmap in zip(axes, cmap_list):
    ax.imshow(gradient, aspect="auto", cmap=cmap)
    ax.text(
        -0.01,
        0.5,
        cmap.name,
        va="center",
        ha="right",
        fontsize=10,
        transform=ax.transAxes,
    )
    ax.set_axis_off()

fig.tight_layout()
plt.show(block=False)
