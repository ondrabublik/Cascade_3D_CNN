import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv


def sorted_vtk_files(path, prefix):
    files = list(path.glob(f'{prefix}_*.vtu'))
    return sorted(files, key=lambda f: int(re.search(r'(\d+)\.vtu$', f.name).group(1)))


def load_dt(data_dir):
    params_file = Path(data_dir).parents[0] / 'data_3D' / 'parameters.json'
    if params_file.exists():
        with open(params_file) as f:
            return float(json.load(f)['dt'])
    return 1.0


def prepare_surface_geometry(surface):
    points = surface.points
    blade_idx_pts = surface.point_data['blade_index']
    n_cells = surface.n_cells

    cell_point_ids = np.zeros((n_cells, 4), dtype=int)
    cell_blade = np.full(n_cells, -1, dtype=int)
    normals = np.zeros((n_cells, 3))
    areas = np.zeros(n_cells)

    for i in range(n_cells):
        ids = surface.get_cell(i).point_ids
        cell_point_ids[i] = ids

        blade_vals = blade_idx_pts[ids]
        if np.all(blade_vals == blade_vals[0]):
            cell_blade[i] = int(blade_vals[0])

        pts = points[ids]
        area_vec = 0.5 * (
            np.cross(pts[1] - pts[0], pts[2] - pts[0])
            + np.cross(pts[2] - pts[0], pts[3] - pts[0])
        )
        area = np.linalg.norm(area_vec)
        areas[i] = area
        if area > 1e-14:
            normals[i] = area_vec / area

    return cell_point_ids, cell_blade, normals, areas


def compute_lift(surface, geometry, blade_index, lift_axis=1):
    """Integrate pressure over blade surface: L = -∫ p n_y dA."""
    cell_point_ids, cell_blade, normals, areas = geometry
    pressure = surface.point_data['pressure']

    mask = cell_blade == blade_index
    if not np.any(mask):
        return 0.0

    ids = cell_point_ids[mask]
    p_avg = np.mean(pressure[ids], axis=1)
    return np.sum(-p_avg * normals[mask, lift_axis] * areas[mask])


def collect_lift_history(path, blade_indices, dt):
    unet_files = sorted_vtk_files(path, 'pressure_UNet')
    cfd_files = sorted_vtk_files(path, 'pressure_CFD')

    if not unet_files or not cfd_files:
        raise FileNotFoundError(f"No pressure VTK files found in {path}")
    if len(unet_files) != len(cfd_files):
        raise ValueError("UNet and CFD VTK file counts do not match.")

    geometry = prepare_surface_geometry(pv.read(unet_files[0]))

    n_steps = len(unet_files)
    time = np.arange(n_steps, dtype=float) * dt

    lift_unet = {blade: np.zeros(n_steps) for blade in blade_indices}
    lift_cfd = {blade: np.zeros(n_steps) for blade in blade_indices}

    for step, (unet_file, cfd_file) in enumerate(zip(unet_files, cfd_files)):
        mesh_unet = pv.read(unet_file)
        mesh_cfd = pv.read(cfd_file)

        for blade in blade_indices:
            lift_unet[blade][step] = compute_lift(mesh_unet, geometry, blade)
            lift_cfd[blade][step] = compute_lift(mesh_cfd, geometry, blade)

        if step % 10 == 0:
            print(f'{step} / {n_steps - 1}')

    return time, lift_unet, lift_cfd


def get_blade_indices(path):
    unet_files = sorted_vtk_files(path, 'pressure_UNet')
    mesh = pv.read(unet_files[0])
    blade_idx = np.asarray(mesh.point_data['blade_index'], dtype=int)
    return sorted(np.unique(blade_idx[blade_idx >= 0]).tolist())


def save_lift_csv(path, time, lift_unet, lift_cfd):
    rows = []
    for blade in sorted(lift_unet.keys()):
        for step, t in enumerate(time):
            rows.append([
                step,
                t,
                blade,
                lift_unet[blade][step],
                lift_cfd[blade][step],
                lift_unet[blade][step] - lift_cfd[blade][step],
            ])

    header = 'step,time,blade_index,lift_UNet,lift_CFD,lift_diff'
    np.savetxt(path / 'lift_history.csv', rows, delimiter=',', header=header, comments='')


def plot_lift_per_blade(path, time, lift_unet, lift_cfd):
    plt.rcParams.update({'font.size': 14})

    for blade in sorted(lift_unet.keys()):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(time, lift_cfd[blade], 'b-', linewidth=2, label='CFD')
        ax.plot(time, lift_unet[blade], 'r--', linewidth=2, label='UNet')
        ax.set_xlabel('Time')
        ax.set_ylabel('Lift')
        ax.set_title(f'Lift vs time - blade {blade}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path / f'lift_blade_{blade}.png', dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    projectDir = Path(__file__).resolve().parents[1]

    dataDir = projectDir.parent / 'reader3D' / 'FinalBladeCascade' / 'data' / 'transformed_10o'
    pathResults = projectDir / 'data' / 'net7_3D_multistep_lowo' / 'results_blade_pressure_vent10'

    dt = load_dt(dataDir)
    blade_indices = get_blade_indices(pathResults)

    print(f'Processing {pathResults}')
    print(f'Blade indices: {blade_indices}, dt = {dt}')

    time, lift_unet, lift_cfd = collect_lift_history(pathResults, blade_indices, dt)
    save_lift_csv(pathResults, time, lift_unet, lift_cfd)
    plot_lift_per_blade(pathResults, time, lift_unet, lift_cfd)

    print(f'Saved lift_history.csv and lift_blade_*.png to {pathResults}')
