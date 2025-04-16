from src.PARAMOUNT import DMD


def main():
    """
    Perform multi-resolution DMD (mrDMD) analysis and visualize results.

    Parameters Used
    ----------
    path_parquet: path to folder with parquet subfolders
    path_mrdmd: path to save mrDMD results
    variables: list of variables to consider
    bounds: domain bounds for visualization [xmin, xmax, ymin, ymax, resolution]
    cutoff: percentage of data to keep for mrDMD analysis (from begining up to cutoff%)
    """
    dmd = DMD(show_dashboard=False)
    cutoff = 80
    dmd.set_data_cutoff(cutoff)

    path_parquet = ".raw_pq"
    path_mrdmd = ".mrDMD"

    path_viz_multires = f"{path_mrdmd}/viz_multires"
    path_viz_error = f"{path_mrdmd}/viz_error"
    path_viz_eig = f"{path_mrdmd}/viz_eigs"

    variables = DMD.get_folderlist(path_parquet, boolPrint=True, ignoreStack=True)
    variables = variables[:5]

    dmd.set_time(4e-5)
    bounds = [-0.075, 0.075, -0.005, 0.25, 2.5e-4]
    dmd.set_viz_params(
        dpi=300, linewidth=0.85, contour_levels=120, color="k", cmap="viridis"
    )

    dmd.multires(
        variables,
        path_parquet=path_parquet,
        path_mrdmd=path_mrdmd,
        levels=4,
        end=2980,
    )

    dmd.multires_predict(variables, path_mrdmd=path_mrdmd, end=2980)

    dmd.viz_multires(
        variables,
        num_frames=10,
        path_mrdmd=path_mrdmd,
        path_viz=path_viz_multires,
        bounds=bounds,
        dist=0.0026,
        cbar=False,
        vmax=None,
        vmin=None,
    )

    dmd.viz_error_combined(
        variables,
        path_dmd=f"{path_mrdmd}/level_0/level_prediction",
        path_viz=path_viz_error,
        path_mrdmd=path_mrdmd,
    )

    dmd.viz_eigs_circle(
        variables,
        path_dmd=f"{path_mrdmd}/level_0/0",
        path_viz=path_viz_eig,
        maxmode=None,
    )

    dmd.viz_eigs_spectrum(
        variables,
        path_dmd=f"{path_mrdmd}/level_0/0",
        path_viz=path_viz_eig,
        maxmode=None,
        freq_max=3000,
    )


if __name__ == "__main__":
    main()
