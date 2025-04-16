from src.PARAMOUNT_DMD import DMD


def main():
    """
    Process parquet datasets and perform Dynamic Mode Decomposition (DMD) and visualize the results.

    Parameters Used
    ----------
    path_parquet: path to folder containing the parquet datasets
    path_pod: path to folder to store Singular Value Decomposition (SVD) results
    path_dmd: path to folder to store DMD results
    path_dmd2: path to folder to store secondary DMD results for comparison
    path_viz: path to folder to store generated visualizations
    variables: list of variables to process

    DMD(): initialize a dask cluster on local system
    """
    dmd = DMD(show_dashboard=False)

    dmd.set_data_cutoff(80)

    path_parquet = ".data"
    path_dmd = ".usv"
    path_dmd2 = ".usv_projected"
    path_pod = ".usv_pod"
    path_viz = f"{path_dmd}/viz"
    variables = DMD.get_folderlist(path_parquet, boolPrint=True, ignoreStack=True)
    variables = variables[:3]

    dmd.set_time(4e-5)
    bounds = "auto"
    dmd.set_viz_params(
        dpi=300, linewidth=0.85, contour_levels=120, color="k", cmap="viridis"
    )

    dmd.svd_save_usv(
        variables,
        path_parquet=path_parquet,
        path_pod=path_pod,
        dmd_X1=True,
    )

    dmd.save_Atilde(
        variables,
        path_parquet=path_parquet,
        path_pod=path_pod,
        path_dmd=path_dmd,
    )
    dmd.save_modes(
        variables,
        path_parquet=path_parquet,
        path_dmd=path_dmd,
        path_pod=path_pod,
        projection_method=True,
    )
    dmd.save_prediction(variables, path_dmd=path_dmd, end=2980)

    dmd.viz_eigs_circle(
        variables, path_dmd=path_dmd, path_viz=f"{path_viz}/eigs", maxmode=None
    )

    dmd.viz_eigs_spectrum(
        variables,
        path_dmd=path_dmd,
        path_pod=path_pod,
        path_viz=f"{path_viz}/eigs",
        maxmode=None,
        freq_max=3000,
    )

    dmd.viz_eigs_spectrum_combined(
        variables,
        path_dmd=path_dmd,
        path_pod=path_pod,
        path_viz=f"{path_viz}/eigs",
        maxmode=None,
        freq_max=3000,
    )

    dmd.viz_modes(
        variables,
        modelist=None,
        coordinates="2D",
        path_dmd=path_dmd,
        path_viz=f"{path_viz}/modes",
        bounds=bounds,
        freqlist=[200, 400, 600],
        dist=2.5e-4 * 13,
    )

    dmd.viz_parquet(
        variables,
        num_frames=10,
        coordinates="2D",
        path_data=path_dmd,
        folder_name="prediction",
        path_viz=f"{path_viz}/prediction",
        bounds=bounds,
        dist=2.5e-4 * 13,
        vmax="auto",
        vmin="auto",
    )

    dmd.describe_parquet(
        variables,
        path_data=path_dmd,
        folder_name="prediction",
    )

    dmd.viz_error(
        variables,
        path_data=path_parquet,
        path_dmd=path_dmd,
        path_viz=f"{path_viz}/error",
    )


if __name__ == "__main__":
    main()
