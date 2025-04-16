from src.PARAMOUNT_POD import POD


def main():
    """
    Perform Proper Orthogonal Decomposition (POD) and store results and visualize the results.

    Parameters Used
    ----------
    path_parquet: path to folder with parquet subfolders
    path_results_pod: path to save SVD results
    path_viz: path to save visualization results
    variables: list of variables to consider
    modelist: list of mode numbers to process
    dt: time step of data acquisition in seconds
    bounds: domain bounds for visualization [xmin, xmax, ymin, ymax, resolution] or "auto"
    freq_max: maximum frequency of interest in PSD plots

    POD(): initialize a dask cluster on local system
    """
    # Paths and parameters
    path_parquet = ".data"
    path_results_pod = ".usv"
    path_viz = ".viz"
    path_signals = ".signals"
    path_signals2 = ".signals2"
    path_results_pod2 = ".usv2"
    modelist = range(5)
    dt = 4e-5
    bounds = "auto"
    coordinates = "2D"
    freq_max = 3000

    # Initialize POD
    pod = POD(show_dashboard=True)

    # Step 1: Perform SVD and save results
    variables = POD.get_folderlist(path_parquet, boolPrint=True)
    pod.svd_save_usv(
        variables,
        path_parquet=path_parquet,
        path_results_pod=path_results_pod,
    )

    # Step 2: Visualize SVD results
    pod.set_time(dt)
    pod.set_viz_params(dpi=600, linewidth=0.85, color="k", cmap="seismic")

    pod.svd_viz(
        variables,
        modelist,
        bounds=bounds,
        coordinates=coordinates,
        path_results_pod=path_results_pod,
        path_viz=path_viz,
        freq_max=freq_max,
        dist=2.5e-4 * 13,
    )

    pod.s_viz_combined(
        variables,
        maxmode=100,
        path_results_pod=path_results_pod,
        path_viz=path_viz,
    )

    pod.average_viz(
        variables,
        path_data=path_parquet,
        path_viz=f"{path_viz}/average",
        dist=2.5e-4 * 13,
        bounds=bounds,
        cbar=True,
    )

    pod.svd_correlation(
        variables,
        maxmode=modelist,
        path_results_pod=path_results_pod,
        path_viz=path_viz,
    )

    pod.svd_correlation_2X(
        variables,
        maxmode=modelist,
        path_results_pod=path_results_pod,
        path_results_pod2=path_results_pod2,
        path_viz=path_viz,
    )

    pod.svd_correlation_signals(
        variables,
        maxmode=modelist,
        path_results_pod=path_results_pod,
        path_signals=path_signals,
        path_viz=path_viz,
    )

    pod.correlation_signals(
        path_signals=path_signals, path_signals2=path_signals2, path_viz=path_viz
    )


if __name__ == "__main__":
    main()
