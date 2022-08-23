from src.PARAMOUNT import POD


def main():
    """

    Inspect SVD results. Visualize modes, mode energy content and mode cross correlation map.
    correlation map betwen two seperate POD results and correlation betwen modes and other signals

    Parameters Used
    ----------
    path_results_pod: path of folder containing SVD results
    path_viz: path to save visualization results
    modelist: list of mode numbers to process
    varlist: list of variables to consider
    dt: time step of data acquisition in seconds
    bounds: domain bounds for visualization [xmin, xmax, ymin, ymax, resolution] or "auto"
    freq_max: maximum frequency of interest in PSD plots
    POD(): initialize a dask cluster on local system

    """
    path_results_pod = ".usv"
    path_viz = ".viz"
    modelist = range(5)
    dt = 4e-5
    bounds = "auto"
    coordinates = "2D"
    freq_max = 3000
    variables = POD.get_folderlist(path_results_pod, boolPrint=True)
    pod = POD(show_dashboard=False)
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
        path_results_pod2=".usv2",
        path_viz=path_viz,
    )

    pod.svd_correlation_signals(
        variables,
        maxmode=modelist,
        path_results_pod=path_results_pod,
        path_signals=".signals",
        path_viz=path_viz,
    )

    pod.correlation_signals(
        path_signals=".signals", path_signals2=".signals2", path_viz=path_viz
    )


if __name__ == "__main__":
    main()
