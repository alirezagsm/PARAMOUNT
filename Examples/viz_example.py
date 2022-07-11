from src.PARAMOUNT import POD


def main():
    """

    Inspect SVD results and visualize modes.

    Parameters Used
    ----------
    path_results_pod: path of folder containing SVD results
    path_viz: path to save visualization results
    modelist: list of mode numbers to process
    varlist: list of variables to consider
    dt: time step of data acquisition in seconds
    bounds: domain bounds for visualization [xmin, xmax, ymin, ymax, resolution]
    POD(): initialize a dask cluster on local system

    """
    path_results_pod = ".usv"
    path_viz = ".viz"
    modelist = range(11)
    dt = 4e-5
    bounds = [-0.075, 0.075, -0.05, 0.43, 2.5e-4]
    varlist = POD.get_folderlist(path_results_pod, boolPrint=True)
    variables = varlist
    pod = POD()
    pod.set_time(dt)
    pod.set_viz_params(dpi=600, linewidth=0.65, color="k", cmap="seismic")
    pod.svd_viz(variables, modelist, bounds, path_results_pod, path_viz=path_viz)


if __name__ == "__main__":
    main()
