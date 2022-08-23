from src.PARAMOUNT import POD


def main():
    """

    Read parquet database of desired variables, perform Proper Orthogonal Decomposition and store results in parquet format.

    Parameters Used
    ----------
    path_parquet: path to folder with parquet subfolders
    path_results_pod: path to save SVD results
    variables: list of variables to consider

    POD(): initialize a dask cluster on local system

    """
    path_parquet = ".data"
    path_results_pod = ".usv"
    variables = POD.get_folderlist(path_parquet, boolPrint=True)

    pod = POD(show_dashboard=True)

    pod.svd_save_usv(
        variables,
        path_parquet=path_parquet,
        path_results_pod=path_results_pod,
    )


if __name__ == "__main__":
    main()
