from src.PARAMOUNT import POD


def main():
    """

    Read csv datasets and perform Proper Orthogonal Decomposition.

    Parameters Used
    ----------
    path_parquet: path to folder with parquet subfolders
    path_results_pod: path to save SVD results
    variables: list of variables to consider

    POD(): initialize a dask cluster on local system

    """
    path_parquet = ".data"
    path_results_pod = ".usv"
    varlist = POD.get_folderlist(path_parquet, boolPrint=True)
    variables = varlist
    pod = POD()
    pod.svd_save_usv(
        variables,
        path_parquet=path_parquet,
        path_results_pod=path_results_pod,
    )


if __name__ == "__main__":
    main()
