from src.PARAMOUNT import POD


def main():
    """

    Read csv datasets and perform Proper Orthogonal Decomposition.

    Parameters Used
    ----------
    path_csv: path to folder containing the csv files
    path_parquet: path to folder to store the converted parquet databases
    skiprows: skip rows in the csv data
    i_start: index of the first csv file starting at 0
    i_end: index of the last csv file. None means all the files
    varlist: list of variables to consider

    POD(): initialize a dask cluster on local system

    """

    path_csv = r".rawdata"
    path_parquet = ".data"
    skiprows = 5
    i_start = 1
    i_end = None
    varlist = POD.read_csv_columns(path_csv, skiprows=skiprows, boolPrint=True)
    variables = varlist
    pod = POD()

    pod.csv_to_parquet(
        variables,
        path_csv,
        path_parquet=path_parquet,
        i_start=i_start,
        i_end=i_end,
        skiprows=skiprows,
    )

    pod.client.shutdown()
    POD.check_parquet(path_parquet=path_parquet)


if __name__ == "__main__":
    main()
