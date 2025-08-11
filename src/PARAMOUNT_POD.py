from src.PARAMOUNT_BASE import Base
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import dask.dataframe as dd
import dask.array as da
from dask import delayed
from tqdm import tqdm
import re
from src.utils import utils as utils


class POD(Base):
    """
    The POD class handles loading and processing CSV datasets for Proper Orthogonal Decomposition (POD) analysis.
    It supports reading CSV files in bulk, converting them to fast Parquet format, and performing Singular Value
    Decomposition (SVD) analysis. Results are stored in Parquet format for efficient processing, and pd.Series
    are serialized and stored in pickle format.

    Examples:
        Refer to `svd_example.py`, `csv_example.py`, and `viz_example.py` for usage examples.

    Key Features:
    -------------
    - Bulk CSV file reading and conversion to Parquet format.
    - SVD analysis with results stored in Parquet format.
    - Correlation analysis between variables and modes.
    - Visualization of SVD results and correlation heatmaps.
    - Extraction and sequential reading of CSV data.
    - Fast Fourier Transform (FFT) analysis of signals.
    - Support for 2D and 3D visualization of mode shapes.

    Available Functions:
    --------------------
    - read_csv_coordinates: Reads spatial coordinates from CSV files.
    - csv_to_parquet: Converts CSV files to Parquet format for faster processing.
    - extract_csv_sequential: Extracts data sequentially from a specific point.
    - read_csv_sequential: Reads sequential values from CSV files with unclean data.
    - check_parquet: Validates and prints the contents of Parquet files.
    - correlate: Computes correlation coefficients between two signals.
    - svd_save_usv: Performs distributed SVD and saves results.
    - svd_correlation: Generates correlation heatmaps between SVD modes.
    - correlation_signals: Computes correlation between signals in two folders.
    - svd_correlation_2X: Computes pairwise correlation maps between two SVD results.
    - svd_correlation_signals: Correlates SVD mode coefficients with external signals.
    - read_csv_columns: Reads and prints column headers from CSV files.
    - svd_viz: Visualizes SVD results, including mode shapes and coefficients.
    - s_viz_combined: Visualizes cumulative energy contributions of SVD modes.
    """

    def __init__(self, show_dashboard=False) -> None:
        super().__init__(show_dashboard)
        self.set_viz_params()

    def overwrite_guard(self, path, variables):
        """
        overwrite_guard ensures that existing folders are not accidentally overwritten.

        Args:
            path (str): The base path where the folders are located.
            variables (list): List of variable names corresponding to folder names.

        Returns:
            list: Updated list of variables after removing those the user chose not to overwrite.
        """
        remove_vars = []
        for var in variables:
            if var in self.get_folderlist(path=path, boolPrint=False):
                choice = input(
                    f"{var.strip()} folder already exists! overwrite existing files? [y/n] "
                )
                if choice.lower().strip() == "y":
                    shutil.rmtree(Path.cwd() / path / var)
                else:
                    remove_vars.append(var)

        for var in remove_vars:
            variables.remove(var)

        return variables

    def svd_save_usv(
        self,
        variables,
        path_parquet=".data",
        path_pod=".usv",
        dmd_X1=False,
        in_memory_df=None,
    ):
        """
        svd_save_usv computes distributed Singular Value Decomposition (SVD) and stores the results in Parquet format.

        Args:
            variables (list or str): List of variables to process.
            path_parquet (str, optional): Path to the Parquet datasets. Defaults to ".data".
            path_pod (str, optional): Path to store the SVD results. Defaults to ".usv".
            dmd_X1 (bool, optional): If True, ignores the last snapshot for the DMD procedure. Defaults to False.
            in_memory_df (DataFrame, optional): DataFrame to use in memory instead of reading from disk. Defaults to None.

        Raises:
            ValueError: Raised if existing folders are found and the user chooses not to overwrite them.
        """
        variables = variables if type(variables) is list else [variables]
        v_ = variables.copy()
        if in_memory_df is None:
            v_ = self.overwrite_guard(path_pod, v_)
        variables = v_

        try:
            shutil.copy(Path.cwd() / path_parquet / "x.pkl", path_pod)
            shutil.copy(Path.cwd() / path_parquet / "y.pkl", path_pod)
            shutil.copy(Path.cwd() / path_parquet / "z.pkl", path_pod)
        except:
            pass

        for var in tqdm(variables, "computing SVD modes"):
            path = Path.cwd() / path_parquet / f"{var}"
            # utils.ensure_dir(path)
            if in_memory_df is None:
                df = dd.read_parquet(path, engine="pyarrow")
            else:
                df = in_memory_df
            df = self.data_decimate(df, X1=dmd_X1)
            u, s, v = da.linalg.svd(df.values)

            utils.save_to_parquet(u, f"{path_pod}/{var}/u", column_dtype=str)
            utils.save_to_parquet(v, f"{path_pod}/{var}/v", column_dtype=str)
            result = dd.from_array(s).compute()
            utils.saveit(result, f"{path_pod}/{var}/s.pkl")

    def svd_correlation(self, variables, maxmode=5, path_pod=".usv", path_viz=".viz"):
        """
        svd_correlation generates a correlation heatmap between modes for specified variables.
        The heatmap displays Pearson's correlation coefficients between mode pairs:
        - The lower triangle shows the default Pearson's correlation.
        - The upper triangle shows the maximum correlation coefficient, accounting for time lag between mode pairs.

        Args:
            variables (list): List of variables to include in the correlation map.
            maxmode (int, optional): Maximum number of modes to consider in the correlation map. Defaults to 5.
            path_pod (str, optional): Path to read SVD results from. Defaults to ".usv".
            path_viz (str, optional): Path to store the generated plots. Defaults to ".viz".
        """

        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, FuncFormatter
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from itertools import combinations
        from scipy import signal

        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        utils.ensure_dir(path_viz)

        maxmode = max(list(maxmode))
        variables = variables if type(variables) is list else [variables]
        path_v = Path.cwd() / path_pod / f"{variables[0]}" / "v"
        df = dd.read_parquet(path_v, engine="pyarrow")
        df = df.compute().iloc[:maxmode, :].transpose()
        for v in variables[1:]:
            path_v = Path.cwd() / path_pod / f"{v}" / "v"
            df2 = dd.read_parquet(path_v, engine="pyarrow")
            df2 = df2.compute().iloc[:maxmode, :].transpose()
            df = pd.concat([df, df2], axis=1)
        df = df.dropna()
        results = []
        combs = list(combinations(range(len(df.columns)), 2))
        for comb in combs:
            v1 = df.iloc[:, comb[0]]
            v2 = df.iloc[:, comb[1]]
            _, corrcoef_adj = self.correlate(v1, v2)
            results.append(
                [
                    comb,
                    corrcoef_adj,
                    # lag * self.dt,
                ]
            )
        corr_adjusted = pd.DataFrame(np.nan, columns=df.columns, index=df.columns)
        for result in results:
            corr_adjusted.iat[result[0][0], result[0][1]] = result[1]
        corr = df.corr()
        corr = corr.where(np.tril(np.ones(corr.shape), k=0).astype(bool)).abs()
        fig, ax = plt.subplots()
        fig.set_size_inches(self.width * 2, self.width * 2)
        fig.patch.set_facecolor("w")

        ax2 = ax.imshow(corr, interpolation="none", aspect="equal")
        ax1 = ax.imshow(corr_adjusted, interpolation="none", aspect="equal")
        ax2.set_clim(0, 1)
        ax1.set_clim(0, 1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad="4%")
        cbar = fig.colorbar(ax1, cax=cax)
        cbar.ax.set_xlabel("Correlation Coefficient")
        for i in range(1, len(variables)):
            ax.axhline(i * maxmode - 0.5, color="w")
            ax.axvline(i * maxmode - 0.5, color="w")

        varlabels = []
        for var in variables:
            label = re.sub(r"\[(.*?)\]", "", var)
            label = re.sub(r"\.", " ", label)
            varlabels.append(label)

        majorloc = np.arange(
            np.floor(maxmode / 2),
            maxmode * len(variables) + np.floor(maxmode / 2),
            step=maxmode,
        )
        ax.xaxis.set_ticks(majorloc, varlabels)
        ax.yaxis.set_ticks(
            majorloc,
            varlabels,
            rotation=90,
            va="center",
        )
        ax.xaxis.set_minor_locator(AutoMinorLocator(maxmode))
        ax.yaxis.set_minor_locator(AutoMinorLocator(maxmode))

        indexskip = int((maxmode + 1) / 5)
        ax.xaxis.set_minor_formatter(
            FuncFormatter(
                lambda x, pos: (
                    f"{x % maxmode:.0f}"
                    if (x % maxmode) in np.arange(0, maxmode - 1, indexskip)
                    else ""
                )
            )
        )
        ax.yaxis.set_minor_formatter(
            FuncFormatter(
                lambda x, pos: (
                    f"{x % maxmode:.0f}"
                    if (x % maxmode) in np.arange(0, maxmode - 1, indexskip)
                    else ""
                )
            )
        )
        ax.xaxis.remove_overlapping_locs = False
        ax.yaxis.remove_overlapping_locs = False
        ax.tick_params(axis="both", which="both", length=7)
        ax.tick_params(axis="both", which="major", pad=20)

        plt.savefig(
            f"{path_viz}/correlation_{maxmode}" + ".png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close("all")

    def svd_correlation_2X(
        self,
        variables,
        maxmode=11,
        path_pod=".usv",
        path_pod2=".usv2",
        path_viz=".corr2X",
    ):
        """
        svd_correlation_2X generates a pairwise correlation map between two separate SVD results.

        Args:
            variables (list): List of variables to include in the correlation map.
            maxmode (int, optional): Maximum number of modes to consider in the correlation map. Defaults to 11.
            path_pod (str, optional): Path to the first SVD results folder. Will be shown as the y-axis. Defaults to ".usv".
            path_pod2 (str, optional): Path to the second SVD results folder. Will be shown as the x-axis. Defaults to ".usv2".
            path_viz (str, optional): Path to store the generated plots. Defaults to ".corr2X".
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, FuncFormatter
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from scipy import signal

        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        utils.ensure_dir(path_viz)

        variables = variables if type(variables) is list else [variables]

        path_v = Path.cwd() / path_pod / f"{variables[0]}" / "v"
        df = dd.read_parquet(path_v, engine="pyarrow")
        df = df.compute().iloc[:maxmode, :].transpose()
        for v in variables[1:]:
            path_v = Path.cwd() / path_pod / f"{v}" / "v"
            df2 = dd.read_parquet(path_v, engine="pyarrow")
            df2 = df2.compute().iloc[:maxmode, :].transpose()
            df = pd.concat([df, df2], axis=1)

        for v in variables:
            path_v = Path.cwd() / path_pod2 / f"{v}" / "v"
            df2 = dd.read_parquet(path_v, engine="pyarrow")
            df2 = df2.compute().iloc[:maxmode, :].transpose()
            df = pd.concat([df, df2], axis=1)

        df = df.dropna()
        len_ = int(len(df.columns) / 2)
        dfcorr_adjusted = pd.DataFrame(np.nan, columns=range(len_), index=range(len_))
        dfcorr = pd.DataFrame(np.nan, columns=range(len_), index=range(len_))
        for i in tqdm(range(len_), "computing correlations"):
            for j in range(len_):
                v1 = df.iloc[:, i]
                v2 = df.iloc[:, len_ + j]
                corrcoef, corrcoef_adj = self.correlate(v1, v2)
                dfcorr.iat[i, j] = corrcoef
                dfcorr_adjusted.iat[i, j] = corrcoef_adj

        diff = dfcorr_adjusted.T - dfcorr_adjusted
        diff = diff.where(np.triu(np.ones(diff.shape), k=0).astype(np.bool))

        for name, df in zip(
            ["corr", "corr_adjusted", "diff"], [dfcorr, dfcorr_adjusted, diff]
        ):
            fig, ax = plt.subplots()
            fig.set_size_inches(self.width * 2, self.width * 2)
            fig.patch.set_facecolor("w")

            ax1 = ax.imshow(df.abs(), interpolation="none", aspect="equal")
            ax1.set_clim(0, df.max().max())
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad="4%")
            cbar = fig.colorbar(ax1, cax=cax)
            cbar.ax.set_xlabel("Correlation Coefficient")
            for i in range(1, len(variables)):
                ax.axhline(i * maxmode - 0.5, color="w")
                ax.axvline(i * maxmode - 0.5, color="w")

            varlabels = []
            for var in variables:
                label = re.sub(r"\[(.*?)\]", "", var)
                label = re.sub(r"\.", " ", label)
                varlabels.append(label)

            majorloc = np.arange(
                np.floor(maxmode / 2),
                maxmode * len(variables) + np.floor(maxmode / 2),
                step=maxmode,
            )
            ax.xaxis.set_ticks(majorloc, varlabels)
            ax.yaxis.set_ticks(
                majorloc,
                varlabels,
                rotation=90,
                va="center",
            )
            ax.xaxis.set_minor_locator(AutoMinorLocator(maxmode))
            ax.yaxis.set_minor_locator(AutoMinorLocator(maxmode))

            indexskip = int((maxmode + 1) / 5)
            ax.xaxis.set_minor_formatter(
                FuncFormatter(
                    lambda x, pos: (
                        f"{x % maxmode:.0f}"
                        if (x % maxmode) in np.arange(0, maxmode - 1, indexskip)
                        else ""
                    )
                )
            )
            ax.yaxis.set_minor_formatter(
                FuncFormatter(
                    lambda x, pos: (
                        f"{x % maxmode:.0f}"
                        if (x % maxmode) in np.arange(0, maxmode - 1, indexskip)
                        else ""
                    )
                )
            )
            ax.xaxis.remove_overlapping_locs = False
            ax.yaxis.remove_overlapping_locs = False
            ax.tick_params(axis="both", which="both", length=7)
            ax.tick_params(axis="both", which="major", pad=20)

            plt.savefig(
                f"{path_viz}/2X_{name}_{maxmode}" + ".png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close("all")

    def svd_correlation_signals(
        self,
        variables,
        maxmode=5,
        path_pod=".usv",
        path_signals=".signals",
        path_viz=".viz",
    ):
        """
        svd_correlation_signals generates a correlation heatmap between signals in a folder
        and the SVD mode coefficients. The signals are assumed to be in pickled format.

        Args:
            variables (list): List of variables to include in the correlation map.
            maxmode (int, optional): Maximum number of modes to consider in the correlation map. Defaults to 5.
            path_pod (str, optional): Path to read SVD results from. Defaults to ".usv".
            path_signals (str, optional): Path to the folder containing the set of signals. Defaults to ".signals".
            path_viz (str, optional): Path to save the generated results. Defaults to ".viz".
        """
        import matplotlib.pyplot as plt
        from matplotlib.ticker import AutoMinorLocator, FuncFormatter
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        utils.ensure_dir(path_viz)

        variables = variables if type(variables) is list else [variables]

        path_v = Path.cwd() / path_pod / f"{variables[0]}" / "v"
        df = dd.read_parquet(path_v, engine="pyarrow")
        df = df.compute().iloc[:maxmode, :].transpose()
        for v in variables[1:]:
            path_v = Path.cwd() / path_pod / f"{v}" / "v"
            df2 = dd.read_parquet(path_v, engine="pyarrow")
            df2 = df2.compute().iloc[:maxmode, :].transpose()
            df = pd.concat([df, df2], axis=1)
        df = df.dropna()

        pathlist = Path(path_signals).resolve().glob("*")
        signals = []
        for path in pathlist:
            signals.append(pd.Series(utils.loadit(path), name=path.name))
        signaldf = pd.concat(signals, axis=1)
        dfcorr_adjusted = pd.DataFrame(
            np.nan, columns=range(signaldf.shape[1]), index=range(df.shape[1])
        )
        dfcorr = pd.DataFrame(
            np.nan, columns=range(signaldf.shape[1]), index=range(df.shape[1])
        )

        for i in tqdm(range(df.shape[1]), "computing correlations"):
            for j in range(signaldf.shape[1]):
                v1 = df.iloc[:, i]
                v2 = signaldf.iloc[:, j]
                corrcoef, corrcoef_adj = self.correlate(v1, v2)
                dfcorr.iat[i, j] = corrcoef
                dfcorr_adjusted.iat[i, j] = corrcoef_adj

        for name, df in zip(["corr", "corr_adjusted"], [dfcorr, dfcorr_adjusted]):
            fig, ax = plt.subplots()
            fig.set_size_inches(
                self.width * signaldf.shape[1] / df.shape[1], self.width * 3
            )
            fig.patch.set_facecolor("w")

            ax1 = ax.imshow(df.abs(), interpolation="none", aspect="equal")
            ax1.set_clim(0, 1)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                "right",
                size=self.width * 5 / 100,
                pad=self.width * 2 / 100,
            )
            cbar = fig.colorbar(ax1, cax=cax)
            cbar.ax.set_xlabel("Correlation Coefficient")

            for i in range(1, len(variables)):
                ax.axhline(i * maxmode - 0.5, color="w")
            for i in range(1, 1 + len(signaldf.columns)):
                ax.axvline(i - 0.5, color="w")

            varlabels = []
            for var in variables:
                label = re.sub(r"\[(.*?)\]", "", var)
                label = re.sub(r"\.", " ", label)
                varlabels.append(label)

            majorloc = np.arange(
                np.floor(maxmode / 2),
                maxmode * len(variables) + np.floor(maxmode / 2),
                step=maxmode,
            )
            sigloc = np.arange(
                0,
                signaldf.shape[1],
                step=1,
            )
            ax.xaxis.set_ticks(
                sigloc,
                signaldf.columns,
                rotation=90,
                ha="center",
            )
            ax.yaxis.set_ticks(
                majorloc,
                varlabels,
                rotation=90,
                va="center",
            )
            ax.yaxis.set_minor_locator(AutoMinorLocator(maxmode))

            indexskip = int((maxmode + 1) / 5)
            ax.yaxis.set_minor_formatter(
                FuncFormatter(
                    lambda x, pos: (
                        f"{x % maxmode:.0f}"
                        if (x % maxmode) in np.arange(0, maxmode - 1, indexskip)
                        else ""
                    )
                )
            )
            ax.xaxis.remove_overlapping_locs = True
            ax.yaxis.remove_overlapping_locs = False
            ax.tick_params(axis="y", which="both", length=7)
            ax.tick_params(axis="y", which="major", pad=20)

            plt.savefig(
                f"{path_viz}/Sig_{name}_{maxmode}" + ".png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close("all")

    def get_units(self, var):
        """
        Extract units from a variable name.

        Args:
            var (str): The variable name containing units in square brackets.

        Returns:
            str: The extracted units if found, otherwise an empty string.
        """
        match = re.search(r"\[(.*?)\]", var)
        if match:
            return match.group(1)
        else:
            return " "

    def average_viz(
        self,
        variables,
        path_data=".data",
        path_viz=".viz",
        coordinates="2D",
        cbar=None,
        dist=False,
        bounds="auto",
    ):
        """
        average_viz visualizes the average of specified variables over a spatial domain.

        Args:
            variables (list or str): List of variables to visualize or a single variable as a string.
            path_data (str, optional): Path to the folder containing the data. Defaults to ".data".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            coordinates (str, optional): Specifies the dimensionality of the data ("2D" or "3D"). Defaults to "2D".
            cbar (str, optional): Color bar range for visualization. Defaults to None.
            dist (bool, optional): Whether to apply a distance-based mask for visualization. Defaults to False.
            bounds (str or list, optional): Domain bounds for visualization. If "auto", bounds are calculated automatically. Defaults to "auto".
        """
        variables = variables if type(variables) is list else [variables]
        self.make_dim(coordinates)
        for var in tqdm(variables, "analyzing variables"):
            utils.ensure_dir(f"{path_viz}/{var}")
            path_u = Path.cwd() / path_data / f"{var}"
            u = dd.read_parquet(path_u, engine="pyarrow")
            u = u.mean(axis=1)
            cbar_label = self.get_units(var)

            if self.dim == "xy":
                x = utils.loadit(Path.cwd() / path_data / "x.pkl")
                y = utils.loadit(Path.cwd() / path_data / "y.pkl")
                if bounds == "auto":
                    bounds = self.make_bounds([x, y])

                if dist:
                    dist_map = self.dist_map(x, y, bounds)

                self.u_viz(
                    x,
                    y,
                    u,
                    f"{path_viz}/{var}",
                    [0],
                    bounds,
                    dist,
                    dist_map,
                    cbar=cbar,
                    cbar_label=cbar_label,
                )

            if self.dim == "xyz":
                x = utils.loadit(Path.cwd() / path_data / "x.pkl")
                y = utils.loadit(Path.cwd() / path_data / "y.pkl")
                z = utils.loadit(Path.cwd() / path_data / "z.pkl")
                if bounds == "auto":
                    bounds = self.make_bounds([x, y, z])

                self.u_viz_3d(
                    x,
                    y,
                    z,
                    u,
                    f"{path_viz}/{var}",
                    [0],
                    bounds,
                )

    def svd_viz(
        self,
        variables,
        modelist,
        bounds="auto",
        coordinates="2D",
        path_pod=".usv",
        path_viz=".viz",
        freq_max=3000,
        dist=False,
        cbar=None,
    ):
        """
        svd_viz visualizes the results of Singular Value Decomposition (SVD).

        Args:
            variables (list or str): List of variables to visualize.
            modelist (list): List of mode indices to visualize.
            bounds (list): Domain bounds for visualization in the format
            [xmin, xmax, ymin, ymax, resolution].
            coordinates (str): Specifies the dimensionality of the visualization ("2D" or "3D").
            path_pod (str, optional): Path to read SVD results from. Defaults to ".usv".
            path_viz (str, optional): Path to store the generated plots. Defaults to ".viz".
            freq_max (float, optional): Maximum frequency of interest for Power Spectral Density (PSD) plots.
            Defaults to 3000 Hz.
            dist (float or bool, optional): Distance threshold to mask the xy meshgrid using the k-d tree method.
            If False, no masking is applied. Defaults to False.
        """
        variables = variables if type(variables) is list else [variables]
        modelist = modelist if type(modelist) is list else list(modelist)

        self.make_dim(coordinates)

        for var in tqdm(variables, "analyzing variables"):
            utils.ensure_dir(f"{path_viz}/{var}")

            path_u = Path.cwd() / path_pod / f"{var}" / "u"
            path_v = Path.cwd() / path_pod / f"{var}" / "v"
            path_s = Path.cwd() / path_pod / f"{var}" / "s.pkl"
            u = dd.read_parquet(path_u, engine="pyarrow")
            v = dd.read_parquet(path_v, engine="pyarrow")
            s = utils.loadit(path_s)

            if self.dim == "xy":
                path_x = Path.cwd() / path_pod / "x.pkl"
                path_y = Path.cwd() / path_pod / "y.pkl"
                x = utils.loadit(path_x)
                y = utils.loadit(path_y)

                if bounds == "auto":
                    bounds = self.make_bounds([x, y])

                if dist:
                    dist_map = self.dist_map(x, y, bounds)

                self.u_viz(
                    x,
                    y,
                    u,
                    f"{path_viz}/{var}",
                    modelist,
                    bounds,
                    dist,
                    dist_map,
                    cbar=cbar,
                )

            if self.dim == "xyz":
                path_x = Path.cwd() / path_pod / "x.pkl"
                path_y = Path.cwd() / path_pod / "y.pkl"
                path_z = Path.cwd() / path_pod / "z.pkl"
                x = utils.loadit(path_x)
                y = utils.loadit(path_y)
                z = utils.loadit(path_z)

                if bounds == "auto":
                    bounds = self.make_bounds([x, y, z])

                self.u_viz_3d(
                    x,
                    y,
                    z,
                    u,
                    f"{path_viz}/{var}",
                    modelist,
                    bounds,
                )
            self.v_viz(
                v,
                f"{path_viz}/{var}",
                modelist,
                freq_max,
            )
            self.s_viz(s, f"{path_viz}/{var}")

    def s_viz_combined(
        self,
        variables,
        maxmode=100,
        path_pod=".usv",
        path_viz=".viz",
    ):
        """
        s_viz_combined visualizes the cumulative energy contributions of all singular values (s)
        for the specified variables in a single plot.

        Args:
            variables (list or str): List of variables to include in the visualization.
            maxmode (int): The maximum number of modes to consider. Defaults to 100.
            path_pod (str, optional): Path to the folder containing SVD results. Defaults to ".usv".
            path_viz (str, optional): Path to save the generated plot. Defaults to ".viz".
        """

        variables = variables if type(variables) is list else [variables]

        utils.ensure_dir(path_viz)

        s_combined = pd.DataFrame(columns=variables)
        for var in variables:
            path_s = Path.cwd() / path_pod / f"{var}" / "s.pkl"
            s = utils.loadit(path_s)

            mode_energy = [x**2 for x in s]
            mode_energy = [x / sum(mode_energy) * 100 for x in mode_energy]
            cumsum = np.cumsum(mode_energy)
            s_combined[var] = cumsum[:maxmode]

        self.s_viz_combined_plot(s_combined, f"{path_viz}")

    def u_viz_3d(
        self,
        x,
        y,
        z,
        u,
        path_viz,
        modelist,
        bounds,
    ):
        """
        u_viz_3d performs 3D visualization of SVD mode shapes.

        Args:
            x (list): List of x-coordinate values.
            y (list): List of y-coordinate values.
            z (list): List of z-coordinate values.
            u (DataFrame): U matrix from SVD analysis.
            path_viz (str): Path to store the generated plots.
            modelist (list): List of mode indices to visualize.
            bounds (list): Visualization boundaries and resolution in the format
            [xmin, xmax, ymin, ymax, zmin, zmax, resolution].
        """
        from scipy.interpolate import griddata
        import plotly.graph_objects as go
        import plotly.io as pio
        from ipywidgets import (
            interact,
            FloatSlider,
            FloatRangeSlider,
            Layout,
            Button,
        )
        from IPython.display import display
        from IPython.core.interactiveshell import InteractiveShell

        InteractiveShell.ast_node_interactivity = "all"

        pio.templates["custom"] = go.layout.Template(
            layout=go.Layout(
                font_family=self.font,
                font_size=self.fontsize,
            )
        )
        pio.templates.default = "simple_white+custom"

        xmin, xmax, ymin, ymax, zmin, zmax, res = bounds
        xx, yy, zz = self.make_meshgrid(bounds)

        figs = [[] for x in range(max(modelist) + 1)]

        for mode in tqdm(modelist, "creating plots", leave=False):
            uu = u.iloc[:, mode].compute()
            kk = griddata(
                (x, z, y),
                uu,
                (xx, zz, yy),
                method="linear",
                fill_value=min(abs(uu)),
            )
            kmin = kk.min()
            kmax = kk.max()
            krng = kmax - kmin
            kmean = kk.mean()
            stp = krng / 100
            kwargs = dict(layout=Layout(width="700px"), readout_format=".4f")
            fig = go.FigureWidget(
                data=go.Isosurface(
                    x=xx.flatten(),
                    z=yy.flatten(),
                    y=zz.flatten(),
                    value=kk.flatten(),
                    isomin=kk.min(),
                    isomax=kk.max(),
                    surface_count=2,
                    opacity=1,
                    colorscale="bluered",
                    caps=dict(x_show=False, y_show=False, z_show=False),
                    showscale=False,
                    flatshading=False,
                    lighting=dict(ambient=0.65, specular=0.8),
                )
            )

            fig.update_layout(
                width=600,
                height=400 * (ymax - ymin) / (xmax - xmin),
                scene=dict(
                    xaxis=dict(
                        gridcolor="lightgray",
                        showticklabels=False,
                        title="",
                        showgrid=True,
                        range=[xmin, xmax],
                    ),
                    yaxis=dict(
                        gridcolor="lightgray",
                        showticklabels=False,
                        title="",
                        showgrid=True,
                        range=[zmin, zmax],
                    ),
                    zaxis=dict(
                        gridcolor="lightgray",
                        showticklabels=False,
                        title="",
                        showgrid=True,
                        range=[ymin, ymax],
                    ),
                    aspectmode="data",
                    camera=dict(eye=dict(x=1.35, y=1.35, z=1.35)),
                ),
                margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=0),
            )
            figs[mode] = fig
            print(f"fig_id = {mode}")

            @interact(
                fig_id=modelist,
                surface_count=(1, 20, 1),
                opacity=FloatSlider(value=0.5, min=0, max=1, step=0.01, **kwargs),
                iso=FloatSlider(value=kmean, min=kmin, max=kmax, step=stp, **kwargs),
                iso_spread=FloatSlider(value=5, min=0, max=100, step=0.05, **kwargs),
                iso_limits=FloatRangeSlider(
                    value=(kmin, kmax), min=kmin, max=kmax, step=stp, **kwargs
                ),
            )
            def update(
                opacity=0.5,
                surface_count=2,
                iso=kmean,
                iso_spread=5,
                fig_id=mode,
                iso_limits=(kmin, kmax),
            ):
                with fig.batch_update():
                    figs[fig_id].data[0].surface.count = surface_count
                    figs[fig_id].data[0].opacity = opacity
                    l_bound = max(iso_limits[0], iso - krng * iso_spread / 100)
                    h_bound = min(iso_limits[1], iso + krng * iso_spread / 100)
                    figs[fig_id].data[0].isomin = l_bound
                    figs[fig_id].data[0].isomax = h_bound
                    figs[fig_id].data[0].cmin = l_bound
                    figs[fig_id].data[0].cmax = h_bound

            def buttonfunction(btn):
                figs[int(btn.description[5:])].write_image(
                    f"{path_viz}/u{btn.description[5:]}_3D" + ".png", scale=4
                )

            savebutton = Button(description=f"save {mode}", button_style="primary")
            savebutton.on_click(buttonfunction)

            display(savebutton, fig)

    def u_viz(
        self,
        x,
        y,
        u,
        path_viz,
        modelist,
        bounds,
        dist,
        dist_map,
        vmax=None,
        vmin=None,
        cbar=False,
        cbar_label="value",
    ):
        """
        u_viz performs 2D visualization of SVD mode shapes.

        Args:
            x (Series): x-coordinate values.
            y (Series): y-coordinate values.
            u (DataFrame or Series): U matrix or vector from SVD analysis.
            path_viz (str): Path to store the generated plots.
            modelist (list): List of mode indices to visualize.
            bounds (list): Visualization boundaries and resolution in the format
            [xmin, xmax, ymin, ymax, resolution].
            dist (float or None): Distance threshold to mask the visualization using the k-d tree method.
            If None, no masking is applied.
            dist_map (ndarray): k-d tree distance map for coordinates.
            vmax (float or None): Maximum value for the color scale. If None, it is determined automatically.
            vmin (float or None): Minimum value for the color scale. If None, it is determined automatically.
            cbar (bool): Whether to include a color bar in the visualization. Defaults to False.
            cbar_label (str): Label for the color bar. Defaults to "value".
        """
        from scipy.interpolate import griddata
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        import matplotlib.ticker as ticker

        plt.switch_backend("agg")
        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        xmin, xmax, ymin, ymax, res = bounds
        xx, yy = self.make_meshgrid(bounds)

        # for mode in tqdm(modelist, "plotting 2D mode shapes", leave=False):
        @delayed
        def _plot(mode):
            if isinstance(u, dd.DataFrame):
                uu = u.iloc[:, mode].compute()
            elif isinstance(u, dd.Series):
                uu = u.compute()
            elif isinstance(u, pd.DataFrame):
                uu = u.iloc[:, mode]
            elif isinstance(u, pd.Series):
                uu = u
            else:
                raise ("u must be a pandas or dask dataframe")
            kk = griddata(
                (x, y),
                uu,
                (xx, yy),
                method="linear",
                fill_value=min(abs(uu)),
            )
            if dist is not None:
                # Mask regions where the distance between points exceeds the specified threshold
                kk = np.where(dist_map >= dist, np.nan, kk)
                kk[dist_map >= dist] = np.nan

            fig, ax = plt.subplots(1)
            fig.set_size_inches(self.width, self.height)
            fig.patch.set_facecolor("w")
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.axes.xaxis.set_visible(False)
            ax.axes.yaxis.set_visible(False)
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.set_aspect(1)
            ax.set_axisbelow(True)
            ax.grid(alpha=0.5)
            kk[np.isnan(kk)] = np.min(abs(kk))
            if vmax is None:
                contour = ax.contourf(
                    xx,
                    yy,
                    kk,
                    self.contour_levels,
                    cmap=self.cmap,
                    antialiased=True,
                )
            else:
                contour = ax.contourf(
                    xx,
                    yy,
                    kk,
                    self.contour_levels,
                    cmap=self.cmap,
                    antialiased=True,
                    vmax=vmax,
                    vmin=vmin,
                )

            nonlocal cbar
            if cbar:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="4%", pad="4%")
                cbar = fig.colorbar(contour, cax=cax)
                cbar.ax.set_xlabel(cbar_label, loc="left")

                ticklabs = cbar.ax.get_yticklabels()
                for t in ticklabs:
                    t.set_horizontalalignment("left")

                formatter = ticker.ScalarFormatter(useMathText=False)
                formatter.set_scientific(True)
                formatter.set_powerlimits((-3, 3))
                cbar.ax.yaxis.set_major_formatter(formatter)
                cbar.ax.yaxis.set_tick_params(pad=6)
                cbar.ax.yaxis.offsetText.set_ha("right")
                cbar.ax.xaxis.set_label_coords(0, -0.1)

            contour.set_edgecolor("face")
            fig.tight_layout()
            for axis in ["top", "bottom", "left", "right"]:
                ax.spines[axis].set_linewidth(self.ax_width)
            fig.savefig(
                f"{path_viz}/u{mode}" + ".png",
                dpi=self.dpi,
                bbox_inches="tight",
            )

            plt.close("all")

        tasks = []
        for mode in tqdm(modelist, "plotting 2D mode shapes", leave=False):
            tasks.append(_plot(mode))
        import dask
        from dask.diagnostics import ProgressBar

        num_threads = len(self.client.nthreads()) * 2
        with ProgressBar():
            dask.compute(tasks, scheduler="threads", num_workers=num_threads)

    def v_viz(self, v, path_viz, modelist, freq_max):
        """
        v_viz visualizes the mode coefficients (v matrix) from SVD analysis.

        Args:
            v (DataFrame): The v matrix from SVD analysis.
            path_viz (str): Path to save the visualization results.
            modelist (list): List of mode indices to visualize.
            freq_max (int): Maximum frequency to consider in Power Spectral Density (PSD) plots.
        """
        from scipy.signal import find_peaks

        import matplotlib.pyplot as plt
        import matplotlib.mlab as mlab

        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)
        for mode in tqdm(modelist, "plotting mode coefficients", leave=False):
            vv = v.compute().iloc[mode, :]
            tt = np.arange(self.t0, vv.shape[0] * self.dt, self.dt)

            fig, ax = plt.subplots(1)
            fig.set_size_inches(self.width, self.height)
            fig.patch.set_facecolor("w")
            ax.set_xlabel("Time [s]")
            ax.set_ylabel("Coefficient")
            ax.grid(alpha=0.5)
            ax.set_xlim(tt[0], tt[-1])
            ax.plot(tt, vv, self.color, linewidth=self.linewidth)

            fig.tight_layout()
            for axis in ["bottom", "left"]:
                ax.spines[axis].set_linewidth(self.ax_width)
            for axis in ["top", "right"]:
                ax.spines[axis].set_linewidth(0)
            plt.savefig(
                f"{path_viz}/v{mode}" + ".png", dpi=self.dpi, bbox_inches="tight"
            )

            fig, ax = plt.subplots(1)
            fig.set_size_inches(self.width, self.height)
            fig.patch.set_facecolor("w")
            ax.set_xlabel("Frequency [Hz]")
            ax.set_ylabel("Power Spectral Density [db/Hz]")
            ax.grid(alpha=0.5)
            ax.set_xlim(0, freq_max)

            Pxx, freqs = mlab.psd(
                vv,
                Fs=1 / self.dt,
                window=mlab.window_hanning,
                detrend="linear",
            )
            freqs = freqs[np.where(freqs < freq_max)]
            Pxx = Pxx[: len(freqs)]
            dbPxx = 10 * np.log10(Pxx)
            peaks, _ = find_peaks(dbPxx, prominence=3)
            ax.plot(freqs, dbPxx, self.color, linewidth=self.linewidth)
            npeaks = 3
            for n in range(0, min(npeaks, len(peaks))):
                ax.scatter(
                    freqs[peaks[n]],
                    dbPxx[peaks[n]],
                    s=80,
                    facecolors="none",
                    edgecolors="grey",
                )
                acc = 0
                if freq_max < 10:
                    acc = 2
                ax.annotate(
                    f"{freqs[peaks[n]]:0.{acc}f}",
                    xy=(freqs[peaks[n]] + freq_max / 25, dbPxx[peaks[n]] * 0.99),
                )
            fig.tight_layout()
            for axis in ["bottom", "left"]:
                ax.spines[axis].set_linewidth(self.ax_width)
            for axis in ["top", "right"]:
                ax.spines[axis].set_linewidth(0)
            plt.savefig(
                f"{path_viz}/v{mode}_PSD" + ".png", dpi=self.dpi, bbox_inches="tight"
            )
            plt.close("all")

    def s_viz(self, s, path_viz, maxmode=100):
        """
        s_viz visualizes the cumulative energy contributions of the singular values (s)
        from the SVD analysis.

        Args:
            s (list or array-like): Singular values from the SVD analysis.
            path_viz (str): Path to save the visualization results.
            maxmode (int, optional): Maximum number of modes to visualize. Defaults to 100.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.width, self.height)
        fig.patch.set_facecolor("w")
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Cumulative share of mode energy")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_axisbelow(True)
        ax.grid(alpha=0.5, which="both")

        mode_energy = [x**2 for x in s]
        mode_energy = [x / sum(mode_energy) * 100 for x in mode_energy]
        cumsum = np.cumsum(mode_energy)
        s = cumsum[:maxmode]
        ax.set_ylim(s[0] - 10, 100.5)
        ax.set_xlim(0, maxmode)
        ax.plot(s, self.color, linewidth=self.linewidth)
        fig.tight_layout()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(self.ax_width)
        for axis in ["top", "right"]:
            ax.spines[axis].set_linewidth(0)
        plt.savefig(f"{path_viz}/s" + ".png", dpi=self.dpi, bbox_inches="tight")
        plt.close("all")

    def s_viz_combined_plot(self, s, path_viz):
        """
        s_viz_combined_plot visualizes the cumulative energy contributions of all singular values (s)
        for the specified variables in a single combined plot.

        Args:
            s (pd.DataFrame): DataFrame where each column contains the cumulative energy contributions
                      of singular values for a specific variable.
            path_viz (str): Path to save the generated plot.
        """
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        plt.switch_backend("agg")
        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        clrs_list = ["k", "b", "g", "r"]
        styl_list = ["-", "--", "-.", ":"]

        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.width, self.height)
        fig.patch.set_facecolor("w")
        ax.set_xlabel("Mode number")
        ax.set_ylabel("Cumulative share of mode energy")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_axisbelow(True)
        ax.grid(alpha=0.5, which="both")

        ax.set_ylim(min(s.min(axis=1)) - 10, 100)
        ax.set_xlim(0, s.shape[0])
        for i, var in enumerate(s.columns):
            label = re.sub(r"\[(.*?)\]", "", var)
            label = re.sub(r"\.", " ", label)
            clrr = clrs_list[i // 4]
            styl = styl_list[i % 4]
            ax.plot(s[var], linewidth=self.linewidth, label=label, color=clrr, ls=styl)

        ax.legend(fontsize="small")
        fig.tight_layout()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(self.ax_width)
        for axis in ["top", "right"]:
            ax.spines[axis].set_linewidth(0)
        plt.savefig(
            f"{path_viz}/s_combined" + ".png", dpi=self.dpi, bbox_inches="tight"
        )
        plt.close("all")
