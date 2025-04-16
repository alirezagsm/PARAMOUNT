from src.PARAMOUNT_POD import POD
from pathlib import Path
import shutil
from tqdm import tqdm
import dask.dataframe as dd
import dask.array as da
import numpy as np
import pandas as pd
import re
import src.utils as utils


class DMD(POD):
    """
    The DMD class extends the POD class to implement Dynamic Mode Decomposition (DMD) analysis.
    It provides methods for computing DMD modes, eigenvalues, and predictions, as well as
    visualizing the results. The class also supports multi-resolution DMD (mrDMD) for analyzing
    data at different temporal scales.

    Key Features:
    -------------
    - Compute DMD modes, eigenvalues, and predictions.
    - Multi-resolution DMD (mrDMD) for hierarchical temporal analysis.
    - Visualization of DMD modes, eigenvalues, and predictions.
    - Error analysis and comparison between DMD predictions and original data.

    Available Functions:
    --------------------
    - stack: Combines multiple variables into a single stacked dataset.
    - save_Atilde: Computes and saves the reduced-order Atilde matrix.
    - save_modes: Computes and saves DMD modes and initial coefficients.
    - save_prediction: Predicts future states using DMD and saves the results.
    - describe_parquet: Provides statistical descriptions of Parquet datasets.
    - viz_error: Visualizes prediction errors.
    - viz_error_combined: Combines and visualizes errors for multiple variables.
    - viz_parquet: Visualizes snapshots of Parquet datasets.
    - viz_eigs_circle: Visualizes eigenvalues on the unit circle.
    - viz_eigs_spectrum: Visualizes the spectrum of eigenvalues.
    - viz_modes: Visualizes DMD modes in 2D or 3D.
    - multires: Performs multi-resolution DMD (mrDMD) analysis.
    - multires_predict: Combines predictions from all mrDMD levels.
    - viz_multires: Visualizes results from mrDMD analysis.

    Examples:
    ---------
    Refer to the provided examples for usage of DMD and mrDMD methods.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_cutoff = None

    def stack(self, variables, path_parquet=".data", stack_idx=None):
        """
        stack combines multiple variables into a single stacked dataset.

        Args:
            variables (list or str): List of variable names to stack or a single variable name as a string.
            path_parquet (str, optional): Path to the folder containing the Parquet datasets. Defaults to ".data".
            stack_idx (list, optional): List of indices specifying which variables to stack. If None, all variables are stacked. Defaults to None.
        """
        variables = variables if type(variables) is list else [variables]
        if len(variables) == 1:
            print("only one variable found, stacking not possible")
            return
        if stack_idx is not None:
            vars = [variables[i] for i in stack_idx]
        path_var0 = Path.cwd() / path_parquet / f"{vars[0]}"
        df = dd.read_parquet(path_var0, engine="pyarrow")
        for var in vars[1:]:
            path_var = Path.cwd() / path_parquet / f"{var}"
            df2 = dd.read_parquet(path_var, engine="pyarrow")
            # df = dd.concat(
            #     [df.loc[:: len(variables)], df2.loc[:: len(variables)]], axis=0
            # )
            df = dd.concat([df, df2], axis=0)

        stack_name = "_".join(vars)
        self.stack_name = f"stack_{len(vars)}_{stack_name}"
        df.to_parquet(
            f"{path_parquet}/{self.stack_name}",
            compression="snappy",
            write_metadata_file=True,
        )

    def save_Atilde(
        self,
        variables,
        path_parquet=".data",
        path_pod=".usv",
        path_dmd=".dmd",
        in_memory_df=None,
    ):
        """
        save_Atilde computes and saves the reduced-order Atilde matrix for Dynamic Mode Decomposition (DMD).

        Args:
            variables (list or str): List of variables to process or a single variable as a string.
            path_parquet (str, optional): Path to the folder containing the Parquet datasets. Defaults to ".data".
            path_pod (str, optional): Path to the folder containing the SVD results. Defaults to ".usv".
            path_dmd (str, optional): Path to the folder where the Atilde matrix will be saved. Defaults to ".dmd".
            in_memory_df (DataFrame, optional): DataFrame to use in memory instead of reading from disk. Defaults to None.
        """
        variables = variables if type(variables) is list else [variables]
        v_ = variables.copy()

        if in_memory_df is None:
            v_ = self.overwrite_guard(path_dmd, v_)
        variables = v_

        try:
            shutil.copy(Path.cwd() / path_parquet / "x.pkl", path_dmd)
            shutil.copy(Path.cwd() / path_parquet / "y.pkl", path_dmd)
            shutil.copy(Path.cwd() / path_parquet / "z.pkl", path_dmd)
        except:
            pass

        for var in tqdm(variables, "computing Atilde matrix"):
            path_var = Path.cwd() / path_parquet / f"{var}"
            if in_memory_df is None:
                df = df = dd.read_parquet(path_var, engine="pyarrow").to_dask_array()
            else:
                df = in_memory_df.to_dask_array(lengths=True)

            df2 = self.data_decimate(df, X2=True)
            df2.compute_chunk_sizes()
            path_u = Path.cwd() / path_pod / f"{var}" / "u"
            path_s = Path.cwd() / path_pod / f"{var}" / "s.pkl"
            path_v = Path.cwd() / path_pod / f"{var}" / "v"
            u = utils.load_from_parquet(path_u)
            s = utils.loadit(path_s)
            v = utils.load_from_parquet(path_v)
            u = u.to_dask_array()
            u.compute_chunk_sizes()
            v = v.to_dask_array()
            v.compute_chunk_sizes()

            Atilde = da.matmul(
                da.matmul(da.matmul(u.transpose().conj(), df2), v.transpose().conj()),
                da.diag(s.values**-1),
            )
            utils.saveit(Atilde.compute(), f"{path_dmd}/{var}/Atilde.pkl")

    def get_init(self, path, index=0):
        """
        get_init retrieves the initial snapshot (column) from a Parquet dataset.

        Args:
            path (str): Path to the Parquet dataset.
            index (int, optional): Index of the snapshot to retrieve. Defaults to 0.

        Returns:
            dask.array.Array: The specified snapshot as a Dask array.
        """
        df = dd.read_parquet(path, engine="pyarrow").to_dask_array()
        return df.T[index]

    def save_modes(
        self,
        variables,
        path_parquet=".data",
        path_dmd=".dmd",
        path_pod=".usv",
        projection_method=True,
        in_memory_df=None,
    ):
        """
        save_modes computes and saves the DMD modes and initial coefficients.

        Args:
            variables (list or str): List of variables to process or a single variable as a string.
            path_parquet (str, optional): Path to the folder containing the Parquet datasets. Defaults to ".data".
            path_dmd (str, optional): Path to the folder where the DMD results will be saved. Defaults to ".dmd".
            path_pod (str, optional): Path to the folder containing the SVD results. Defaults to ".usv".
            projection_method (bool, optional): If True, uses the projection method to compute the modes.
                            If False, uses the direct method. Defaults to True.
            in_memory_df (DataFrame, optional): DataFrame to use in memory instead of reading from disk. Defaults to None.
        """
        variables = variables if type(variables) is list else [variables]

        for var in tqdm(variables, "computing DMD modes and coefficients"):
            path_Atilde = Path.cwd() / path_dmd / f"{var}" / "Atilde.pkl"
            Atilde = utils.loadit(path_Atilde)
            Lambda, eigvecs = np.linalg.eig(Atilde)
            path_u = Path.cwd() / path_pod / f"{var}" / "u"
            path_v = Path.cwd() / path_pod / f"{var}" / "v"
            # u = dd.read_parquet(path_u, engine="pyarrow")
            u = utils.load_from_parquet(path_u)
            columns = u.columns
            # v = dd.read_parquet(path_v, engine="pyarrow")
            v = utils.load_from_parquet(path_v)
            u = u.to_dask_array()
            u.compute_chunk_sizes()
            v = v.to_dask_array()
            v.compute_chunk_sizes()

            path_var = Path.cwd() / path_parquet / f"{var}"

            if in_memory_df is None:
                df = dd.read_parquet(path_var, engine="pyarrow").to_dask_array()
            else:
                df = in_memory_df.to_dask_array(lengths=True)

            # Two methods to compute phi
            if projection_method:
                phi = da.matmul(u, eigvecs)
            else:
                df2 = self.data_decimate(df, X2=True)
                path_s = Path.cwd() / path_pod / f"{var}" / "s.pkl"
                s = utils.loadit(path_s)
                phi = da.matmul(
                    da.matmul(da.matmul(df2, v.transpose()), da.diag(s.values**-1)),
                    eigvecs,
                )

            utils.save_to_parquet(phi, f"{path_dmd}/{var}/modes", columns=columns)

            # df.compute_chunk_sizes()
            init = df[:, 0].compute()
            # b = np.linalg.pinv(phi) @ init
            b = np.linalg.lstsq(phi.compute(), init, rcond=None)[0]
            utils.saveit(b, f"{path_dmd}/{var}/b.pkl")
            utils.saveit(Lambda, f"{path_dmd}/{var}/lambda.pkl")

    def save_prediction(
        self,
        variables,
        path_dmd=".dmd",
        end=None,
        frame_skip=1,
        mrDMD_sublevel=None,
    ):
        """
        save_prediction predicts future states using DMD and saves the results.

        Args:
            variables (list or str): List of variables to process or a single variable as a string.
            path_dmd (str, optional): Path to the folder containing the DMD results. Defaults to ".dmd".
            end (int, optional): The number of time steps to predict. If None, predicts up to the length of the data. Defaults to None.
            frame_skip (int, optional): The interval of time steps to skip between predictions. Defaults to 1.
            mrDMD_sublevel (int, optional): The current level of multi-resolution DMD (mrDMD) analysis.
                             If None, standard DMD is performed. Defaults to None.
        """
        variables = variables if type(variables) is list else [variables]
        print(self.rhos)

        for var in tqdm(variables, "predicting variables"):
            path_dmd_values = Path.cwd() / path_dmd / f"{var}"
            b = utils.loadit(path_dmd_values / "b.pkl")
            eigs = utils.loadit(path_dmd_values / "lambda.pkl")
            omega = np.log(eigs) / self.dt
            phi = utils.load_from_parquet(f"{path_dmd}/{var}/modes")
            phi = phi.to_dask_array(lengths=True)

            if end is None:
                end = phi.shape[1]

            time = np.arange(0, end, frame_skip) * self.dt
            if mrDMD_sublevel is not None:
                if mrDMD_sublevel == 0:
                    self.period = self.dt * phi.shape[1]
                    if self.freq_mrDMD is None:
                        self.cycle_count = 1
                    else:
                        self.cycle_count = self.freq_mrDMD * self.period
                rho_min = np.min(abs(omega)) / 2 / np.pi
                rho_NQ = 1 / (self.period / 2**mrDMD_sublevel)
                rho = self.cycle_count * rho_NQ
                rho = max(rho_min, rho)
                print(f"rho_min: {rho_min}, rho_NQ: {rho_NQ}, rho_st: {rho}")
                print(f"rho selected as: {rho}")
                slow_mask = np.where(np.abs(omega) / 2 / np.pi <= rho)[0]
                with open(f"{path_dmd_values}/mrdmd.txt", "w") as f:
                    log = f"rho_min: {rho_min}, rho_NQ: {rho_NQ}, rho_st: {rho}\n"
                    log += f"rho selected as: {rho}\n"
                    log += f"Total modes: {len(omega)}\n"
                    log += f"Slow modes: {len(slow_mask)}\n"
                    log += f"Discarded slow modes: {len(slow_mask)/len(omega)}"
                    f.write(log)
                print(f"Total modes: {len(omega)}")
                print(f"Slow modes: {len(slow_mask)}")
                b = b[slow_mask]
                omega = omega[slow_mask]
                phi = phi[:, slow_mask]

            dynamics = np.zeros((len(b), len(time)), dtype=phi.dtype)
            for i, t in enumerate(time):
                dynamics[:, i] = b * np.exp(omega * t)

            prediction_da = da.matmul(phi, dynamics)

            if var.startswith("stack"):
                varnum = int(var.split("_")[1])
                idx_end = prediction_da.shape[0] // varnum
                prediction_da = prediction_da[:idx_end, :]

            utils.save_to_parquet(
                prediction_da,
                path=f"{path_dmd}/{var}/prediction",
                column_dtype=str,
            )

    def describe_parquet(
        self,
        variables,
        path_data=".data",
        folder_name="",
        path_data2=None,
        folder_name2=None,
    ):
        """
        describe_parquet provides statistical descriptions of Parquet datasets.

        Args:
            variables (list or str): List of variables to describe or a single variable as a string.
            path_data (str, optional): Path to the folder containing the Parquet datasets. Defaults to ".data".
            folder_name (str, optional): Subfolder name within the dataset folder. Defaults to "".
            path_data2 (str, optional): Path to a second dataset for comparison. Defaults to None.
            folder_name2 (str, optional): Subfolder name within the second dataset folder. Defaults to None.

        Returns:
            None: Saves the statistical description of the dataset(s) to a text file.
        """
        variables = variables if type(variables) is list else [variables]
        for var in tqdm(variables, "describing variables"):
            data = dd.read_parquet(f"{path_data}/{var}/{folder_name}", engine="pyarrow")

            if folder_name2 is not None:
                if path_data2 is None:
                    path_data2 = path_data
                data2 = dd.read_parquet(
                    f"{path_data2}/{var}/{folder_name2}", engine="pyarrow"
                )
                data = data.map_partitions(lambda a, b: a - b, data2)
            data = data.compute()

            def remove_outliers(df):
                df = df.transpose()
                mask = pd.Series(data=True, index=df.index)
                for index in df.index:
                    Q1 = df.loc[index].quantile(0.25)
                    Q3 = df.loc[index].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    mask = (df.loc[index] < lower_bound) | (df.loc[index] > upper_bound)
                    df.loc[index, mask] = np.nan
                df = df.transpose()
                return df

            data = remove_outliers(data)
            print(data.mean(axis=0).describe())
            if folder_name2 is None:
                with open(f"{path_data}/{var}/{folder_name}/description.txt", "w") as f:
                    f.write(data.mean(axis=0).describe().to_string())
            else:
                with open(
                    f"{path_data}/{var}/{folder_name}/description_diff.txt", "w"
                ) as f:
                    f.write(data.mean(axis=0).describe().to_string())

    def viz_error(
        self,
        variables,
        path_data=".data1",
        path_dmd=".data2",
        path_viz=".viz",
        subfolder="prediction",
    ):
        """
        viz_error visualizes the prediction error between the original data and the predicted data.

        Args:
            variables (list or str): List of variables to analyze or a single variable as a string.
            path_data (str, optional): Path to the folder containing the original data. Defaults to ".data1".
            path_dmd (str, optional): Path to the folder containing the predicted data. Defaults to ".data2".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            subfolder (str, optional): Subfolder name within the predicted data folder. Defaults to "prediction".
        """
        variables = variables if type(variables) is list else [variables]
        for var in tqdm(variables, "prediction error calculation"):

            utils.ensure_dir(f"{path_viz}/{var}")
            path_pq1 = Path.cwd() / path_data / f"{var}"
            path_pq2 = Path.cwd() / path_dmd / f"{var}" / f"{subfolder}"
            df1 = dd.read_parquet(path_pq1, engine="pyarrow")
            df2 = dd.read_parquet(path_pq2, engine="pyarrow")

            # get minimum columns of the two dataframes
            min_cols = min(df1.shape[1], df2.shape[1])
            df1 = df1.iloc[:, :min_cols]
            df2 = df2.iloc[:, :min_cols]

            df1 = df1.to_dask_array()
            df2 = df2.to_dask_array()
            df1.compute_chunk_sizes()
            df2.compute_chunk_sizes()

            # get minimum rows of the two dataframes
            min_rows = min(df1.shape[0], df2.shape[0])
            df1 = df1[:min_rows, :]
            df2 = df2[:min_rows, :]

            # Other Error Metris:

            # Root Mean Square Error (RMSE)
            # rmse = da.mean((df1 - df2) ** 2, axis=0).compute()**0.5

            # # Mean Squared Logarithmic Error (MSLE)
            # msle = da.mean((da.log(df1 + 3e5) - da.log(df2 + 3e5)) ** 2, axis=0).compute()

            # # Mean Absolute Percentage Error (MAPE)
            # mape = da.mean(da.abs((df1 - df2) / df1), axis=0).compute() * 100

            # Mean Absolute Error (MAE)
            mae = da.mean(da.abs(df1 - df2), axis=0).compute()
            error = mae
            utils.saveit(error, f"{path_dmd}/{var}/mae.pkl")

            import matplotlib.pyplot as plt
            import matplotlib.ticker as mtick

            plt.rc("font", family=self.font)
            plt.rc("font", size=self.fontsize)
            plt.switch_backend("agg")
            fig, ax = plt.subplots(1)
            fig.set_size_inches(self.width, self.height)
            fig.patch.set_facecolor("w")
            ax.set_xlabel("Snapshot")
            ax.set_ylabel("MAE")
            ax.set_yscale("log")
            # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            ax.set_axisbelow(True)
            ax.grid(alpha=0.5, which="both")
            ax.grid(alpha=0.3, which="minor", linewidth=self.linewidth * 0.75)
            ax.yaxis.set_minor_locator(mtick.LogLocator(numticks=999, subs="auto"))

            ax.set_xlim(1, error.shape[0])
            if self.data_cutoff is not None:
                ax.axvline(
                    self.data_cutoff * error.shape[0] / 100,
                    color="grey",
                    linestyle="--",
                    linewidth=1,
                )
            ax.plot(
                error[1:],
                self.color,
                linewidth=self.linewidth,
            )
            fig.tight_layout()
            for axis in ["bottom", "left"]:
                ax.spines[axis].set_linewidth(self.ax_width)
            for axis in ["top", "right"]:
                ax.spines[axis].set_linewidth(0)
            plt.savefig(
                f"{path_viz}/{var}/mae" + ".png", dpi=self.dpi, bbox_inches="tight"
            )
            plt.close("all")

    def viz_error_combined(
        self, variables, path_dmd=".data2", path_viz=".viz", path_mrdmd=""
    ):
        """
        viz_error_combined visualizes the combined prediction errors for multiple variables.

        Args:
            variables (list or str): List of variables to analyze or a single variable as a string.
            path_dmd (str, optional): Path to the folder containing the predicted data. Defaults to ".data2".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            path_mrdmd (str, optional): Path to the folder containing multi-resolution DMD (mrDMD) results. Defaults to "".
        """
        variables = variables if type(variables) is list else [variables]
        utils.ensure_dir(path_viz)

        error_combined = pd.DataFrame(columns=variables)
        path_error = Path.cwd() / path_dmd / f"{variables[0]}" / "mae.pkl"
        _data = utils.loadit(path_error)
        cutoff_index = self.data_cutoff * _data.shape[0] // 100
        for var in variables:
            path_error = Path.cwd() / path_dmd / f"{var}" / "mae.pkl"
            error_combined[var] = utils.loadit(path_error)

            if path_mrdmd:
                path_mrdmd = Path.cwd() / path_mrdmd
                error_path = path_mrdmd / f"level_0/level_prediction/{var}/mae.pkl"
                error_data = utils.loadit(error_path)
                if len(error_data) > len(error_combined.index):
                    error_data = error_data[: len(error_combined.index)]
                elif len(error_data) < len(error_combined.index):
                    error_combined = error_combined.iloc[: len(error_data)]
                error_combined[f"{var} mrDMD adjusted"] = error_data

        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick

        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)

        # adjust colors and symbols based on your variables
        plt.switch_backend("agg")
        clrs_list = ["purple", "k", "orange", "r", "g", "b"]
        styl_list = ["solid"] * len(clrs_list)
        # styl_list[2] = "dashed"
        # styl_list[-1] = "dashed"
        # styl_list = ["-", "--", "-.", ":"]
        # clrs_list = ["k", "r", "b"]

        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.width, self.height)
        fig.patch.set_facecolor("w")
        ax.set_xlabel("Snapshot")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_yscale("log")
        ax.set_axisbelow(True)

        ax.grid(alpha=0.5, which="both")
        ax.grid(alpha=0.3, which="minor", linewidth=self.linewidth * 0.75)
        ax.yaxis.set_minor_locator(mtick.LogLocator(numticks=999, subs="auto"))

        ax.set_xlim(0, error_combined.shape[0])
        ymax = error_combined.max().max()
        ymin = error_combined.min().min()
        ymax = ymax * 4
        ymin = ymin * 0.25
        ax.set_ylim(ymin, ymax)
        ax.axvline(
            self.data_cutoff * error_combined.shape[0] // 100 - 1,
            color="grey",
            linestyle="--",
            linewidth=1,
        )

        for i, var in enumerate(error_combined.columns):
            label = re.sub(r"\[(.*?)\]", "", var)  # remove units between brackets
            label = re.sub(r"\.", " ", label)  # remove dots
            if label.startswith("stack"):
                first = f"{label.split('_')[2]}".strip()
                rest = [x.strip() for x in label.split("_")[3:]]
                label = f"{first} w/ {', '.join(rest)}"
                label = label.replace("Velocity u, Velocity v, Velocity w", "Velocity")
            clrr = clrs_list[i % len(clrs_list)]
            styl = styl_list[i % len(styl_list)]
            ax.plot(
                error_combined[var],
                linewidth=self.linewidth,
                label=label,
                color=clrr,
                ls=styl,
            )

        ax.legend(fontsize="xx-small")

        fig.tight_layout()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(self.ax_width)
        for axis in ["top", "right"]:
            ax.spines[axis].set_linewidth(0)

        plt.savefig(
            f"{path_viz}/mae_combined_full" + ".png", dpi=self.dpi, bbox_inches="tight"
        )

        # plot the post-cutoff data
        ax.set_xlim(cutoff_index, error_combined.shape[0])
        ymax = error_combined.iloc[cutoff_index:, :].max().max()
        ymin = error_combined.iloc[cutoff_index:, :].min().min()
        ymax = ymax * 4
        ymin = ymin * 0.25
        ax.set_ylim(ymin, ymax)
        plt.savefig(
            f"{path_viz}/mae_combined_post" + ".png", dpi=self.dpi, bbox_inches="tight"
        )

        # plot the immediate post-cutoff data
        ax.set_xlim(cutoff_index, cutoff_index + 100)
        ymax = error_combined.iloc[cutoff_index : cutoff_index + 100, :].max().max()
        ymin = error_combined.iloc[cutoff_index : cutoff_index + 100, :].min().min()
        ymax = ymax * 4
        ymin = ymin * 0.25
        ax.set_ylim(ymin, ymax)
        plt.savefig(
            f"{path_viz}/mae_combined_immediate" + ".png",
            dpi=self.dpi,
            bbox_inches="tight",
        )

        plt.close("all")

    def viz_parquet(
        self,
        variables,
        num_frames=None,
        path_data=".dmd",
        folder_name="",
        path_data2=None,
        folder_name2=None,
        path_viz=".viz",
        bounds="auto",
        coordinates="2D",
        dist=None,
        dist_map=None,
        cbar=True,
        vmax="auto",
        vmin="auto",
    ):
        """
        viz_parquet visualizes snapshots of Parquet datasets.

        Args:
            variables (list or str): List of variables to visualize or a single variable as a string.
            num_frames (int, optional): Number of frames to visualize. If None, all snapshots are visualized. Defaults to None.
            path_data (str, optional): Path to the folder containing the Parquet datasets. Defaults to ".dmd".
            folder_name (str, optional): Subfolder name within the dataset folder. Defaults to "".
            path_data2 (str, optional): Path to a second dataset for comparison. Defaults to None.
            folder_name2 (str, optional): Subfolder name within the second dataset folder. Defaults to None.
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            bounds (str or list, optional): Domain bounds for visualization. If "auto", bounds are calculated automatically. Defaults to "auto".
            coordinates (str, optional): Specifies the dimensionality of the data ("2D" or "3D"). Defaults to "2D".
            dist (float or None, optional): Distance threshold to mask the visualization using the k-d tree method. If None, no masking is applied. Defaults to None.
            dist_map (ndarray or None, optional): k-d tree distance map for coordinates. If None, it is calculated automatically. Defaults to None.
            cbar (bool, optional): Whether to include a color bar in the visualization. Defaults to True.
            vmax (float or str, optional): Maximum value for the color scale. If "auto", it is determined automatically. Defaults to "auto".
            vmin (float or str, optional): Minimum value for the color scale. If "auto", it is determined automatically. Defaults to "auto".
        """
        variables = variables if type(variables) is list else [variables]

        variables = self.overwrite_guard(path_viz, variables)
        for var in tqdm(variables, "predicting variables"):
            utils.ensure_dir(f"{path_viz}/{var}")

            path = Path.cwd() / path_data / f"{var}"
            data = dd.read_parquet(f"{path}/{folder_name}", engine="pyarrow")

            self.make_dim(coordinates)

            if self.dim == "xy":
                path_x = f"{path_data}/x.pkl"
                path_y = f"{path_data}/y.pkl"
                x = utils.loadit(path_x)
                y = utils.loadit(path_y)

                if bounds == "auto":
                    bounds = self.make_bounds([x, y])

                if dist:
                    dist_map = self.dist_map(x, y, bounds)

            # if a second path is provided, visualize the difference
            if folder_name2 is not None:
                if path_data2 is None:
                    path_data2 = path_data
                path2 = Path.cwd() / path_data2 / f"{var}"
                data2 = dd.read_parquet(f"{path2}/{folder_name2}", engine="pyarrow")
                data = data.map_partitions(lambda a, b: a - b, data2)

            # visualize data Snapshots
            if num_frames is None:
                modelist = range(0, data.shape[1])
            else:
                modelist = np.linspace(0, data.shape[1] - 1, num_frames).astype(int)
            vmax_val = data.iloc[:, 0].max().compute() if vmax == "auto" else vmax
            vmin_val = data.iloc[:, 0].min().compute() if vmin == "auto" else vmin
            match = re.search(r"\[(.*?)\]", var)
            if match:
                cbar_label = match.group(1)
            else:
                cbar_label = " "
            self.u_viz(
                x,
                y,
                data.compute(),
                f"{path_viz}/{var}",
                modelist,
                bounds,
                dist,
                dist_map,
                vmax=vmax_val,
                vmin=vmin_val,
                cbar=cbar,
                cbar_label=cbar_label,
            )
            # self.animate(
            #     Path.cwd() / f"{path_viz}/{var}",
            # )

    def animate(self, path_frames):
        """
        animate creates an animation from a sequence of image frames.

        Args:
            path_frames (Path): Path to the directory containing the image frames.

        Returns:
            None: Saves the animation as an MP4 file in the same directory.
        """
        import imageio

        file_pattern = re.compile(r".*?(\d+).*?")

        def get_order(file):
            match = file_pattern.match(Path(file).name)
            if not match:
                return np.inf
            return int(match.groups()[0])

        with imageio.get_writer(
            path_frames / "animation.mp4", quality=9, fps=24
        ) as writer:
            for png_file in sorted(path_frames.glob("*.png"), key=get_order):
                image = imageio.imread(png_file)
                writer.append_data(image)
            writer.close()

    def viz_eigs_circle(
        self, variables, path_dmd=".dmd", path_viz=".viz", maxmode=None
    ):
        """
        viz_eigs_circle visualizes the eigenvalues of the DMD modes on the complex unit circle.

        Args:
            variables (list or str): List of variables to analyze or a single variable as a string.
            path_dmd (str, optional): Path to the folder containing the DMD results. Defaults to ".dmd".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            maxmode (int, optional): Maximum number of modes to visualize. If None, all modes are visualized. Defaults to None.
        """
        variables = variables if type(variables) is list else [variables]
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        from matplotlib.colors import Normalize

        plt.switch_backend("agg")
        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)
        plt.rc("text", usetex=True)
        for var in tqdm(variables, "plotting DMD modes eigenvalues"):
            utils.ensure_dir(f"{path_viz}/{var}")
            eigs = utils.loadit(f"{path_dmd}/{var}/lambda.pkl")
            if maxmode is None:
                maxmode = eigs.shape[0]
            eigs = eigs[:maxmode]

            fig, ax = plt.subplots(1)
            fig.set_size_inches(self.width, self.height)
            fig.patch.set_facecolor("w")
            ax.set_xlabel("$\\Re(\\lambda_i$)")
            ax.set_ylabel("$\\Im(\\lambda_i$)")
            ax.set_axisbelow(True)
            ax.grid(alpha=0.5, which="both")

            ax.set_ylim(-1.05, 1.05)
            ax.set_xlim(-1.05, 1.05)
            ax.set_aspect(1)
            circ = plt.Circle((0, 0), radius=1, edgecolor="black", facecolor="None")
            ax.add_patch(circ)
            ax.scatter(
                np.real(eigs),
                np.imag(eigs),
                s=7,
                # c=np.abs(eigs),
                c="black",
                edgecolors="none",
            )

            fig.tight_layout()
            for axis in ["bottom", "left"]:
                ax.spines[axis].set_linewidth(self.ax_width)
            for axis in ["top", "right"]:
                ax.spines[axis].set_linewidth(0)
            plt.show()
            plt.savefig(
                f"{path_viz}/{var}/eig_z" + ".png",
                dpi=self.dpi,
                bbox_inches="tight",
            )
            plt.close("all")

    def viz_eigs_spectrum(
        self,
        variables,
        path_dmd=".dmd",
        path_pod=".pod",
        path_viz=".viz",
        maxmode=None,
        freq_max=3000,
    ):
        """
        viz_eigs_spectrum visualizes the spectrum of eigenvalues for DMD modes.

        Args:
            variables (list or str): List of variables to analyze or a single variable as a string.
            path_dmd (str, optional): Path to the folder containing the DMD results. Defaults to ".dmd".
            path_pod (str, optional): Path to the folder containing the POD results. Defaults to ".pod".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            maxmode (int, optional): Maximum number of modes to visualize. If None, all modes are visualized. Defaults to None.
            freq_max (int, optional): Maximum frequency to consider in the spectrum. Defaults to 3000.
        """
        variables = variables if type(variables) is list else [variables]
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        from matplotlib.colors import Normalize
        import matplotlib.mlab as mlab
        from scipy.signal import find_peaks

        plt.switch_backend("agg")
        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)
        plt.rc("text", usetex=True)

        for var in tqdm(variables, "plotting DMD modes eigenvalues PSD"):
            utils.ensure_dir(f"{path_viz}/{var}")
            eigs = utils.loadit(f"{path_dmd}/{var}/lambda.pkl")
            b = utils.loadit(f"{path_dmd}/{var}/b.pkl")

            phi_real = dd.read_parquet(f"{path_dmd}/{var}/modes_real")
            phi_imag = dd.read_parquet(f"{path_dmd}/{var}/modes_imag")

            # Combine the real and imaginary parts to form complex numbers
            phi = phi_real + 1j * phi_imag

            freqs_base = np.imag(np.log(eigs) / self.dt / 2 / np.pi)  # frequency in Hz
            idx = np.argsort(freqs_base)
            Pxx_b = np.abs(b)  # based on first snapshot spectrum

            phi_array = phi.to_dask_array()
            p = da.dot(phi_array.T, phi_array)
            Pxx_phi = da.diag(p).compute().real  # based on phi spectrum

            for name, Pxx_base in zip(["b", "Phi"], [Pxx_b, Pxx_phi]):
                freqs = freqs_base[idx]
                Pxx = Pxx_base[idx]
                Pxx = Pxx[(0 < freqs) & (freqs < freq_max)]
                freqs = freqs[(0 < freqs) & (freqs < freq_max)]
                prominence = np.max(Pxx) / 4
                peaks, _ = find_peaks(
                    Pxx, prominence=prominence, distance=freq_max / 110
                )

                fig, ax = plt.subplots(1)
                fig.set_size_inches(self.width, self.height)
                fig.patch.set_facecolor("w")
                ax.set_xlabel("Frequency [Hz]")
                if name == "b":
                    ax.set_ylabel(f"DMD Spectrum $|$b$|$")
                else:
                    ax.set_ylabel(f"DMD Spectrum $\\Phi^\\dagger\\Phi$")
                ax.grid(alpha=0.5)
                ax.set_xlim(0, freq_max)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

                ax.plot(freqs, Pxx, self.color, linewidth=self.linewidth)
                npeaks = 5
                for n in range(0, min(npeaks, len(peaks))):
                    ax.scatter(
                        freqs[peaks[n]],
                        Pxx[peaks[n]],
                        s=80,
                        facecolors="none",
                        edgecolors="grey",
                    )
                    acc = 0
                    if freq_max < 10:
                        acc = 2
                    ax.annotate(
                        f"{freqs[peaks[n]]:0.{acc}f}",
                        xy=(
                            freqs[peaks[n]] + freq_max / 25,
                            Pxx[peaks[n]] * 0.99,
                        ),
                    )
                fig.tight_layout()
                for axis in ["bottom", "left"]:
                    ax.spines[axis].set_linewidth(self.ax_width)
                for axis in ["top", "right"]:
                    ax.spines[axis].set_linewidth(0)
                plt.savefig(
                    f"{path_viz}/{var}/eig_PS_{name}" + ".png",
                    dpi=self.dpi,
                    bbox_inches="tight",
                )
                plt.close("all")

    def viz_eigs_spectrum_combined(
        self,
        variables,
        path_dmd=".dmd",
        path_pod=".pod",
        path_viz=".viz",
        maxmode=None,
        freq_max=3000,
    ):
        """
        viz_eigs_spectrum_combined visualizes the combined spectrum of eigenvalues for multiple variables.

        Args:
            variables (list or str): List of variables to analyze or a single variable as a string.
            path_dmd (str, optional): Path to the folder containing the DMD results. Defaults to ".dmd".
            path_pod (str, optional): Path to the folder containing the POD results. Defaults to ".pod".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            maxmode (int, optional): Maximum number of modes to visualize. If None, all modes are visualized. Defaults to None.
            freq_max (int, optional): Maximum frequency to consider in the spectrum. Defaults to 3000.

        Returns:
            None: Saves the combined spectrum visualization as a plot.
        """
        variables = variables if type(variables) is list else [variables]
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mtick
        from matplotlib.colors import Normalize
        import matplotlib.mlab as mlab
        from scipy.signal import find_peaks
        from scipy.signal import convolve

        plt.switch_backend("agg")
        plt.rc("font", family=self.font)
        plt.rc("font", size=self.fontsize)
        plt.rc("text", usetex=True)

        def gaussian_kernel(x, sigma):
            return np.exp(-0.5 * (x / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

        eigs_combined = 0
        df_combined = pd.DataFrame()

        for var in tqdm(variables, "plotting DMD modes eigenvalues PSD"):
            utils.ensure_dir(f"{path_viz}/{var}")
            _eig = utils.loadit(f"{path_dmd}/{var}/lambda.pkl")
            _b = utils.loadit(f"{path_dmd}/{var}/b.pkl")
            eigs_combined += _eig
            df_combined[var] = np.abs(_b) / np.abs(_b).max()

        # normalize df_combined
        df_combined = df_combined / df_combined.max()

        df_combined["freqs"] = np.imag(
            np.log(eigs_combined) / self.dt / 2 / np.pi
        )  # frequency in Hz
        df_combined = df_combined.sort_values(by="freqs")
        df_combined = df_combined[
            (0 < df_combined["freqs"]) & (df_combined["freqs"] < freq_max)
        ]

        freqs = df_combined["freqs"].values
        df_combined.drop(columns=["freqs"], inplace=True)
        Pxx = df_combined.sum(axis=1).values

        fig, ax = plt.subplots(1)
        fig.set_size_inches(self.width, self.height)
        fig.patch.set_facecolor("w")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("DMD Combined Spectrum $\\Sigma|b/b_0|$")
        ax.grid(alpha=0.5)
        ax.set_xlim(0, freq_max)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        # ax.set_yscale("log")

        prominence = np.max(Pxx) / 2
        peaks, _ = find_peaks(Pxx, prominence=prominence, distance=freq_max / 100)
        ax.plot(freqs, Pxx, self.color, linewidth=self.linewidth)

        npeaks = 5
        for n in range(0, min(npeaks, len(peaks))):
            ax.scatter(
                freqs[peaks[n]],
                Pxx[peaks[n]],
                s=80,
                facecolors="none",
                edgecolors="grey",
            )
            acc = 0
            if freq_max < 10:
                acc = 2
            ax.annotate(
                f"{freqs[peaks[n]]:0.{acc}f}",
                xy=(
                    freqs[peaks[n]] + freq_max / 25,
                    Pxx[peaks[n]] * 0.99,
                ),
            )
        fig.tight_layout()
        for axis in ["bottom", "left"]:
            ax.spines[axis].set_linewidth(self.ax_width)
        for axis in ["top", "right"]:
            ax.spines[axis].set_linewidth(0)
        plt.savefig(
            f"{path_viz}/eig_PS_combined" + ".png",
            dpi=self.dpi,
            bbox_inches="tight",
        )
        plt.close("all")

    def viz_modes(
        self,
        variables,
        modelist=None,
        freqlist=None,
        coordinates="2D",
        path_dmd=".dmd",
        path_viz=".viz",
        bounds="auto",
        dist=False,
    ):
        """
        viz_modes visualizes the DMD modes for the specified variables.

        Args:
            variables (list or str): List of variables to visualize or a single variable as a string.
            modelist (list, optional): List of mode indices to visualize. Defaults to None.
            freqlist (list, optional): List of frequencies to visualize. If provided, the closest modes to these frequencies will be selected. Defaults to None.
            coordinates (str, optional): Specifies the dimensionality of the data ("2D" or "3D"). Defaults to "2D".
            path_dmd (str, optional): Path to the folder containing the DMD results. Defaults to ".dmd".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            bounds (str or list, optional): Domain bounds for visualization. If "auto", bounds are calculated automatically. Defaults to "auto".
            dist (bool, optional): Whether to apply a distance-based mask for visualization. Defaults to False.
        """
        variables = variables if type(variables) is list else [variables]

        if modelist is not None:
            modelist = modelist if type(modelist) is list else list(modelist)
        if freqlist is not None:
            freqlist = freqlist if type(freqlist) is list else list(freqlist)
            if modelist is not None:
                print("please either provide modelist or freqlist")
                return

        self.make_dim(coordinates)

        for var in tqdm(variables, "analyzing variables"):
            utils.ensure_dir(f"{path_viz}/{var}")

            eigs = utils.loadit(f"{path_dmd}/{var}/lambda.pkl")
            freqs = np.imag(np.log(eigs) / self.dt / 2 / np.pi)

            if freqlist is not None:
                modelist = []
                for f in freqlist:
                    diffs = abs(freqs - f)
                    sorted_indices = np.argsort(diffs)
                    closest_indices = sorted_indices[:3]
                    for idx in closest_indices:
                        modelist.append(idx)

            freq_lookup = {idx: freqs[idx] for idx in modelist}

            if self.dim == "xy":
                path_x = f"{path_dmd}/x.pkl"
                path_y = f"{path_dmd}/y.pkl"
                x = utils.loadit(path_x)
                y = utils.loadit(path_y)

                if bounds == "auto":
                    bounds = self.make_bounds([x, y])

                if dist:
                    dist_map = self.dist_map(x, y, bounds)

            from scipy.interpolate import griddata
            import matplotlib.pyplot as plt

            plt.switch_backend("agg")
            plt.rc("font", family=self.font)
            plt.rc("font", size=self.fontsize)

            xmin, xmax, ymin, ymax, res = bounds
            xx, yy = self.make_meshgrid(bounds)

            for mode in tqdm(modelist, "plotting 2D mode shapes", leave=False):
                dmd_mode = dd.read_parquet(
                    f"{path_dmd}/{var}/modes_real", engine="pyarrow"
                )
                dm = dmd_mode.iloc[:, mode].compute()
                kk = griddata(
                    (x, y),
                    dm,
                    (xx, yy),
                    method="linear",
                    fill_value=min(abs(dm)),
                )
                if dist:
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

                contour = ax.contourf(
                    xx,
                    yy,
                    kk,
                    self.contour_levels,
                    cmap=self.cmap,
                    antialiased=True,
                    extend="both",
                )
                for c in contour.collections:
                    c.set_edgecolor("face")
                fig.tight_layout()
                for axis in ["top", "bottom", "left", "right"]:
                    ax.spines[axis].set_linewidth(self.ax_width)
                plt.savefig(
                    f"{path_viz}/{var}/freq_{freq_lookup[mode]}_mode_{mode}" + ".png",
                    dpi=self.dpi,
                    bbox_inches="tight",
                )
                plt.close("all")

    def viz_modes_3d(
        self,
        variables,
        bounds="auto",
        modelist=None,
        freqlist=None,
        coordinates="3D",
        path_dmd=".dmd",
        path_viz=".viz",
    ):
        """
        viz_modes_3d visualizes the DMD modes in 3D for the specified variables.

        Args:
            variables (list or str): List of variables to visualize or a single variable as a string.
            bounds (str or list, optional): Domain bounds for visualization. If "auto", bounds are calculated automatically. Defaults to "auto".
            modelist (list, optional): List of mode indices to visualize. Defaults to None.
            freqlist (list, optional): List of frequencies to visualize. If provided, the closest modes to these frequencies will be selected. Defaults to None.
            coordinates (str, optional): Specifies the dimensionality of the data ("3D"). Defaults to "3D".
            path_dmd (str, optional): Path to the folder containing the DMD results. Defaults to ".dmd".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
        """
        variables = variables if type(variables) is list else [variables]
        self.make_dim(coordinates)

        for var in tqdm(variables, "analyzing variables"):
            utils.ensure_dir(f"{path_viz}/{var}")

            eigs = utils.loadit(f"{path_dmd}/{var}/lambda.pkl")
            freqs = np.imag(np.log(eigs) / self.dt / 2 / np.pi)

            if freqlist is not None:
                modelist = []
                for f in freqlist:
                    diffs = abs(freqs - f)
                    sorted_indices = np.argsort(diffs)
                    closest_indices = sorted_indices[:1]
                    for idx in closest_indices:
                        modelist.append(idx)

            path_x = f"{path_dmd}/x.pkl"
            path_y = f"{path_dmd}/y.pkl"
            path_z = f"{path_dmd}/z.pkl"
            x = utils.loadit(path_x)
            y = utils.loadit(path_y)
            z = utils.loadit(path_z)
            if bounds == "auto":
                bounds = self.make_bounds([x, y, z])

            # [xmin, xmax, ymin, ymax, zmin, zmax, res]
            bounds[3] = 0.3

            u = dd.read_parquet(f"{path_dmd}/{var}/modes_real", engine="pyarrow")

            self.u_viz_3d(
                x,
                y,
                z,
                u,
                path_viz,
                modelist,
                bounds,
            )

    def multires(
        self, variables, path_parquet=".data", path_mrdmd=".mrdmd", levels=4, end=None
    ):
        """
        multires performs multi-resolution Dynamic Mode Decomposition (mrDMD) analysis.

        Args:
            variables (list or str): List of variables to analyze or a single variable as a string.
            path_parquet (str, optional): Path to the folder containing the Parquet datasets. Defaults to ".data".
            path_mrdmd (str, optional): Path to the folder where the mrDMD results will be saved. Defaults to ".mrdmd".
            levels (int, optional): Number of levels for the mrDMD analysis. Defaults to 4.
            end (int, optional): The number of time steps to analyze. If None, analyzes the entire dataset. Defaults to None.
        """
        variables = variables if type(variables) is list else [variables]
        path_mrdmd = Path.cwd() / path_mrdmd

        for var in tqdm(variables, "computing mrDMD modes and coefficients"):
            for level in range(levels):
                for i in range(2**level):
                    utils.ensure_dir(path_mrdmd / f"level_{level}/{i}")
                utils.ensure_dir(path_mrdmd / f"level_{level}/level_prediction")

            for level in range(0, levels):  # dmd levels
                if level > 0:
                    df = utils.load_from_parquet(
                        path_mrdmd / f"level_{level-1}/level_prediction/{var}"
                    )
                else:
                    df = utils.load_from_parquet(f"{path_parquet}/{var}")
                df = df.persist()
                shape_new = df.shape[1]
                if shape_new % 2**level != 0:
                    df = df.iloc[:, : shape_new - shape_new % 2**level]
                slice_size = df.shape[1] // 2**level
                dfs = [
                    df.iloc[:, i * slice_size : (i + 1) * slice_size]
                    for i in range(2**level)
                ]

                for i in range(0, 2**level):  # sublevels
                    print(f"level {level}, sublevels {i+1}/{2**level}")
                    df_i = dfs[i]
                    path_pod = path_mrdmd / f"level_{level}/{i}"
                    path_dmd = path_mrdmd / f"level_{level}/{i}"
                    self.svd_save_usv(
                        var,
                        path_parquet=path_parquet,
                        path_pod=path_pod,
                        dmd_X1=True,
                        in_memory_df=df_i,
                    )
                    self.save_Atilde(
                        var,
                        path_parquet=path_parquet,
                        path_pod=path_pod,
                        path_dmd=path_dmd,
                        in_memory_df=df_i,
                    )
                    self.save_modes(
                        var,
                        path_parquet=path_parquet,
                        path_dmd=path_dmd,
                        path_pod=path_pod,
                        projection_method=True,
                        in_memory_df=df_i,
                    )
                    self.save_prediction(
                        var,
                        path_dmd=path_dmd,
                        end=slice_size,
                        mrDMD_sublevel=level,
                    )

                # save level prediction
                dfs = []
                for i in range(2**level):
                    df_part = utils.load_from_parquet(
                        path_mrdmd / f"level_{level}/{i}/{var}/prediction"
                    )
                    dfs.append(df_part)
                df_new = dd.concat(dfs, axis=1)
                nparts = df.npartitions
                df_new = df_new.repartition(npartitions=nparts)
                df_new.columns = df.columns
                utils.save_to_parquet(
                    df_new, path_mrdmd / f"level_{level}/slow_modes/{var}"
                )
                df -= df_new
                utils.save_to_parquet(
                    df, path_mrdmd / f"level_{level}/level_prediction/{var}"
                )
                del df, df_new, dfs

    def multires_predict(self, variables, path_mrdmd=".mrdmd", end=None):
        """
        multires_predict combines predictions from all levels of multi-resolution DMD (mrDMD).

        Args:
            variables (list or str): List of variables to process or a single variable as a string.
            path_mrdmd (str, optional): Path to the folder containing the mrDMD results. Defaults to ".mrdmd".
            end (int, optional): The number of time steps to predict. If None, predicts up to the length of the data. Defaults to None.
        """
        variables = variables if type(variables) is list else [variables]
        for var in variables:
            path_mrdmd = Path.cwd() / path_mrdmd

            levels = 0
            while True:
                if not (path_mrdmd / f"level_{levels}").exists():
                    break
                levels += 1
            print(f"Number of levels: {levels}")

            for level in tqdm(range(0, levels), "mrDMD level calculation"):
                temp_df = utils.load_from_parquet(
                    path_mrdmd / f"level_{level}/level_prediction/{var}",
                    imaginary_part=False,
                )
                if level == 0:
                    prediction_df = temp_df
                else:
                    prediction_df += temp_df
                utils.save_to_parquet(
                    prediction_df,
                    path_mrdmd / f"level_{level}/dmd_prediction" / f"{var}",
                )

            for level in range(levels):
                shutil.copy(
                    path_mrdmd / f"level_0/0/x.pkl",
                    path_mrdmd / f"level_{level}/level_prediction",
                )
                shutil.copy(
                    path_mrdmd / f"level_0/0/y.pkl",
                    path_mrdmd / f"level_{level}/level_prediction",
                )

                shutil.copy(
                    path_mrdmd / f"level_0/0/x.pkl",
                    path_mrdmd / f"level_{level}/dmd_prediction",
                )
                shutil.copy(
                    path_mrdmd / f"level_0/0/y.pkl",
                    path_mrdmd / f"level_{level}/dmd_prediction",
                )

                shutil.copy(
                    path_mrdmd / f"level_0/0/x.pkl",
                    path_mrdmd / f"level_{level}/slow_modes",
                )
                shutil.copy(
                    path_mrdmd / f"level_0/0/y.pkl",
                    path_mrdmd / f"level_{level}/slow_modes",
                )

    def viz_multires(
        self,
        variables,
        num_frames=0,
        path_mrdmd=".mrdmd",
        path_viz=".viz",
        bounds="auto",
        coordinates="2D",
        dist=None,
        cbar=True,
        vmax="auto",
        vmin="auto",
    ):
        """
        viz_multires visualizes the results of multi-resolution DMD (mrDMD) analysis.

        Args:
            variables (list or str): List of variables to visualize or a single variable as a string.
            num_frames (int, optional): Number of frames to visualize. If 0, all snapshots are visualized. Defaults to 0.
            path_mrdmd (str, optional): Path to the folder containing the mrDMD results. Defaults to ".mrdmd".
            path_viz (str, optional): Path to save the visualization results. Defaults to ".viz".
            bounds (str or list, optional): Domain bounds for visualization. If "auto", bounds are calculated automatically. Defaults to "auto".
            coordinates (str, optional): Specifies the dimensionality of the data ("2D" or "3D"). Defaults to "2D".
            dist (float or None, optional): Distance threshold to mask the visualization using the k-d tree method. If None, no masking is applied. Defaults to None.
            cbar (bool, optional): Whether to include a color bar in the visualization. Defaults to True.
            vmax (float or str, optional): Maximum value for the color scale. If "auto", it is determined automatically. Defaults to "auto".
            vmin (float or str, optional): Minimum value for the color scale. If "auto", it is determined automatically. Defaults to "auto".
        """
        variables = variables if type(variables) is list else [variables]
        path_mrdmd = Path.cwd() / path_mrdmd
        path_viz = Path.cwd() / path_viz

        levels = 0
        while True:
            if not (path_mrdmd / f"level_{levels}").exists():
                break
            levels += 1
        print(f"Number of levels: {levels}")

        for level in range(0, levels):
            self.viz_parquet(
                variables,
                num_frames=num_frames,
                path_data=path_mrdmd / f"level_{level}/level_prediction",
                folder_name="",
                folder_name2=None,
                path_viz=path_viz / f"level_{level}",
                bounds=bounds,
                coordinates=coordinates,
                dist=dist,
                cbar=cbar,
                vmax=vmax,
                vmin=vmin,
            )
            self.viz_parquet(
                variables,
                num_frames=num_frames,
                path_data=path_mrdmd / f"level_{level}/slow_modes",
                folder_name="",
                folder_name2=None,
                path_viz=path_viz / f"slow_{level}",
                bounds=bounds,
                coordinates=coordinates,
                dist=dist,
                cbar=cbar,
                vmax=vmax,
                vmin=vmin,
            )
            self.viz_parquet(
                variables,
                num_frames=num_frames,
                path_data=path_mrdmd / f"level_{level}/dmd_prediction",
                folder_name="",
                folder_name2=None,
                path_viz=path_viz / f"mrdmd_level_{level}",
                bounds=bounds,
                coordinates=coordinates,
                dist=dist,
                cbar=cbar,
                vmax=vmax,
                vmin=vmin,
            )
