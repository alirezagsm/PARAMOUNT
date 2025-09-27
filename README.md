# PARAMOUNT: Parallel Modal Analysis of Large Datasets

PARAMOUNT is a lightweight Python toolkit for computing _Proper Orthogonal Decomposition (POD)_ and _Dynamic Mode Decomposition (DMD)_ on large numerical and experimental datasets. It leverages parallel processing to analyze large amounts of data efficiently.

## Overview

- **Distributed Processing:** Ideal for multi-core parallel processing of large data.
- **Methodology:** A brief video introduction to the theory is available [here](https://www.youtube.com/watch?v=uz0q_TKrC84).

### Proper Orthogonal Decomposition (POD)

- Perform distributed computation of POD to extract dominant spatial patterns.
- Accompanying research paper:

<p align="center"><img src="demo_pod.png" alt="POD demo" style="width:60%; max-width:720px; height:auto;"></p>

> Alireza Ghasemi, et al. "Combustion Dynamics Analysis of a Pressurized Airblast Swirl Burner using Proper Orthogonal Decomposition." _International Journal of Spray and Combustion Dynamics_, 2023. [DOI:10.1177/17568277231207252](https://journals.sagepub.com/doi/10.1177/17568277231207252)

### Dynamic Mode Decomposition (DMD)

- Compute DMD modes, eigenvalues, and generate future state predictions.
- Multi-resolution DMD (MRDMD) to analyze data across various temporal scales.
- Accompanying research paper:

<p align="center"><img src="demo_dmd.png" alt="DMD demo" style="width:60%; max-width:720px; height:auto;"></p>

> Alireza Ghasemi, Jim B.W. Kok. "Exploring Liquid Fuel Combustion Dynamics in a Swirl Burner using Dynamic Mode Decomposition." _Engineering Applications of Computational Fluid Mechanics_, 2025. [DOI:10.1080/19942060.2025.2557009](https://www.tandfonline.com/doi/full/10.1080/19942060.2025.2557009)

### Visualization

- Easily visualize POD/DMD modes and coefficients using Matplotlib.
- 3D data can be visualized interactively with Plotly.

## Using PARAMOUNT

1. **Installation**

   ```bash
   pip install -r requirements.txt
   ```
2. **Data Preparation**

   - PARAMOUNT works well with CSV datasets and can convert them to Parquet for better performance.
   - The domain coordinates `x/y/z.pkl` are inferred based on input CSV data.
   - Specify variables of interest and convert to Parquet. See `csv_example` for details.

   ```python
   # Convert your data into parquet format
   pod = POD()
   variables = POD.get_folderlist(...)
   pod.csv_to_parquet(...)
   ```
3. **Analysis (POD / DMD / MRDMD)**

   - Vertex distance `dist` may be set to mask out external regions within the domain bounding box.
   - Other parameters such as timestep `dt` and maximum analysis frequency `freq_max` can be specified.
   - For POD: use the `POD` class to compute SVD from Parquet datasets. Results (U, S, V) are stored. See `svd_example`.

   ```python
   pod.svd_save_usv(...)
   ```

   - For DMD: use the `DMD` class to compute dynamic modes and eigenvalues. See `dmd_example`.

   ```python
   dmd = DMD()
   dmd.save_Atilde(...)
   dmd.save_modes(...)
   dmd.save_prediction(...)
   ```

   - For MRDMD: use `MRDMD` utilities. See `mrdmd_example`.

   ```python
   dmd.multires(...)
   dmd.multires_predict(...)
   ```
4. **Visualization**

   - Visualization parameters can be customized.

   ```python
   pod.set_time(dt)
   pod.set_viz_params(dpi=600, linewidth=0.85, color="black", cmap="seismic")
   ```

   - 2D visualization methods for each analysis are included in the provided sample scripts.
   - Interactive 3D visualization of results can be performed similar to the provided `3D_viz_example.ipynb`.

## Proposed Project Folder Structure

This is a sample project structure for using the PARAMOUNT library to perform POD and DMD analysis.

```
Project
├── myproject.py
├── .data
│   ├── variable_1/
│   │   └── *.parquet
│   ├── variable_2/
│   │   └── *.parquet
│   ├── x.pkl
│   └── y.pkl
├── .usv
│   ├── variable_1/
│   │   ├── s.pkl
│   │   ├── u/*.parquet
│   │   └── v/*.parquet
│   └── variable_2/
│       ├── s.pkl
│       ├── u/*.parquet
│       └── v/*.parquet
├── .dmd
│   ├── variable_1/
│   │   ├── Atilde.pkl
│   │   ├── b.pkl
│   │   ├── lambda.pkl
│   │   ├── modes_imag/*.parquet
│   │   ├── modes_real/*.parquet
│   │   └── prediction/*.parquet
│   └── variable_2/ ...
├── .mrdmd
│   └── variable_1/levels/...
├── .viz
│   └── variable_1/results.png
└── src
    ├── PARAMOUNT_BASE.py
    ├── PARAMOUNT_POD.py
    ├── PARAMOUNT_DMD.py
    └── utils.py
```

## Notes and Acknowledgements

If you use PARAMOUNT in published work, please cite the relevant paper. \
This toolkit is developed by [Alireza Ghasemi](https://www.linkedin.com/in/alirezaaghasemi/) at University of Twente under the [MAGISTER](https://www.magister-itn.eu/) project.
