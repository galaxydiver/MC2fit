# MC2fit

**Multi-component, Multi-condition GALFIT Pipeline**

MC2fit is a Python-based workflow for two-dimensional galaxy image decomposition using [GALFIT](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html). It is designed to fit many galaxies under multiple model configurations, initial conditions, and masking choices, then collect and compare the results systematically.

The included Jupyter notebooks demonstrate a complete workflow with **DESI Legacy Imaging Surveys** data: downloading images and point-spread functions (PSFs), preparing masks, running single- and multi-component GALFIT models, inspecting the fits, and classifying preferred models. The core fitting code can also work with user-prepared FITS images, PSFs, noise maps, and masks.

> **Project status:** MC2fit is currently a collection of Python modules and research notebooks, not a packaged `pip install mc2fit` distribution. Some notebooks are more extensively documented than others. Paths and sample-specific settings must be adapted before use.

## Features

- **Survey preparation:** Download Legacy Survey DR9/DR10 image cutouts and PSF products; prepare sigma images and survey masks.
- **Automated masking:** Detect contaminating sources with `sep`, generate masks at different thresholds, protect the central target, and combine masks across bands.
- **Multi-condition fitting:** Run GALFIT with different starting parameters, constraints, Sérsic/PSF components, and masking choices.
- **Batch processing:** Manage per-galaxy directories and run supported operations in parallel.
- **Fit evaluation:** Extract GALFIT results, calculate fit statistics (including reduced chi-square and AIC-based measures), compare models over selected image regions, and produce diagnostic plots.
- **Model classification:** Compare structural model families and identify preferred decompositions from packed fitting results.

## Workflow

```text
Target catalog (RA, Dec, object names)
                |
                v
1. Data preparation       Legacy Survey images, PSFs, sigma maps, quality checks
                |
                v
2. Masking                Source detection, target protection, mask combination
                |
                v
3. GALFIT fitting         Multiple Sérsic/PSF models, starting points, constraints
                |
                v
   Post-processing        Fit statistics, image inspection, packed results
                |
                v
4. Classification         Preferred structural models and fitted parameters
```

Start with the notebooks in numerical order. The fitting and classification notebooks contain sample-specific branches, so they should be treated as worked examples rather than one-click scripts.

## Tutorials

| Step | Notebook | Purpose |
| --- | --- | --- |
| 1 | [Setting and downloading](MC2fit_1_Setting_and_downloading_docs.ipynb) | Set up a target catalog and directories; download Legacy Survey products and select usable data. |
| 2 | [Masking](MC2fit_2_Masking_docs.ipynb) | Prepare and inspect masks, measure PSFs, and combine masks across filters. |
| 3 | [GALFIT runs — PSB example](MC2fit_3_Run_PSB.ipynb) | Fit single- through multi-component models, collect results, and compare fitting conditions. |
| 3 | [GALFIT runs — KCWI-named example](MC2fit_3_Run_KCWI.ipynb) | Additional sample-specific fitting workflow; check its catalog and path settings before running. |
| 4 | [Classification — PSB example](MC2fit_4_Classification_psb.ipynb) | Load packed results and inspect model classifications. |
| 4 | [Classification — guided version](MC2fit_4_Classification_psb_guided.ipynb) | More detailed classification walkthrough, including a notebook-local classification implementation. |

The two `*_docs.ipynb` notebooks cover initial preparation in greater detail. The other notebooks demonstrate the fitting/classification analysis and are being consolidated into fuller documentation.

## Repository modules

| File | Role |
| --- | --- |
| `DH_MC2fit.py` | Main fitting pipeline: settings, model grids, GALFIT execution, extraction, statistics, results, visualization, and classification. |
| `DH_MC2fit_sub.py` | Sérsic-profile and surface-brightness conversion utilities. |
| `DH_masking2.py` | SEP-based detection, segmentation masks, central-region handling, and mask combination. |
| `DH_LegacyCrawling.py` | Legacy Survey downloads, brick lookup, PSF generation, sigma and maskbits preparation. |
| `DH_FileCheck.py` | Directory management, file-quality checks, symbolic links, and parallel file operations. |
| `DH_path.py` | Configuration of the GALFIT executable and Legacy Survey data paths. |
| `DH_array.py` | Supporting array utilities used by the pipeline. |
| `DH_multicore.py` | Supporting parallel-execution utilities used by the pipeline. |

**Important:** `DH_array.py` and `DH_multicore.py` are required at runtime. They exist in the repository but were not part of the six Python modules included in the current documentation review. Keep these supporting modules alongside the other `DH_*.py` files.

## Installation

### 1. Get the source code

```bash
git clone https://github.com/galaxydiver/MC2fit.git
cd MC2fit
```

MC2fit currently uses local Python imports (for example, `import DH_MC2fit`). Run the notebooks from the repository, or make its directory available on `PYTHONPATH` if you keep the notebooks elsewhere.

### 2. Create a Python environment

The following is a suggested environment setup; a fully tested, version-pinned environment has not yet been published.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy scipy astropy pandas matplotlib scikit-image \
    sep opencv-python requests tqdm jupyterlab ipykernel
```

For a headless server, `opencv-python-headless` may be preferable to `opencv-python`. The workflow also uses standard-library modules such as `multiprocessing`, `subprocess`, and `pickle`, which do not need separate installation.

If your notebooks are outside the repository, set the module search path **before starting Jupyter**:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
jupyter lab
```

### 3. Install GALFIT separately

**GALFIT is an external executable, not a Python dependency installed by the command above.** Download a compatible GALFIT 3 executable from the [official GALFIT website](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html), and ensure it can run on your system. The MC2fit code invokes that executable through `subprocess`.

Record the **full path to the executable** (not just its containing directory); you will set it in `DH_path.py`.

### 4. Download Legacy Survey brick-summary files

The Legacy Survey download functions use three brick-summary FITS tables:

- [DR9 South — `survey-bricks-dr9-south.fits.gz`](https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/south/survey-bricks-dr9-south.fits.gz)
- [DR9 North — `survey-bricks-dr9-north.fits.gz`](https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr9/north/survey-bricks-dr9-north.fits.gz)
- [DR10 South — `survey-bricks-dr10-south.fits.gz`](https://portal.nersc.gov/cfs/cosmo/data/legacysurvey/dr10/south/survey-bricks-dr10-south.fits.gz)

Download and decompress them to `.fits` files in a location of your choice. The file names must match the paths configured in the next step. See the [DR9](https://www.legacysurvey.org/dr9/files/) and [DR10](https://www.legacysurvey.org/dr10/files/) data documentation for more information.

**Note:** `DH_LegacyCrawling.py` loads all three summary tables when it is imported. Configure and obtain these files *before* importing `DH_MC2fit` or `DH_LegacyCrawling`, even if you plan to analyze your own prepared images.

### 5. Configure `DH_path.py`

For a new computer, edit the `MANUAL_PATHS` dictionary near the top of `DH_path.py`:

```python
MANUAL_PATHS = {
    "legacy_sum_south": "/absolute/path/to/survey-bricks-dr9-south.fits",
    "legacy_sum_south_dr10": "/absolute/path/to/survey-bricks-dr10-south.fits",
    "legacy_sum_north": "/absolute/path/to/survey-bricks-dr9-north.fits",
    "brick_dir": "/absolute/path/to/Legacy_bricks/",
    "brick_test_dir": "/absolute/path/to/Legacy_bricks_test/",
    "galfit": "/absolute/path/to/galfit",
}
```

Create the brick storage directories as needed. They are used to cache downloaded brick-level products.

`DH_path.py` also recognizes the optional `~/this_machine.dat` file: machine IDs `1`, `2`, and `3` select predefined paths and **override** `MANUAL_PATHS`. If you are not using these machine-specific presets, remove or rename `~/this_machine.dat` (or adjust the configuration logic) so the manual paths are selected.

### 6. Check the installation

After configuring the paths, verify that the main module imports successfully:

```bash
python -c "import DH_MC2fit; print('MC2fit import OK')"
```

An import error at this step commonly indicates a missing local `DH_*.py` module, a missing Python dependency, or an incorrect Legacy Survey summary-file path. Import success verifies basic setup, not the complete GALFIT fitting workflow.

## Getting started

### Prepare a target catalog

The tutorial notebooks use a CSV file named `fulldata.csv` inside their example working directory. Their executable code expects at least these columns (lowercase spelling matters):

```csv
name,ra,dec
example_001,185.7119,13.5928
```

Here `ra` and `dec` are celestial coordinates in degrees. You can replace the example target and file location with your own catalog.

In the tutorial notebooks, the sample working directory is:

```python
dir_main = "./Galfit_Run/PSB_new"
```

Edit `dir_main`, the catalog filename, number of workers, cutout size, and survey settings to suit your project. In particular, the later notebooks assume the earlier notebooks have already created the working directories and data files.

### Run the notebooks in order

1. **Setting and downloading:** Define `DIdat`/`DI`, create target directories, download image and PSF products, and check the north/south data coverage.
2. **Masking:** Link selected survey files into the fitting directories, measure PSFs, run the masking presets, and inspect/merge `g`- and `r`-band masks.
3. **GALFIT runs:** Create `Setting` and `Runlist` objects, fit Sérsic and PSF component combinations, extract results, calculate statistics, and save `ResPack` objects.
4. **Classification:** Load the `ResPack` files and use the comparison statistics to classify each galaxy's preferred structural model.

The main fitting API follows this sequence once `DIgood`, input FITS files, and a suitable `Runlist` have been prepared:

```python
import DH_MC2fit as mc2fit

# Construct a settings object with your filenames and image parameters.
settings = mc2fit.Setting(
    proj_folder="g_1comp/",
    fni_image="image_lg.fits",
    fni_psf="psf_model_lg.fits",
    fni_sigma="sigma_lg.fits",
    fni_masking="masking_c_nb.fits",
    image_size=121,
    plate_scale=0.262,
    est_sky=10,
    Ncore=4,
)

# `runlist` and `DIgood` are created in the tutorial setup/fitting cells.
settings.update_Class(runlist, DIgood)
settings.run()
settings.extract()
settings.postprocessing()
packed_results = settings.pack_result()
```

This is an **API outline**, not a standalone script: first create `runlist` with `mc2fit.Runlist(...)` and `DIgood` with `DH_FileCheck.DirInfo`/the quality-check workflow, and verify the image/PSF/sigma/mask files exist. Refer to the step-3 notebooks for complete multi-model examples.

## Model families and outputs

The classification examples compare six main model families:

| Label | Components |
| --- | --- |
| `S` | Single Sérsic |
| `SS` | Sérsic + Sérsic |
| `SP` | Sérsic + PSF |
| `P` | Single PSF |
| `SSP` | Sérsic + Sérsic + PSF |
| `SPP` | Sérsic + PSF + PSF |

The fitting notebooks also explore additional components and conditions, such as central masking, fixed Sérsic indices, and transferring the best-fitting `g`-band structure to the `r` band. Model selection in the notebooks uses statistics including AIC-based quantities, sometimes calculated within a galaxy-dependent aperture.

A typical example directory structure is:

```text
Galfit_Run/PSB_new/
├── fulldata.csv
├── gal_dat/                 # Downloaded survey products by galaxy
│   └── gal0000/
├── gal_out/                 # Standardized fitting inputs and outputs
│   └── gal0000/
│       ├── image_lg.fits
│       ├── psf_model_lg.fits
│       ├── sigma_lg.fits
│       ├── masking_c_nb.fits
│       └── g_1comp/
│           └── result_*.fits
└── ResPack_g_1Comp.pkl      # Example packed fitting results
```

File names and model folders vary with the notebook, filter, and selected fitting configuration. `extract()` can remove the larger intermediate GALFIT output blocks after extracting the smaller result FITS files; retain originals if you need them for debugging.

## Practical notes

- **Inspect masks and fit residuals:** Automated segmentation and statistical model selection do not replace visual quality checks.
- **Check your units:** Pixel scale, photometric zero point, sky offset, and image units must be appropriate for the data. For the `sersic2` component, the notebook's parameter slot labeled `mag` is used as surface brightness at the effective radius, not total integrated magnitude.
- **Use reproducible paths:** The notebooks contain example-specific relative paths, and `DH_path.py` contains machine-specific presets. Update these before using the code on a new system or publishing a cleaned repository.
- **Do not commit large data products:** Keep downloaded survey bricks, FITS cutouts, fit outputs, and generated pickle files out of version control unless they are intentional example datasets. Clearing stored notebook outputs can also reduce repository size.
- **Current limitations:** Dependencies and Python/GALFIT versions are not pinned; notebooks have not been presented here as automated installation or integration tests. Some notebooks use different classification implementations or example settings.

## References

- [GALFIT home page and downloads](https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html)
- Peng et al. (2002), *AJ*, **124**, 266 — original GALFIT paper.
- Peng et al. (2010), *AJ*, **139**, 2097 — GALFIT 3.
- [DESI Legacy Imaging Surveys documentation](https://www.legacysurvey.org/)

## Maintainer

**Donghyeon J. Khim** — [galaxydiver](https://github.com/galaxydiver)
