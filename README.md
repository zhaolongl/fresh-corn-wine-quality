# Fresh corn wine quality

Python code and processed data for examining cultivar, thermal pretreatment and fermentation effects on fresh corn wine quality. The package recalculates descriptive results, renders the revised manuscript figures and checks numerical agreement with the accompanying supplementary workbook.

[中文说明](README_zh-CN.md) · [Data dictionary](docs/data_dictionary.md) · [Figure guide](docs/figures.md) · [Reproduction scope](docs/reproduction.md)

This is a **processed-data and archived-result reproduction package**. The original materials contain model output arrays but do not contain a complete model-training workflow. PCA/PLS-DA/XGBoost/SHAP panels use the supplied results; running this project does not retrain these models. FigS1 is a supplied SIMCA export and is copied unchanged. The other ten figures are drawn by Python, including panels based on digitized coordinates where identified in the data manifest.

## Installation

Use Python 3.11 for the recorded dependency versions. Create an environment from the repository directory:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

Install the dependencies and package:

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
```

No scikit-learn, XGBoost, SIMCA, Word or Excel installation is required to run this release. The supplementary XLSX is read with openpyxl. JSON resources replace the original executable pickle objects.

## Reproduce the release

```bash
python -m fresh_corn_wine build --output outputs
```

The default PNG resolution is 600 dpi. This command writes:

| Directory | Contents |
| --- | --- |
| `outputs/tables/` | Four recalculated CSV tables and a JSON description of their scope |
| `outputs/figures/` | Fig1–Fig7 and FigS1–FigS4, each as PNG and PDF |
| `outputs/checks/` | `validation.json`, with pass/fail status and individual discrepancies |

The static FigS1 files retain their original resolution. Most Python-generated PDF elements are vector graphics; rasterized elements remain rasterized. Font availability can change layout slightly; Arial is preferred and DejaVu Sans is the fallback.

Individual tasks can also be run:

```bash
python -m fresh_corn_wine analyze --output outputs
python -m fresh_corn_wine figures --figures Fig1 Fig6 FigS4 --dpi 300 --output outputs
python -m fresh_corn_wine validate --output outputs
python -m fresh_corn_wine --help
```

After installation, `corn-wine` is an equivalent entry point. Paths are resolved from package resources and the chosen output directory; the original manuscript directory is not needed. Existing generated files with matching names in the output directory are replaced.

## What is recalculated

| Output | Calculation |
| --- | --- |
| `variance_decomposition.csv` | Between-group/total sum-of-squares ratios for nine indices across 20 thermal-treatment means |
| `volatile_class_summary.csv` | Concentration sums, detected compound counts and approximate propagated SDs after the reviewed class corrections |
| `composite_scores.csv` | Weighted sums of the released, reviewed Z-scores using the fixed reported weights |
| `estimated_aroma_indices.csv` | Concentration/retained unified threshold and within-sample relative OAV |

Reference results include a mean method eta-squared of **0.8503012094**, cultivar scores **HN 1.14, BT −0.47, HT −0.66**, and fermentation scores **WMF 0.64, PMF −0.17, CJF −0.47**. ST-60 has the highest thermal-treatment composite score, **0.92**. Scores are comparable within each experiment only.

The bundled [supplementary workbook](fresh_corn_wine/resources/Supplementary%20Tables.xlsx) contains Tables S1–S9. Validation checks the relevant released data against Tables S2, S3, S5–S9, as well as resource hashes and score arithmetic. These checks establish consistency among the distributed artifacts; they do not validate the original experiments or constitute independent model validation.

## Repository layout

```text
fresh_corn_wine/
  __main__.py             Command-line interface
  analysis.py             Descriptive calculations
  dataio.py               Typed JSON decoder and resource access
  validation.py           Data/workbook consistency checks
  plotting/               Figure builders and plotting helpers
  resources/
    bundles/              Five typed JSON bundles
    tables/               36 readable CSV exports
    config/               Reviewed classes and fixed scoring scheme
    assets/               Archived FigS1 PNG/PDF
    Supplementary Tables.xlsx
    manifest.json         Resource descriptions and data levels
    provenance.json       Source inventory, changes and SHA-256 hashes
docs/                     Scientific scope and data/figure documentation
tests/                    Numerical and figure regression checks
.github/workflows/        Windows and Ubuntu reproducibility checks
```

## Development checks

```bash
python -m pip install pytest==7.4.0
python -m pytest
python -m fresh_corn_wine build --dpi 100 --output outputs/smoke
```

The tests check supplementary-table agreement, important numerical results, decoder isolation, zero-based sensory radar axes, multiclass SHAP labels and unchanged copying of FigS1. The GitHub Actions workflow runs these tests and a lower-resolution build on Windows and Ubuntu after upload. Local release verification is recorded in [docs/verification.json](docs/verification.json).

Generated outputs, virtual environments, caches and original pickle files are excluded by `.gitignore`. Resource text uses LF line endings so that its recorded hashes remain stable after Git checkout. Scientific interpretation and the limitations of the supplied source material are documented in [Reproduction scope](docs/reproduction.md).
