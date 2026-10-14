# NAG-MVC main-model code

**NAG-MVC** stands for **Neighborhood Agreement-Guided Multiview Clustering**.

Main-model code for **NAG-MVC: Multiview Clustering of Patients With Major Depressive Disorder Using Resting-State fMRI**.

## Quick start

Use Python 3.10. Run from the repository root:

```bash
python -m pip install -r requirements.txt
python run_model.py --demo --device cpu --out-dir ../nag_mvc_demo_output
```

## Your data

Provide a NumPy `.npz` file containing the following aligned arrays:

| Array | Shape | Contents |
| --- | --- | --- |
| `local_metrics` | N × 4 × 116 | Finite AAL116 regional feature values |
| `view_names` | 4 | Strings: `ALFF`, `fALFF`, `ReHo`, `DC`, in this order |
| `groups` | N | Integer or string codes; `1` selects MDD participants |
| `sites` | N | Integer or string site labels |

Use non-object array types. At least four MDD participants are required. Image preprocessing and regional feature extraction precede this workflow. Obtain data through the [REST-meta-MDD provider](https://rfmri.org/REST-meta-MDD) and follow its data-use conditions.

```bash
python run_model.py --dataset /path/to/data.npz --device cpu --out-dir /path/to/new_results
```

Use `--device cuda` for GPU execution. Keep data and outputs outside the repository.
