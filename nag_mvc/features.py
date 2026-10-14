"""Site adjustment, neighborhood agreement, and construction of clustering features."""
from __future__ import annotations
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler

def residualize_site_means(local_metrics: np.ndarray, sites: np.ndarray) -> np.ndarray:
    """Subtract each site's feature mean and add the pooled MDD cohort mean.

    This preprocessing step uses the selected clustering cohort and precedes
    feature standardization. It adjusts site-specific means; site differences
    in variance and covariance are outside its scope.
    """

    values = np.asarray(local_metrics, dtype=np.float32)
    adjusted = values.copy()
    pooled_mean = values.mean(axis=0, keepdims=True)
    for site in np.unique(np.asarray(sites).astype(str)):
        mask = np.asarray(sites).astype(str) == site
        adjusted[mask] = values[mask] - values[mask].mean(axis=0, keepdims=True) + pooled_mean
    return adjusted.astype(np.float32)

def standardize_mdd(local: np.ndarray) -> np.ndarray:
    mean = local.mean(axis=0, keepdims=True)
    std = local.std(axis=0, keepdims=True)
    return ((local - mean) / np.maximum(std, 1e-6)).astype(np.float32)

def _pairs(n_views: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(n_views), 2))

def compute_subject_discordance(
    x: np.ndarray, n_neighbors: int, reliability_temperature: float = 0.20
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
    """Compute neighborhood agreement and fusion weights before training.

    agreement[i, p] is the Jaccard similarity between patient i's neighbor
    sets for the view pair pairs[p]. With four views, agreement has shape
    (N, 6), and fusion has shape (N, 4). A softmax of each view's summed
    agreement with the other views gives its fusion weight. Both arrays
    remain fixed during training; discordance is computed as 1 - agreement.
    """

    from sklearn.neighbors import NearestNeighbors

    n_samples, n_views, _ = x.shape
    if n_samples < 4:
        raise ValueError("NAG-MVC needs at least four MDD participants.")
    k = min(max(2, n_neighbors), n_samples - 1)
    neighbor_sets: list[list[set[int]]] = []
    for view in range(n_views):
        indices = NearestNeighbors(n_neighbors=k + 1).fit(x[:, view]).kneighbors(
            x[:, view], return_distance=False
        )
        neighbor_sets.append(
            [
                {int(index) for index in row if int(index) != subject}
                for subject, row in enumerate(indices)
            ]
        )

    pairs = _pairs(n_views)
    agreement = np.zeros((n_samples, len(pairs)), dtype=np.float32)
    for pair_index, (left, right) in enumerate(pairs):
        for subject in range(n_samples):
            union = neighbor_sets[left][subject] | neighbor_sets[right][subject]
            agreement[subject, pair_index] = (
                len(neighbor_sets[left][subject] & neighbor_sets[right][subject]) / max(len(union), 1)
            )

    reliability = np.zeros((n_samples, n_views), dtype=np.float32)
    for pair_index, (left, right) in enumerate(pairs):
        reliability[:, left] += agreement[:, pair_index]
        reliability[:, right] += agreement[:, pair_index]
    if reliability_temperature <= 0.0:
        raise ValueError("reliability_temperature must be positive.")
    logits = reliability / reliability_temperature
    logits -= logits.max(axis=1, keepdims=True)
    fusion = np.exp(logits)
    fusion /= fusion.sum(axis=1, keepdims=True)
    return agreement, fusion.astype(np.float32), pairs

def _discordance_conditioned_diffusion(
    joint: np.ndarray,
    consensus_anchor: np.ndarray,
    discordance: np.ndarray,
    n_neighbors: int,
    strength: float,
    mutual_neighbors: bool = False,
    reliability_weights: np.ndarray | None = None,
    reliability_temperature: float = 0.10,
) -> np.ndarray:
    """Average each joint representation once with its neighbors' representations.

    In the manuscript workflow, consensus_anchor contains the 464 standardized
    regional measurements used to find neighbors. discordance contains the six
    unstandardized values 1 - agreement. Each patient's averaging coefficient
    is strength times their mean discordance divided by the cohort's 95th
    percentile (floored at 1e-6), then clipped to [0, 1]. All coordinates are
    updated using neighbor means calculated from the original joint array.

    The manuscript uses directed neighbors and their arithmetic mean, with
    mutual_neighbors=False and reliability_weights=None.
    """

    if strength <= 0.0:
        return joint
    from sklearn.neighbors import NearestNeighbors

    n_samples = len(joint)
    k = min(max(2, n_neighbors), n_samples - 1)
    indices = NearestNeighbors(n_neighbors=k + 1).fit(consensus_anchor).kneighbors(
        consensus_anchor, return_distance=False
    )
    directed = indices[:, 1:]
    if mutual_neighbors:
        # Optional reciprocal-neighbor selection; inactive in the manuscript.
        # Keep neighbors that also include the current patient. If none qualify,
        # retain that patient's original directed neighbor set.
        membership = [set(row.tolist()) for row in directed]
        selected = []
        for subject, row in enumerate(directed):
            reciprocal = [int(neighbour) for neighbour in row if subject in membership[int(neighbour)]]
            selected.append(np.asarray(reciprocal if reciprocal else row, dtype=np.int64))
        directed = selected

    if reliability_weights is not None:
        if reliability_temperature <= 0.0:
            raise ValueError("diffusion reliability temperature must be positive.")
        neighbour_mean_rows = []
        for subject, neighbours in enumerate(directed):
            similarity = reliability_weights[neighbours] @ reliability_weights[subject]
            logits = similarity / reliability_temperature
            logits -= logits.max()
            weights = np.exp(logits)
            weights /= weights.sum()
            neighbour_mean_rows.append((joint[neighbours] * weights[:, None]).sum(axis=0))
        neighbour_mean = np.stack(neighbour_mean_rows, axis=0)
    else:
        neighbour_mean = joint[directed].mean(axis=1)
    discordance_score = discordance.mean(axis=1)
    scale = max(float(np.quantile(discordance_score, 0.95)), 1e-6)
    patient_strength = np.clip(strength * discordance_score / scale, 0.0, 1.0).astype(np.float32)
    return ((1.0 - patient_strength[:, None]) * joint + patient_strength[:, None] * neighbour_mean).astype(
        np.float32
    )


def prepare_features(local, sites):
    """Apply site mean adjustment and standardization to aligned MDD features.

    Input shape is (N, 4, 116), in ALFF, fALFF, ReHo, DC order. Standardization
    uses the selected cohort and a standard-deviation floor of 1e-6."""
    local = np.asarray(local, dtype=np.float32)
    sites = np.asarray(sites)
    if local.ndim != 3 or local.shape[1:] != (4, 116):
        raise ValueError("Expected patient features with shape (N, 4, 116).")
    if len(local) < 4 or sites.shape != (len(local),) or not np.isfinite(local).all():
        raise ValueError("At least four patients, aligned site labels, and finite features are required.")
    return standardize_mdd(residualize_site_means(local, sites))


def assemble_representation(fused, x, agreement, n_neighbors=50, strength=0.20):
    """Concatenate fused features, discordance, and regional measurements.

    Standardize fused features and discordance across patients; x already
    contains standardized regional measurements. The manuscript configuration
    gives 64 + 6 + 464 = 534 coordinates, with unit measurement weight.
    Apply one local average before passing the result to K-means."""
    discordance = 1.0 - agreement
    measurements = x.reshape(len(x), -1).astype(np.float32)
    joint = np.concatenate([
        StandardScaler().fit_transform(fused).astype(np.float32),
        StandardScaler().fit_transform(discordance).astype(np.float32),
        measurements,
    ], axis=1).astype(np.float32)
    return _discordance_conditioned_diffusion(
        joint, measurements, discordance, n_neighbors, strength,
    )
