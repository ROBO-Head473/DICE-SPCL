"""Neighborhood Agreement-Guided Multiview Clustering (NAG-MVC)."""

from .model import NAGMVC, nag_mvc_loss
from .features import prepare_features, compute_subject_discordance, assemble_representation
