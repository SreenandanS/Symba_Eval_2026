"""Shared graph-contract constants for the symmetry-respecting custom graph encoder."""

from __future__ import annotations

# Node roles
NODE_ROLE_EXTERNAL = 0
NODE_ROLE_VERTEX = 1
NODE_ROLE_PROPAGATOR = 2
NUM_NODE_ROLES = 3

# Kinematic relation types
KIN_REL_LEG = 0
KIN_REL_PROP_S = 1
KIN_REL_PROP_T = 2
KIN_REL_PROP_U = 3
KIN_REL_PROP_UNKNOWN = 4
NUM_KIN_RELATIONS = 5

# Color relation types
COLOR_REL_FUND = 0
COLOR_REL_ANTIFUND = 1
COLOR_REL_ADJ_FUND = 2
COLOR_REL_ADJ_ANTIFUND = 3
NUM_COLOR_RELATIONS = 4

# Spinor relation types
SPINOR_REL_BOSON = 0
SPINOR_REL_FERMION_ALONG = 1
SPINOR_REL_FERMION_AGAINST = 2
NUM_SPINOR_RELATIONS = 3

# Vertex interaction types
VERTEX_INT_NONE = 0
VERTEX_INT_QQG = 1
VERTEX_INT_GGG = 2
NUM_VERTEX_INTERACTIONS = 3
