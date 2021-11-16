import os
from collections import defaultdict, namedtuple

PLOTS_FOLDER = "plots"
ENV_FOLDER = "envs"
MODELS_FOLDER = "models"

BASE_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
PLOTS_PATH = os.path.join(BASE_PATH, PLOTS_FOLDER)
ENV_PATH = os.path.join(BASE_PATH, ENV_FOLDER)
MODELS_PATH = os.path.join(BASE_PATH, MODELS_FOLDER)

TELEGRAM_TOKEN = ""
TELEGRAM_MESSAGE_ID = 0

# source: https://gist.github.com/thriveth/8560036
color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#e41a1c",
    "#dede00",
    "#999999",
    "#f781bf",
    "#a65628",
    "#984ea3",
]

hatch_cycle = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]

marker_cycle = ["*", "^", "o", "x", "d", "+", "v", "."]


# pygame keys
keys_nt = namedtuple(
    "keys_nt",
    [
        "RIGHT",
        "LEFT",
        "UP",
        "DOWN",
        "ONE",
        "TWO",
        "THREE",
        "FOUR",
        "FIVE",
        "SIX",
        "SEVEN",
        "EIGTH",
        "NINE",
        "ZERO",
    ],
)
KEYS = keys_nt(
    275,
    276,
    273,
    274,
    ord("1"),
    ord("2"),
    ord("3"),
    ord("4"),
    ord("5"),
    ord("6"),
    ord("7"),
    ord("8"),
    ord("9"),
    ord("0"),
)

# LEGACY
KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_DOWN = 275, 276, 273, 274


# Acquisition functions

LABELS = {
    "rand": "random",
    "rand_unobs": "random_unobserved",
    "var": "variance",
    "pi": "probability_of_improvement",
    "ei": "expected_improvement",
    "epd": "expected_policy_divergence",
    "idrl": "directed_information_gain",
    "evr": "expected_volume_removal",
    "evrb": "expected_volume_removal_bernoulli",
    "maxreg": "maximum_regret",
}

color_cycle = [
    "#377eb8",
    "#ff7f00",
    "#4daf4a",
    "#e41a1c",
    "#dede00",
    "#999999",
    "#f781bf",
    "#a65628",
    "#984ea3",
]

## dig is deprecated but was still in some results

AF_COLORS = {
    "rand": "#4daf4a",
    "var": "#984ea3",
    "ei": "#ff7f00",
    "epd": "#377eb8",
    "idrl": "#e41a1c",
    "maxreg": "#b3b300",
}
AF_COLORS = defaultdict(lambda: "blue", AF_COLORS)

AF_MARKERS = {
    "rand": "s",
    "var": "^",
    "ei": "d",
    "epd": "x",
    "idrl": "o",
    "maxreg": "v",
}
AF_MARKERS = defaultdict(lambda: ".", AF_MARKERS)

AF_ALPHA = {
    "idrl": 1.0,
}
AF_ALPHA = defaultdict(lambda: 0.6, AF_ALPHA)

AF_ZORDER = {
    "idrl": 2,
}
AF_ZORDER = defaultdict(lambda: 1, AF_ZORDER)
