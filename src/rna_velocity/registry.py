"""Map method → conda env and native velocity layer key.

Subset = scenario recommendations from Luo et al. 2026 Fig. 6D
(extra-large / low-quality / complex dynamics). CellRank stays in
velo-core for shared downstream only — not as a velocity inference method.
"""

METHOD_ENV = {
    "scvelo_stochastic": "velo-core",
    "dynamo": "velo-core",
    "velovi": "velo-core",
    "deepvelo": "deepvelo",
    "unitvelo": "unitvelo",
    "latentvelo": "latentvelo",
    "pyrovelocity": "pyrovelocity",
}

# Native layer/key written by the method before aliasing to layers['velocity']
METHOD_VKEY = {
    "scvelo_stochastic": "velocity",
    "dynamo": "velocity_S",
    "velovi": "velocity",
    "deepvelo": "velocity",
    "unitvelo": "velocity",
    "latentvelo": "spliced_velocity",
    "pyrovelocity": "velocity_pyro",
}
