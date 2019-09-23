import datetime
import itertools
import random

import gensim
import numpy as np

import sampl.paradigm_smith as ps
import sampl.paradigm_coman as coman


def search_smith(
    n_sims=30,
    n_jobs=4,
    monte_carlo=None,
    start_i=0,
    end_i=None,
    memmap_path=None,
    overwrite=False,
):
    y_max_grid = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0])
    y_min_grid = -1 * y_max_grid
    dip_center_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    dip_width_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    discount_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]

    today = datetime.datetime.today().strftime("%Y-%m-%d")

    parameter_grid = list(
        itertools.product(
            y_min_grid, y_max_grid, dip_center_grid, dip_width_grid, discount_grid
        )
    )
    np.save(file=f"params_smith_{today}.npy", arr=parameter_grid)

    if monte_carlo is not None:
        random.shuffle(parameter_grid)
        parameter_grid = parameter_grid[0:monte_carlo]

    costs = ps.parameter_search_mem(
        parameter_grid,
        n_sims=n_sims,
        n_jobs=n_jobs,
        start_i=start_i,
        end_i=end_i,
        memmap_path=memmap_path,
        overwrite=overwrite,
    )
    np.save(file=f"costs_smith_{today}.npy", arr=costs)


def search_coman(
    n_sims=20,
    n_jobs=4,
    monte_carlo=None,
    start_i=0,
    end_i=None,
    memmap_path=None,
    overwrite=False,
):
    y_max_grid = np.array([1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    y_min_grid = -1 * y_max_grid
    dip_center_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    dip_width_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
    discount_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    learning_rate_grid = [0.2, 0.4, 0.6, 0.8, 1.0]

    parameter_grid = list(
        itertools.product(
            y_min_grid,
            y_max_grid,
            dip_center_grid,
            dip_width_grid,
            discount_grid,
            learning_rate_grid,
        )
    )

    if monte_carlo is not None:
        random.shuffle(parameter_grid)
        parameter_grid = parameter_grid[0:monte_carlo]

    costs = coman.parameter_search(
        parameter_grid,
        n_runs=n_sims,
        n_jobs=n_jobs,
        start_i=start_i,
        end_i=end_i,
        memmap_path=memmap_path,
        overwrite=overwrite,
    )

    today = datetime.datetime.today().strftime("%Y-%m-%d")
    np.save(file=f"params_coman_{today}.npy", arr=parameter_grid)
    np.save(file=f"costs_coman_{today}.npy", arr=costs)


if __name__ == "__main__":
    print("Starting Smith...")
    search_smith(n_jobs=24)
    print("Finished Smith.")

    print("Starting Coman...")
    search_coman(n_jobs=24)
    print("Finished Coman.")
