import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import matplotlib.cm as cm
from matplotlib.cm import ScalarMappable
import math

def bitarr_to_int(bitarr):
    val = 0
    for bit in bitarr:
        val <<= 1
        val |= bit
    return val

def create_hypergraph(bin_rep_len=5):
    n = bin_rep_len
    shape=(2**math.floor(n/2), 2**math.ceil(n/2))
    t = [[[] for _ in range(shape[1])] for _ in range(shape[0])]

    counter = 0
    while counter != n:
        times_before_switch = 2 ** math.floor(counter/2)
        if counter % 2 == 0: 
            switch = 0
            for col in range(shape[1]):
                for row in range(shape[0]):
                    t[row][col].insert(0, switch)
                if (col+1) % times_before_switch == 0:
                    switch = int(not switch)
        else:
            switch = 0
            for row in range(shape[0]):
                for col in range(shape[1]):
                    t[row][col].insert(0, switch)
                if (row+1) % times_before_switch == 0:
                    switch = int(not switch)
        counter += 1

    new_format = [["" for _ in range(shape[1])] for _ in range(shape[0])]

    for i, row in enumerate(t):
        for j, val in enumerate(row):
            new_format[i][j] = "".join(str(x) for x in val)

    return new_format


def plot_fitness_grid(fitness_grid, optima, show_optima=False, output_file=None, incomplete=False, max_min="min"):
    finite_matrix = np.copy(fitness_grid)
    finite_matrix[finite_matrix == math.inf] = np.nanmax(finite_matrix[np.isfinite(finite_matrix)]) * 1.1

    zero_mask = (finite_matrix == optima)

    incomplete_mask = (finite_matrix == -1)

    if max_min == "min":
        norm = Normalize(vmin=optima, vmax=np.nanmax(finite_matrix))
    else:
        norm = Normalize(vmin=0.0, vmax=optima)

    red_cmap = LinearSegmentedColormap.from_list('redscale', [(1,1,1), (1,0,0)])


    rgb_image = red_cmap(norm(finite_matrix))[:, :, :3]

    if show_optima:
        rgb_image[zero_mask] = [0, 1, 0]

    if incomplete:
        rgb_image[incomplete_mask] = [0.3, 0.3, 0.3]


    _, ax = plt.subplots(figsize=(20, 8))
    ax.imshow(rgb_image)
    
    sm = ScalarMappable(cmap=red_cmap, norm=norm)
    sm.set_array([])  
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_yticklabels([])  
    cbar.set_label("Fitness Value (Redder -> Higher)", fontsize=12)

    rows, cols = finite_matrix.shape

    for i in range(1, rows):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(1, cols):
        ax.axvline(j - 0.5, color='black', linewidth=0.5)

    max_dim = max(rows, cols)

    p = 0
    while (2 ** p) < max_dim:
        val = 2 ** p
        curr_val = val
        linewidth = 0.5 + p  
        while curr_val < max_dim:
            if curr_val < rows:
                ax.axhline(curr_val - 0.5, color='black', linewidth=linewidth)
            if curr_val < cols:
                ax.axvline(curr_val - 0.5, color='black', linewidth=linewidth)
            curr_val += val
        p += 1

    ax.set_xticks(np.arange(-.5, finite_matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, finite_matrix.shape[0], 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    

    plt.title("Hypergraph of Fitness Landscape for Feature Selection")
    if output_file is None:
        plt.show()
    else:
        plt.savefig(f"./images/{output_file}.png")


def produce_hypergraph_plot(lookup_table, genotype_length, optima, show_optima=True, output=None, incomplete=False, max_min="min"):
    hypergraph_layout = create_hypergraph(genotype_length)
    for row in hypergraph_layout:
        for j, val in enumerate(row):
            try:
                row[j] = lookup_table[val]
            except:
                if incomplete:
                    row[j] = -1
    
    plot_fitness_grid(hypergraph_layout, optima, show_optima, output, incomplete, max_min)
        



