import cupy as cp
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

GRID_SIZES = (5, 10, 25, 50, 100, 125, 250, 500)
SEIDEL_ITERATIONS = 1000

def create_random_grid(size: int):
    grid = cp.random.rand(size, size).astype(cp.float64)
    # Boundaries are set to zero.
    grid[[0, -1]] = 0
    grid[:, [0, -1]] = 0
    return grid

def gauss_seidel(grid):
    # Shift the grid values to perform the Gauss-Seidel iteration in a vectorized manner
    up = cp.roll(grid, shift=1, axis=0)
    down = cp.roll(grid, shift=-1, axis=0)
    left = cp.roll(grid, shift=1, axis=1)
    right = cp.roll(grid, shift=-1, axis=1)
    # Compute the new grid values
    newGrid = 0.25 * (up + down + left + right)
    # Set the boundary values to zero
    newGrid[[0, -1]] = 0
    newGrid[:, [0, -1]] = 0
    return newGrid

def run_GS_solver(grid, size: int):
    startTime = timer()
    for i in range(SEIDEL_ITERATIONS):
        grid = gauss_seidel(grid)

    # Save the resulting matrix as an hdf5 file
    with h5py.File('newGrid_cupy.hdf5', 'w') as f:
        # create a new dataset in the hdf5 file
        dset = f.create_dataset("newGrid", data=grid)

    # Move the grid back to the CPU
    grid = cp.asnumpy(grid)
    return timer() - startTime

if __name__ == "__main__":
    timeSpents = []
    for gridSize in tqdm(GRID_SIZES):
        grid = create_random_grid(gridSize)
        timeSpents.append(run_GS_solver(grid, gridSize))

    plt.plot(GRID_SIZES, timeSpents, label="CuPy GPU")
    plt.title("The performance of the Gauss-Seidel solver")
    plt.xlabel("Grid Sizes")
    plt.ylabel("Time spent (s)")
    plt.legend()
    plt.show()
