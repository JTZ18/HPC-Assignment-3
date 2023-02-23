import torch
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from timeit import default_timer as timer

GRID_SIZES = (5, 10, 25, 50, 100, 125, 250, 500)
SEIDEL_ITERATIONS = 1000

def create_random_grid(size: int):
    grid = torch.rand(size, size, dtype=torch.float64)
    # Boundaries are set to zero.
    grid[[0, -1]] = 0
    grid[:, [0, -1]] = 0
    return grid

@torch.jit.script
def gauss_seidel(grid):
    # Shift the grid values to perform the Gauss-Seidel iteration in a vectorized manner
    up = torch.roll(grid, shifts=1, dims=0)
    down = torch.roll(grid, shifts=-1, dims=0)
    left = torch.roll(grid, shifts=1, dims=1)
    right = torch.roll(grid, shifts=-1, dims=1)
    # Compute the new grid values
    newGrid = 0.25 * (up + down + left + right)
    # Set the boundary values to zero
    newGrid[[0, -1]] = 0
    newGrid[:, [0, -1]] = 0
    return newGrid

def run_GS_solver(grid, size: int):
    # Move the grid to the GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grid = grid.to(device)
    startTime = timer()
    for i in range(SEIDEL_ITERATIONS):
        grid = gauss_seidel(grid)

    # Save the resulting matrix as an hdf5 file
    with h5py.File('newGrid_pytorch.hdf5', 'w') as f:
        # create a new dataset in the hdf5 file
        dset = f.create_dataset("newGrid", data=grid)

    # Move the grid back to the CPU
    grid = grid.cpu()
    return timer() - startTime

if __name__ == "__main__":
    timeSpents = []
    for gridSize in tqdm(GRID_SIZES):
        grid = create_random_grid(gridSize)
        timeSpents.append(run_GS_solver(grid, gridSize))

    plt.plot(GRID_SIZES, timeSpents, label="PyTorch GPU")
    plt.title("The performance of the Gauss-Seidel solver")
    plt.xlabel("Grid Sizes")
    plt.ylabel("Time spent (s)")
    plt.legend()
    plt.show()
