import numpy as np
from itertools import product, combinations
import random


def split_range_with_overlap_percentage(
    range_min, range_max, num_parts, overlap_percentage
):
    """
    Splits a range into equal parts with overlap specified as a percentage of each part's length.

    Args:
    - range_min (int): The minimum value of the range.
    - range_max (int): The maximum value of the range.
    - num_parts (int): The number of parts to divide the range into.
    - overlap_percentage (float): The percentage of overlap between adjacent parts.

    Returns:
    - List of tuples, each representing the start and end of a part.
    """
    total_length = range_max - range_min
    part_length_without_overlap = total_length / num_parts
    overlap_length = part_length_without_overlap * overlap_percentage / 100
    print(total_length, part_length_without_overlap, overlap_length)

    split_ranges = []
    for i in range(num_parts):
        if i != 0 and i != (num_parts - 1):
            split_ranges.append(
                (
                    range_min + part_length_without_overlap * i - overlap_length,
                    range_min + part_length_without_overlap * (i + 1) + overlap_length,
                )
            )
        elif i == 0:
            split_ranges.append(
                (
                    range_min,
                    range_min + part_length_without_overlap * (i + 1) + overlap_length,
                )
            )
        else:
            split_ranges.append(
                (
                    range_min + part_length_without_overlap * i - overlap_length,
                    range_max,
                )
            )

    return split_ranges


def get_neighbors_2D(x, y, width, height, n):
    """Get up to n neighbors for a point located at (x, y) in an image of size width x height."""
    neighbors = []
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:
                continue  # Skip the point itself
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                neighbors.append((nx, ny))
                count += 1

            if count > n:
                break

    if len(neighbors) == 0:
        return None

    return neighbors  # Return up to n neighbors

def get_neighbors_3D(x, y, z, width, height, depth, n):
    """Get up to n neighbors for a point located at (x, y) in an image of size width x height."""
    neighbors = []
    count = 0
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                if dx == 0 and dy == 0 and dz == 0:
                    continue  # Skip the point itself
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
                    neighbors.append((nx, ny, nz))
                    count += 1

                if count > n:
                    break

    if len(neighbors) == 0:
        return None

    return neighbors  # Return up to n neighbors

def sample_combinations(all_neighbors, budget):
    """
    Sample combinations of neighboring points under a computational budget.

    :param all_neighbors: A list of lists, where each inner list contains the n neighboring points of one of the k points.
    :param budget: An integer specifying the maximum number of combinations to consider.
    :return: A list of sampled combinations, each a possible set of points.
    """
    # Calculate the total possible combinations
    total_combinations = np.prod([len(neighbors) for neighbors in all_neighbors])
    # print(total_combinations)

    # If the total combinations are within the budget, compute them all
    if total_combinations <= budget and total_combinations > 0:
        return list(product(*all_neighbors))

    # If exceeding the budget, sample combinations
    sampled_combinations = []
    sampled_indices = set()  # To avoid duplicate combinations due to random sampling

    while len(sampled_combinations) < budget:
        # Randomly select one neighbor from each point's neighborhood
        # print(all_neighbors)
        combination_indices = tuple(
            random.sample(range(len(neighbors)), 1)[0] for neighbors in all_neighbors
        )

        # Check if this combination has already been selected
        if combination_indices not in sampled_indices:
            sampled_indices.add(combination_indices)
            sampled_combinations.append(
                tuple(
                    all_neighbors[i][index]
                    for i, index in enumerate(combination_indices)
                )
            )

    return sampled_combinations


def sample_neighboring_points_2D(k_points, n, image_width, image_height, budget=2000):
    """Sample n neighboring points for each of the k points and return all combinations."""
    all_neighbors = []
    for point in k_points:
        x, y = point
        neighbors = get_neighbors_2D(x, y, image_width, image_height, n)
        if neighbors is not None:
            all_neighbors.append(neighbors)

    # Generate all possible combinations of one neighbor from each point's neighborhood
    # print(all_neighbors)
    all_combinations = sample_combinations(all_neighbors, budget)
    return all_combinations


def sample_neighboring_points_3D(k_points, n, image_width, image_height, image_depth, budget=2000):
    """Sample n neighboring points for each of the k points and return all combinations."""
    all_neighbors = []
    for point in k_points:
        x, y, z = point
        neighbors = get_neighbors_3D(x, y, z, image_width, image_height, image_depth, n)
        if neighbors is not None:
            all_neighbors.append(neighbors)

    # Generate all possible combinations of one neighbor from each point's neighborhood
    # print(all_neighbors)
    all_combinations = sample_combinations(all_neighbors, budget)
    return all_combinations
