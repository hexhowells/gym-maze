import random
from grid import Grid


def generate_maze(width, height, wall_tile=1, path_tile=0):
    if (width % 2) == 0:
        print(f"Width of {width} is not odd, adding 1 to maze width.")
        width += 1

    if (height % 2) == 0:
        print(f"Height of {height} is not odd, adding 1 to maze height.")
        height += 1

    # initialise grid
    maze_grid = [[wall_tile] * width] * height
    maze = Grid(maze_grid)
    
    # random start position
    start_x = random.randrange(1, width, 2)
    start_y = random.randrange(1, height, 2)
    maze.set((start_y, start_x), path_tile)
    
    cells = [(start_y, start_x)]
    
    # generate maze using growing tree algorithm (DFS)
    while cells:
        y, x = cells[-1]

        for new_y, new_x in maze.get_neighbours((y, x), rand=True):
            if maze.get((new_y, new_x)) == wall_tile:
                # get wall tile between current location and new location
                wall_y = (new_y + y) // 2
                wall_x = (new_x + x) // 2
                
                # cut path
                maze.set((wall_y, wall_x), path_tile)
                maze.set((new_y, new_x), path_tile)

                cells.append((new_y, new_x))
                break
        else:
            cells.pop()
    
    return maze