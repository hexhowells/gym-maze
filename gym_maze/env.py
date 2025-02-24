import gymnasium as gym
import pygame
import random
import math
import numpy as np
from gymnasium import spaces
from .generator import generate_maze


SCREEN_WIDTH, SCREEN_HEIGHT = 256, 256
CELL_SIZE = 20
FOV = math.pi / 3
NUM_RAYS = 120
MAX_DEPTH = 500

LEFT = 0
RIGHT = 1
FORWARD = 2
BACKWARD = 3

GREY = (150, 150, 150)
BLACK = (0, 0, 0)
FLOOR_COLOR = (50, 50, 50)
CEILING_COLOR = (100, 100, 100)


class MazeEnv(gym.Env):
    def __init__(self, maze_width=101, maze_height=101, headless=False, early_stop_threshold=1000):
        super(MazeEnv, self).__init__()
        
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.grid = generate_maze(maze_width, maze_height)
        self.total_path_tiles = self.grid.count(' ')
        self.headless = headless
        
        self.wall_colors = {}
        for cell in self.grid.all_points():
            r = random.random()
            self.wall_colors[cell] = (
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ) if r < 0.2 else GREY
        
        self.player_x = 60
        self.player_y = 60
        self.player_angle = 0
        self.player_speed = 1
        self.visited_cells = set()
        self.total_cells_visited = 0
        self.steps_since_last_reward = 0
        self.early_stop_threshold = early_stop_threshold
        self.last_move = None
        self.total_steps = 0
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)
        
        pygame.init()
        
        if not self.headless:
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            self.clock = pygame.time.Clock()
        else:
            self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))


    def __str__(self):
        return str(self.grid)


    def _get_info(self):
        return {
            'player_loc': (self.player_x, self.player_y),
            'player_angle': self.player_angle,
            'player_speed': self.player_speed,
            'visited_cells': self.visited_cells,
            'steps_since_last_reward': self.steps_since_last_reward
        }


    def _update_player_loc(self, new_x, new_y):
        if self.grid.get((int(new_y / CELL_SIZE), int(new_x / CELL_SIZE))) == 0:
            self.player_x = new_x
            self.player_y = new_y
        

    def step(self, action):
        self.total_steps += 1

        # take action
        if action == LEFT:
            self.player_angle -= 0.05
        elif action == RIGHT:
            self.player_angle += 0.05
        elif action == FORWARD:
            new_x = self.player_x + (math.cos(self.player_angle) * self.player_speed)
            new_y = self.player_y + (math.sin(self.player_angle) * self.player_speed)
            self._update_player_loc(new_x, new_y)
        elif action == BACKWARD:
            new_x = self.player_x - (math.cos(self.player_angle) * self.player_speed)
            new_y = self.player_y - (math.sin(self.player_angle) * self.player_speed)
            self._update_player_loc(new_x, new_y)

        # get current tile
        cell = (int(self.player_x / CELL_SIZE), int(self.player_y / CELL_SIZE))
        
        # compute reward
        if cell not in self.visited_cells:
            reward = 1
            self.visited_cells.add(cell)
            self.steps_since_last_reward = 0
        else:
            reward = 0
            self.steps_since_last_reward += 1
        
        # get current frame
        obs = self.render(mode='rgb_array')

        # check for completion or early stopping
        terminated = len(self.visited_cells) == self.total_path_tiles
        truncated = self.steps_since_last_reward >= self.early_stop_threshold
        
        return obs, reward, terminated, truncated, self._get_info()


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.player_x, self.player_y = 60, 60
        self.player_angle = 0
        self.visited_cells = set()
        self.steps_since_last_reward = 0
        self.total_cells_visited = 0
        self.total_steps = 0

        obs = self.render(mode='rgb_array')
        return obs, {}


    def render(self, mode='human'):
        # draw empty environment (walls, floor, etc)
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, FLOOR_COLOR, (0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
        pygame.draw.rect(self.screen, CEILING_COLOR, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
        
        start_angle = self.player_angle - FOV / 2

        # use DDA for each ray
        for ray in range(NUM_RAYS):
            # calculate current ray angle
            angle = start_angle + ray * (FOV / NUM_RAYS)
            sin_a, cos_a = math.sin(angle), math.cos(angle)

            # get the current tile coord that the player is in
            map_x = int(self.player_x / CELL_SIZE)
            map_y = int(self.player_y / CELL_SIZE)

            # distance the ray has to travel along X and Y to cross a tile
            delta_dist_x = abs(1 / cos_a) if cos_a != 0 else 1e30
            delta_dist_y = abs(1 / sin_a) if sin_a != 0 else 1e30

            
            # calculate the direction the ray moves (step)
            # calcuate the distance to the next side of the grid
            #
            # -cos | +cos
            # +sin | +sin
            # -----+------
            # -cos | -cos
            # -sin | +sin

            if cos_a < 0:
                step_x = -1
                side_dist_x = (self.player_x / CELL_SIZE - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - self.player_x / CELL_SIZE) * delta_dist_x

            if sin_a < 0:
                step_y = -1
                side_dist_y = (self.player_y / CELL_SIZE - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - self.player_y / CELL_SIZE) * delta_dist_y

            # DDA
            hit = False
            side = 0
            while not hit:
                # jump to next map square
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1

                # check if we've hit a wall
                if self.grid.get((map_y, map_x)) == 1:
                    hit = True

            # calculate distance projected on camera direction (avoid fish-eye)
            if side == 0:
                perp_wall_dist = (side_dist_x - delta_dist_x)
            else:
                perp_wall_dist = (side_dist_y - delta_dist_y)

            # convert tile-based distance to world units
            perp_wall_dist *= CELL_SIZE

            # correct for fish-eye by angle difference
            perp_wall_dist *= math.cos(self.player_angle - angle)

            # compute wall height
            if perp_wall_dist > 0:
                wall_height = min(SCREEN_HEIGHT, int(5000 / (perp_wall_dist + 0.0001)))
            else:
                wall_height = SCREEN_HEIGHT

            # determine wall slice position
            wall_x = ray * (SCREEN_WIDTH / NUM_RAYS)
            wall_width = SCREEN_WIDTH / NUM_RAYS
            wall_y = (SCREEN_HEIGHT - wall_height) / 2

            # choose color
            color = self.wall_colors.get((map_x, map_y), GREY)
            pygame.draw.rect(self.screen, color, (wall_x, wall_y, wall_width+1, wall_height))
        
        if mode == 'human':
            pygame.display.flip()
        elif mode == 'rgb_array':
            return np.transpose(pygame.surfarray.array3d(self.screen), (2, 0, 1))
    

    def close(self):
        pygame.quit()
