import gym
import pygame
import random
import math
import numpy as np
from gym import spaces
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
    metadata = {'render.modes': ['human', 'rgb_array']}
    
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
        done = len(self.visited_cells) == self.total_path_tiles or self.steps_since_last_reward >= self.early_stop_threshold
        
        return obs, reward, done, self._get_info()


    def reset(self):
        self.player_x, self.player_y = 60, 60
        self.player_angle = 0
        self.visited_cells = set()
        self.steps_since_last_reward = 0
        self.total_cells_visited = 0
        self.total_steps = 0

        return self.render(mode='rgb_array')
    

    def render(self, mode='human'):
        self.screen.fill(BLACK)
        pygame.draw.rect(self.screen, FLOOR_COLOR, (0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
        pygame.draw.rect(self.screen, CEILING_COLOR, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
        
        start_angle = self.player_angle - FOV / 2
        for ray in range(NUM_RAYS):
            angle = start_angle + ray * (FOV / NUM_RAYS)
            sin_a, cos_a = math.sin(angle), math.cos(angle)
            
            for depth in range(1, MAX_DEPTH):
                target_x = int((self.player_x + cos_a * depth) / CELL_SIZE)
                target_y = int((self.player_y + sin_a * depth) / CELL_SIZE)
                
                if self.grid.get((target_y, target_x)) == 1:
                    color = self.wall_colors.get((target_x, target_y), GREY)
                    depth *= math.cos(self.player_angle - angle)
                    wall_height = min(SCREEN_HEIGHT, 5000 / (depth + 0.0001))
                    wall_x = ray * (SCREEN_WIDTH / NUM_RAYS)
                    wall_width = SCREEN_WIDTH / NUM_RAYS
                    wall_y = (SCREEN_HEIGHT - wall_height) / 2
                    pygame.draw.rect(self.screen, color, (wall_x, wall_y, wall_width + 1, wall_height))
                    break
        
        if mode == 'human':
            pygame.display.flip()
        elif mode == 'rgb_array':
            return np.transpose(pygame.surfarray.array3d(self.screen), (2, 0, 1))
    

    def close(self):
        pygame.quit()
