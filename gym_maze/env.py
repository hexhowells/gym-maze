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
    
    def __init__(self, maze_width=101, maze_height=101):
        super(MazeEnv, self).__init__()
        
        self.maze_width = maze_width
        self.maze_height = maze_height
        self.grid = generate_maze(maze_width, maze_height)
        
        # Set wall colors
        self.wall_colors = {}
        for cell in self.grid.all_points():
            r = random.random()
            self.wall_colors[cell] = (
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            ) if r < 0.2 else GREY
        
        # Player state
        self.player_x = 60
        self.player_y = 60
        self.player_angle = 0
        self.player_speed = 1
        self.visited_cells = set()
        self.total_cells_visited = 0
        
        # Left, Right, Forward, Backward
        self.action_space = spaces.Discrete(4)
        
        # RGB image of screen
        self.observation_space = spaces.Box(low=0, high=255, shape=(SCREEN_WIDTH, SCREEN_HEIGHT, 3), dtype=np.uint8)
        
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()


    def step(self, action):
        # Apply action
        if action == LEFT:
            self.player_angle -= 0.05
        elif action == RIGHT:
            self.player_angle += 0.05
        elif action == FORWARD:
            new_x = self.player_x + (math.cos(self.player_angle) * self.player_speed)
            new_y = self.player_y + (math.sin(self.player_angle) * self.player_speed)

            if self.grid.get((int(new_y / CELL_SIZE), int(new_x / CELL_SIZE))) == 0:
                self.player_x = new_x
                self.player_y = new_y
        elif action == BACKWARD:
            new_x = self.player_x - (math.cos(self.player_angle) * self.player_speed)
            new_y = self.player_y - (math.sin(self.player_angle) * self.player_speed)

            if self.grid.get((int(new_y / CELL_SIZE), int(new_x / CELL_SIZE))) == 0:
                self.player_x = new_x
                self.player_y = new_y
        
        # Track visited cells
        cell = (int(self.player_x / CELL_SIZE), int(self.player_y / CELL_SIZE))
        self.visited_cells.add(cell)
        
        reward = len(self.visited_cells) - self.total_cells_visited
        total_cells_visited = len(self.visited_cells)
        obs = self.render(mode='rgb_array')
        
        return obs, reward, False, {}

    def reset(self):
        self.player_x, self.player_y = 60, 60
        self.player_angle = 0
        self.visited_cells = set()
        return self.render(mode='rgb_array')
    
    def render(self, mode='human'):
        self.screen.fill(BLACK)
        
        # Draw floor and ceiling
        pygame.draw.rect(self.screen, FLOOR_COLOR, (0, SCREEN_HEIGHT // 2, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
        pygame.draw.rect(self.screen, CEILING_COLOR, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT // 2))
        
        # Raycasting
        start_angle = self.player_angle - FOV / 2
        for ray in range(NUM_RAYS):
            angle = start_angle + ray * (FOV / NUM_RAYS)
            sin_a, cos_a = math.sin(angle), math.cos(angle)
            
            for depth in range(1, MAX_DEPTH):
                target_x = int((self.player_x + cos_a * depth) / CELL_SIZE)
                target_y = int((self.player_y + sin_a * depth) / CELL_SIZE)
                
                # Draw slice of wall
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
            return pygame.surfarray.array3d(self.screen)
    
    def close(self):
        pygame.quit()
