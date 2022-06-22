# ================
# Import libraries
# ================
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fun
import torch.optim as optim
import os

import matplotlib.pyplot as plt
from IPython import display

# ================
# End import
# ================

# ======================================================================================================================
# GAME CODE
# ======================================================================================================================
pygame.init()
font = pygame.font.Font("comfortaa.ttf", 15)


# ================
# Navigation
# ================
class Direction(Enum):
    NORTH = 1
    SOUTH = 2
    WEST = 3
    EAST = 4


coordinate = namedtuple("Turn", ["x", "y"])
# ================
# End navigation
# ================


# ================
# Game parameters
# ================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
CELL = 20  # Size of each cell
SPEED = 30  # Speed of the snake


# ================
# End game parameters
# ================


# ================
# Game window
# ================
class Game:
    # ================
    # GUI settings
    # ================
    def gui_settings(self):
        self.display.fill(WHITE)

        # Render snake
        for border in self.snake:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(border.x, border.y, CELL, CELL))
            pygame.draw.rect(self.display, YELLOW, pygame.Rect(border.x + 2, border.y + 2, 6, 6))

        # Render food
        pygame.draw.rect(self.display,
                         RED,
                         pygame.Rect(self.food.x,
                                     self.food.y,
                                     CELL,
                                     CELL)
                         )

        # Render score
        text = font.render(f"Score: {str(self.score)}", True, BLACK)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def __init__(self, width=720, height=720):
        pygame.display.set_caption("Snake Arena")
        self.width = width
        self.height = height
        self.display = pygame.display.set_mode((width, height))
        self.clock = pygame.time.Clock()
        self.reset()

        # Initialize components
        # self.snake = [0]
        # self.food = None
        # self.head = None
        # self.score = None
        # self.refresh = None
        # self.direction = None

    # ================
    # End GUI settings
    # ================

    # ================
    # Snake navigation
    # ================
    def navigate(self, action):
        right = [Direction.EAST, Direction.SOUTH, Direction.WEST, Direction.NORTH]
        position = right.index(self.direction)
        if np.array_equal(action, np.array([1, 0, 0])):
            change_position = right[position]
        elif np.array_equal(action, np.array([0, 1, 0])):
            new_position = ((position + 1) % 4)
            change_position = right[new_position]
        else:
            new_position = ((position - 1) % 4)
            change_position = right[new_position]
        self.direction = change_position

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.EAST:
            x += CELL
        elif self.direction == Direction.WEST:
            x -= CELL
        elif self.direction == Direction.SOUTH:
            y += CELL
        elif self.direction == Direction.NORTH:
            y -= CELL
        self.head = coordinate(x, y)

    # ================
    # End Snake navigation
    # ================

    # ================
    # Game reset
    # ================
    def reset(self):
        self.direction = Direction.EAST
        self.head = coordinate(self.width // 2, self.height // 2)
        self.snake = [self.head, coordinate(self.head.x - CELL, self.head.y),
                      coordinate(self.head.x - (2 * CELL), self.head.y)]
        self.score = 0
        self.food = None
        self.food_gen()
        self.refresh = 0

    # ================
    # End reset
    # ================

    # ================
    # Check for collision
    # ================
    def collision(self, collide=None):
        if collide is None:
            collide = self.head
        # Quit if the snake head hits the border
        if collide.x > self.width - CELL or collide.x < 0 or collide.y > self.height - CELL or collide.y < 0:
            return True
        # Quit if the snake head hits itself
        if collide in self.snake[1:]:
            return True
        return False

    # ================
    # End collision check
    # ================

    # ================
    # Game actions
    # ================
    def next_action(self, action):
        self.refresh += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Insert new head at the beginning towards the direction of the movement
        self.navigate(action)
        self.snake.insert(0, self.head)

        """
        Initialize the reward variable as 0
        We will check for collisions
        If the snake hits the border:
            i) Game will be reset
            ii) Reward will be set to (-1)
        Negative rewards will result in removal of the snake generation
        """
        # Begin collision check
        reward = 0
        end = False
        if self.collision() or self.refresh > 100 * len(self.snake):
            end = True
            reward = -1
            return reward, end, self.score
        # End collision check

        """
        We consider the snake eating the food as a positive step
        The more food it eats, the more reward it gets
        Increase rewards lets the snake generation pass the test
        """
        # Update score
        if self.head == self.food:
            self.score += 1
            reward = 1
            self.food_gen()
        else:
            self.snake.pop()
        # End update score

        # Update the arena
        self.gui_settings()
        self.clock.tick(SPEED)
        # End update

        # WASTED!
        return reward, end, self.score

    # ================
    # End game actions
    # ================

    # ================
    # Render food
    # ================
    def food_gen(self):
        x = random.randint(0, (self.width - CELL) // CELL) * CELL
        y = random.randint(0, (self.height - CELL) // CELL) * CELL
        self.food = coordinate(x, y)
        if self.food in self.snake:
            self.food_gen()
    # ================
    # End rendering food
    # ================


# ================
# End game window
# ================

# ======================================================================================================================
# END GAME CODE
# ======================================================================================================================

# ======================================================================================================================
# TRAINER CODE
# ======================================================================================================================

# ================
# Learning code
# ================
"""
We will use quality learning to make our snake model evolve
The save data will be stored in 'data/savedata.pth'
The model will gain reward points based on eating food and gaining rewards
The models with higher rewards get accepted and the ones wih lower rewards gets removed
"""


class QLearning(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.initial = nn.Linear(input_size, hidden_size)
        self.final = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = fun.relu(self.initial(x))
        x = self.final(x)
        return x

    def save(self, filename="savedata.pth"):
        path_to_data = "./data"
        if not os.path.exists(path_to_data):
            os.makedirs(path_to_data)
        filename = os.path.join(path_to_data, filename)
        torch.save(self.state_dict(), filename)


# ================
# End learning code
# ================

# ================
# Training code
# ================


class QTraining:
    def __init__(self, trainer, learning_rate, gamma):
        self.learning_rate = learning_rate
        self.trainer = trainer
        self.gamma = gamma
        self.optimizer = optim.Adam(trainer.parameters(), lr=self.learning_rate)  # Preferred optimizer
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, new_state, done):
        state = torch.tensor(state, dtype=torch.float)
        new_state = torch.tensor(new_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            new_state = torch.unsqueeze(new_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        prediction = self.trainer(state)
        target = prediction.clone()
        for index in range(len(done)):
            q_new = reward[index]
            if not done[index]:
                q_new = reward[index] + self.gamma * torch.max(self.trainer(new_state[index]))
            target[index][torch.argmax(action).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, prediction)
        loss.backward()

        self.optimizer.step()


# ================
# End training code
# ================


# ======================================================================================================================
# END TRAINER CODE
# ======================================================================================================================

# ======================================================================================================================
# DATA PLOTTING CODE
# ======================================================================================================================
plt.ion()


def plot(score, mean):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    window = plt.gcf()
    window.canvas.set_window_title("Training data")
    plt.clf()
    plt.title("Training data")
    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.plot(score)
    plt.plot(mean)
    plt.ylim(ymin=0)
    plt.text(len(score) - 1, score[-1], str(score[-1]))
    plt.text(len(mean) - 1, mean[-1], str(mean[-1]))
    plt.show(block=False)
    plt.pause(1)
# ======================================================================================================================
# END PLOTTING
# ======================================================================================================================
