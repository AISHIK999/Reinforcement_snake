# ================
# Import libraries
# ================
import torch
import random
import numpy as np
from collections import deque
from game import Game, Direction, coordinate, QLearning, QTraining, plot

# ================
# End import
# ================

# ================
# Define dependencies
# ================
MEM_LIMIT = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001


# ================
# End dependencies
# ================

# ================
# Begin the AI model
# ================


class Player:
    def __init__(self):
        self.number_of_actions = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MEM_LIMIT)
        self.runner = QLearning(11, 256, 3)
        self.trainer = QTraining(self.runner, learning_rate=LEARNING_RATE, gamma=self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        turn_left = coordinate(head.x - 20, head.y)
        turn_right = coordinate(head.x + 20, head.y)
        turn_up = coordinate(head.x, head.y - 20)
        turn_down = coordinate(head.x, head.y + 20)

        move_west = game.direction == Direction.WEST
        move_east = game.direction == Direction.EAST
        move_north = game.direction == Direction.NORTH
        move_south = game.direction == Direction.SOUTH

        # ================
        # Set movement parameters
        # ================

        state = [
            # Check for obstacles in front and determine next step
            # If there is obstruction in front, move left or right
            # If there is obstruction by the sides (left or right), move up or down

            (move_east and game.collision(turn_right)) or (move_west and game.collision(turn_left)) or (
                    move_north and game.collision(turn_up)) or (move_south and game.collision(turn_down)),

            (move_north and game.collision(turn_right)) or (move_south and game.collision(turn_left)) or (
                    move_west and game.collision(turn_up)) or (move_east and game.collision(turn_down)),

            (move_south and game.collision(turn_right)) or (move_north and game.collision(turn_left)) or (
                    move_east and game.collision(turn_up)) or (move_west and game.collision(turn_down)),

            move_west, move_east, move_north, move_south,

            # Guide the snake towards the food
            # Turn the snake head towards the coordinates of the placed food
            game.food.x < game.head.x,
            game.food.x > game.head.x,
            game.food.y < game.head.y,
            game.food.y > game.head.y
        ]

        return np.array(state, dtype=int)
        # ================
        # End movement parameters
        # ================

    # Use savedata to improve
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Train model based on long time data
    def train_long(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)
        else:
            mini_sample = self.memory

        mult_state, mult_action, mult_reward, mult_next_state, mult_done = zip(*mini_sample)
        self.trainer.train_step(mult_state, mult_action, mult_reward, mult_next_state, mult_done)

    # Train model based on short time data
    def train_short(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    # Make a move based on training
    def get_action(self, state):
        self.epsilon = (80 - self.number_of_actions)
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.runner(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move
# ================
# End AI model
# ================

# ================
# The training code
# ================


def train():
    plot_score = []
    plot_mean = []
    total_score = 0
    record = 0
    player = Player()
    game = Game()
    while True:
        """
        First we get the previous state
        Then we make a prediction and make a new move
        The data from the new state and the old state will be used to train th model (short time training data)
        Finally, the record will be remembered by the model
        """
        previous_state = player.get_state(game)
        next_move = player.get_action(previous_state)
        reward, done, score = game.next_action(next_move)
        next_state = player.get_state(game)
        player.train_short(previous_state, next_move, reward, next_state, done)
        player.remember(previous_state, next_move, reward, next_state, done)

        if done:
            """
            When the short time training is completed, as limited by the BATCH_SIZE,
            the model will take all the data for long time training
            This decides, what trend moves on to the next generation 
            """
            game.reset()
            player.number_of_actions += 1
            player.train_long()

            if score > record:
                record = score
                player.runner.save()

            # Print out the following data on the console
            print(f"Generation: {player.number_of_actions}\n"
                  f"Record: {record}\n"
                  f"Score: {score}\n")

            # Plot the mean score to view the overall improvement of the model
            plot_score.append(score)
            total_score += score
            mean_score = round((total_score / player.number_of_actions), 3)
            plot_mean.append(mean_score)
            plot(plot_score, plot_mean)
# ================
# End training code
# ================


if __name__ == "__main__":
    train()
