import pygame
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import time
from maze_generator import Maze

# --- Pygame configuration---
WIDTH, HEIGHT = 800, 600
GRID_SIZE = 10  # Size of the cell
CELL_WIDTH = WIDTH // GRID_SIZE
CELL_HEIGHT = HEIGHT // GRID_SIZE
FPS = 60
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Reinforcement Learning Cuadrado (DQN)")
clock = pygame.time.Clock()

# --- Colors ---
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# --- Agent configuration ---
SQUARE_SIZE = GRID_SIZE

# --- Object configuration ---
CIRCLE_RADIUS = GRID_SIZE/2

# --- Reinforcement Learning parameters (DQN) ---
BATCH_SIZE = 64        # Size of the batch
GAMMA = 0.99         # Discount Factor
EPSILON_START = 1.0  # Initial epsilon for exploration
EPSILON_END = 0.05   # Minimum epsilon
EPSILON_DECAY = 0.9999 # Epsilon reduction rate per episode
LR = 0.0002           # Learning Rate to optimise the network
TARGET_UPDATE = 10   # Update the network after this number of episodes
MEMORY_SIZE = 10000  # Max size of the replay buffer

# --- Penalty for stagnation ---
MAX_STEPS_PER_EPISODE = 1000 # Max of steps before the episode is finished
STAGNATION_PENALTY = -100    # Negative reward if the max is achived

# Define the actions
ACTIONS = {
    0: (0, -1),   # Up (dy = -1)
    1: (0, 1),    # Down (dy = 1)
    2: (-1, 0),   # Left (dx = -1)
    3: (1, 0)    # Right (dx = 1)
}
NUM_ACTIONS = len(ACTIONS)

# State dimension (input for the neural network)
STATE_SIZE = 7

MODEL_PATH = 'dqn_agent_model_wall_pen.pth' # The file of a model who was trained

# --- Option to load a model ---
LOAD_MODEL = False
# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Neural network definition (Q-Network) ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = x.view(-1, STATE_SIZE)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# --- Experience Memory (Replay Buffer) ---
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition."""
        self.memory.append(tuple(args))

    def sample(self, batch_size):
        """Show a random batch of the transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the actual number of transitions in the memory."""
        return len(self.memory)

# --- Funciones Auxiliares ---

def get_random_position(object):
    """Generate a random position in the grid."""
    if object == 0:
        x = random.randint(0, round((CELL_WIDTH - 1)/4)-1) * GRID_SIZE
        y = random.randint(0, CELL_HEIGHT - 1) * GRID_SIZE
    elif object == 1:
        x = random.randint(round((CELL_WIDTH - 1)/4)-1, round(3*(CELL_WIDTH - 1)/4)-1) * GRID_SIZE
        y = random.randint(0, CELL_HEIGHT - 1) * GRID_SIZE
    elif object == 2:
        x = random.randint(round(3*(CELL_WIDTH - 1)/4)-1, CELL_HEIGHT - 1) * GRID_SIZE
        y = random.randint(0, CELL_HEIGHT - 1) * GRID_SIZE
    return x, y

'''
def get_state_vector(player_x, player_y, target_x, target_y, nowall_x, nowall_y):
    """
    Compute the actual state as a vector for the neural network.
    Normalize the relative distances (dx, dy).
    """
    player_grid_x = player_x // CELL_WIDTH
    player_grid_y = player_y // CELL_HEIGHT
    target_grid_x = target_x // CELL_WIDTH
    target_grid_y = target_y // CELL_HEIGHT
    nowall_grid_x = nowall_x // CELL_WIDTH
    nowall_grid_y = nowall_y // CELL_HEIGHT

    dx_norm = (target_grid_x - player_grid_x) / (GRID_SIZE - 1)
    dy_norm = (target_grid_y - player_grid_y) / (GRID_SIZE - 1)

    if nowall_grid_y == player_grid_y:
        dx_wall_norm = (player_grid_x) / (GRID_SIZE - 1)
    else:
        dx_wall_norm = (player_grid_x-nowall_grid_x) / (GRID_SIZE - 1)
    
    if nowall_grid_y == player_grid_y and nowall_grid_x == player_grid_x:
        dy_wall_norm = 0
    else:
        dy_wall_norm = (player_grid_y) / (GRID_SIZE - 1)

    if target_grid_x > player_grid_x:
        dx_norm = 1
    else:
        dx_norm = 0

    if target_grid_y > player_grid_y:
        dy_norm = 1
    else:
        dy_norm = 0

    return torch.tensor([dx_norm, dy_norm, dx_wall_norm, dy_wall_norm], dtype=torch.float32, device=device).unsqueeze(0)
'''
def get_state_vector(player_x, player_y, target_x, target_y, left_bool, right_bool, center_bool):
    if target_x < player_x:
        target_left = 1
        target_right = 0
    elif target_x > player_x:
        target_left = 0
        target_right = 1
    else:
        target_left = 0
        target_right = 0

    if target_y < player_y:
        target_up = 1
        target_down = 0
    elif target_y > player_y:
        target_up = 0
        target_down = 1
    else:
        target_up = 0
        target_down = 0

    return torch.tensor([target_left, target_right, target_up, target_down, left_bool, right_bool, center_bool], dtype=torch.float32, device=device).unsqueeze(0)

def choose_action(state_tensor, epsilon):
    """
    Select an action using epsilon-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        with torch.no_grad():
            return policy_net(state_tensor).argmax(1).item()
        
def wall_generator(nowall_x, nowall_y):
    wall1_x = nowall_x
    wall1_y = 0
    height_wall1 = nowall_y
    wall2_x = nowall_x
    wall2_y = nowall_y + SQUARE_SIZE
    height_wall2 = HEIGHT - nowall_y - SQUARE_SIZE
    return wall1_x, wall1_y, wall2_x, wall2_y+1, height_wall1-1, height_wall2

# --- Classes of the game ---
class Game:
    def __init__(self):
        self.player_x, self.player_y = WIDTH-50, 100
        if random.randint(0,1) == 1:
            self.target_x, self.target_y = 50, HEIGHT-100
        else:
            self.target_x, self.target_y = 400, 300
        self.nowall_x, self.nowall_y = get_random_position(1)
        self.left_bool, self.right_bool, self.center_bool = 0,0,0
        self.score = 0
        self.game_over = False
        self.maze = Maze(3)


    def reset(self):
        """Reset the game for a new episode."""
        self.var_start = random.randint(0,6)
        self.player_x, self.player_y = WIDTH-50, 100
        if random.randint(0,1) == 1:
            self.target_x, self.target_y = 50, HEIGHT-100
        else:
            self.target_x, self.target_y = 400, 300
        self.nowall_x, self.nowall_y = get_random_position(1)
        while (self.target_x == self.nowall_x):
            self.target_x, self.target_y = get_random_position(0)
            
        self.score = 0
        self.game_over = False
        self.maze = Maze(2)

    def step(self, action_index):
        """
        Exect a new stept in the enviroment based on the action chossed by the agent.
        Return (next_state_tensor, reward, done).
        """
        dx, dy = ACTIONS[action_index]
        
        self.player_x += dx * GRID_SIZE
        self.player_y += dy * GRID_SIZE
            
        
        # Be sure to keep the square inside the screen
        self.player_x = max(0, min(self.player_x, WIDTH - SQUARE_SIZE))
        self.player_y = max(0, min(self.player_y, HEIGHT - SQUARE_SIZE))
        # Compute the reward
        reward = -0.1  # Negative reward for every step
        done = False
        # Check collision
        player_rect = pygame.Rect(self.player_x, self.player_y, SQUARE_SIZE, SQUARE_SIZE)
        target_rect = pygame.Rect(self.target_x, self.target_y, CIRCLE_RADIUS * 2, CIRCLE_RADIUS * 2)

        if player_rect.colliderect(target_rect):
            reward = 200  # High reward for eating the circle
            done = True
            self.score += 1
        else:
            reward, dx,dy = self.maze.colision(player_rect, dx, dy)
            self.player_x += dx * GRID_SIZE
            self.player_y += dy * GRID_SIZE
            self.player_x = max(0, min(self.player_x, WIDTH - SQUARE_SIZE))
            self.player_y = max(0, min(self.player_y, HEIGHT - SQUARE_SIZE))
        '''
        elif player_rect_min.colliderect(wall1_rect) or player_rect_max.colliderect(wall2_rect):
            if abs(self.target_x-self.player_x ) < abs(self.target_x-self.player_x + dx * GRID_SIZE):
                reward = 20
            else:
                reward = -30
        '''
        self.left_bool, self.right_bool,self.center_bool = self.maze.prox_walls(self.player_x, self.player_y, dx, dy)
        next_state_tensor = get_state_vector(self.player_x, self.player_y, self.target_x, self.target_y, self.left_bool, self.right_bool,self.center_bool)

        return next_state_tensor, reward, done

    def draw(self):
        """Draw everything on the screen"""
        screen.fill(BLACK)
        pygame.draw.rect(screen, RED, (self.player_x, self.player_y, SQUARE_SIZE, SQUARE_SIZE))
        self.maze.draw()
        pygame.draw.circle(screen, BLUE, (self.target_x+CIRCLE_RADIUS, self.target_y+CIRCLE_RADIUS), CIRCLE_RADIUS)
        pygame.display.flip()

# --- Functions to optimise the network ---
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.tensor(batch[1], dtype=torch.int64, device=device).unsqueeze(1)
    reward_batch = torch.tensor(batch[2], dtype=torch.float32, device=device)
    next_state_batch = torch.cat(batch[3])
    done_batch = torch.tensor(batch[4], dtype=torch.bool, device=device)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[~done_batch] = target_net(next_state_batch[~done_batch]).max(1)[0]

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Initialization of the network and memory ---
policy_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
target_net = DQN(STATE_SIZE, NUM_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = ReplayMemory(MEMORY_SIZE)

if LOAD_MODEL and os.path.exists(MODEL_PATH):
    # Load the model
    print(f"Loading the model from: {MODEL_PATH}")
    policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    policy_net.eval() # Set to evaluation mode for inference
    target_net.load_state_dict(policy_net.state_dict()) # Be sure that target_net is updated
    target_net.eval()
    epsilon = EPSILON_END # Reduce the epsilon value to the minimum, show you can show what the networks knows
    num_episodes = 0 # No train, just simulate
else:
    # Train a new model
    print("No model is saved or LOAD_MODEL is False. Initiating training with DQN...")
    num_episodes = 25000 # Number of episodes from the training
    epsilon = EPSILON_START # Resete epsilon for the training

# --- Main loop for the training ---
if num_episodes > 0:
    game = Game()
    num_episodes = 25000
    episodes_rewards = []
    moving_avg_rewards = deque(maxlen=100)
    avg_rewards_plot = []
    epsilon = EPSILON_START

    print("Initiating train with DQN...")
    for episode in range(num_episodes):
        game.reset()
        current_state_tensor = get_state_vector(game.player_x, game.player_y, game.target_x, game.target_y, game.left_bool, game.right_bool, game.center_bool)
        done = False
        total_reward = 0
        steps_in_episode = 0

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Loop of the episode
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            
            action = choose_action(current_state_tensor, epsilon)
            next_state_tensor, reward, done = game.step(action)

            # --- Check step limit for stagnation ---
            steps_in_episode += 1
            if steps_in_episode >= MAX_STEPS_PER_EPISODE:
                reward = STAGNATION_PENALTY # Penalty
                done = True                  # End of the episode
                
            memory.push(current_state_tensor, action, reward, next_state_tensor, done)

            current_state_tensor = next_state_tensor
            total_reward += reward
            
            optimize_model()

            game.draw()
            # Comment this line for a faster training
            # clock.tick(FPS) 

        episodes_rewards.append(total_reward)
        moving_avg_rewards.append(total_reward)

        if len(moving_avg_rewards) == 100:
            avg_rewards_plot.append(np.mean(moving_avg_rewards))
        else:
            avg_rewards_plot.append(np.mean(moving_avg_rewards))

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if episode % 100 == 0:
            print(f"Episode: {episode}/{num_episodes}, Total reward: {total_reward:.2f}, Epsilon: {epsilon:.4f}, Memory: {len(memory)}")
            if len(moving_avg_rewards) > 0:
                avg_reward_current = np.mean(moving_avg_rewards)
                print(f"  Average rewards (last {len(moving_avg_rewards)}): {avg_reward_current:.2f}")

    print("Train with DQN completed.")

    MODEL_SAVE_PATH = 'dqn_agent_model_maze.pth' # Nombre del archivo para guardar
    torch.save(policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved in: {MODEL_SAVE_PATH}")

    # --- Graficar el Progreso del Entrenamiento ---
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards_plot)
    plt.title('DQN: Average reward per episode')
    plt.xlabel('Episodes')
    plt.ylabel('Average reward')
    plt.grid(True)
    plt.show()

epsilon_simulation = EPSILON_END
# --- Simulación después del Entrenamiento (agente entrenado) ---
print("\n--- Completed training! Starting Simulation with DQN Agent ---")
epsilon = 0 

game = Game()
running = True

start_time = pygame.time.get_ticks()
time_limit_ms = 10 * 1000 

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    current_state_tensor = get_state_vector(game.player_x, game.player_y, game.target_x, game.target_y, game.left_bool, game.right_bool, game.center_bool)
    action = choose_action(current_state_tensor, epsilon)
    
    next_state_tensor, reward, done = game.step(action)

    if pygame.time.get_ticks() - start_time >= time_limit_ms:
        print(f"Circle not eaten! Score: 0 (Time finished)")
        game.reset()
        start_time = pygame.time.get_ticks() 

    if done:
        print(f"Circle eaten! Score: {game.score}")
        game.reset()
        start_time = pygame.time.get_ticks() 

    game.draw()
    clock.tick(FPS)


pygame.quit()
