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

# --- Pygame configuration---
WIDTH, HEIGHT = 600, 600
GRID_SIZE = 30  # Size of the cell
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
SQUARE_SIZE = 30

# --- Object configuration ---
CIRCLE_RADIUS = 15

# --- Reinforcement Learning parameters (DQN) ---
BATCH_SIZE = 64        # Size of the batch
GAMMA = 0.99         # Discount Factor
EPSILON_START = 1.0  # Initial epsilon for exploration
EPSILON_END = 0.05   # Minimum epsilon
EPSILON_DECAY = 0.9995 # Epsilon reduction rate per episode
LR = 0.001           # Learning Rate to optimise the network
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
STATE_SIZE = 2

MODEL_PATH = 'dqn_agent_model.pth' # The file of a model who was trained

# --- Option to load a model ---
LOAD_MODEL = True
# CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Neural network definition (Q-Network) ---
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

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

def get_random_position():
    """Generate a random position in the grid."""
    x = random.randint(0, CELL_WIDTH - 1) * GRID_SIZE
    y = random.randint(0, CELL_HEIGHT - 1) * GRID_SIZE
    return x, y

def get_state_vector(player_x, player_y, target_x, target_y):
    """
    Compute the actual state as a vector for the neural network.
    Normalize the relative distances (dx, dy).
    """
    player_grid_x = player_x // CELL_WIDTH
    player_grid_y = player_y // CELL_HEIGHT
    target_grid_x = target_x // CELL_WIDTH
    target_grid_y = target_y // CELL_HEIGHT

    dx_norm = (target_grid_x - player_grid_x) / (GRID_SIZE - 1)
    dy_norm = (target_grid_y - player_grid_y) / (GRID_SIZE - 1)

    return torch.tensor([dx_norm, dy_norm], dtype=torch.float32, device=device).unsqueeze(0)

def choose_action(state_tensor, epsilon):
    """
    Select an action using epsilon-greedy.
    """
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, NUM_ACTIONS - 1)
    else:
        with torch.no_grad():
            return policy_net(state_tensor).argmax(1).item()

# --- Classes of the game ---
class Game:
    def __init__(self):
        self.player_x, self.player_y = get_random_position()
        self.target_x, self.target_y = get_random_position()
        self.wall_x, self.wall_y = get_random_position()
        self.score = 0
        self.game_over = False

    def reset(self):
        """Reset the game for a new episode."""
        self.player_x, self.player_y = get_random_position()
        self.target_x, self.target_y = get_random_position()
        self.wall_x, self.wall_y = get_random_position()
        while (self.player_x == self.target_x and self.player_y == self.target_y):
            self.target_x, self.target_y = get_random_position()
            
        self.score = 0
        self.game_over = False

    def step(self, action_index):
        """
        Exect a new stept in the enviroment based on the action chossed by the agent.
        Return (next_state_tensor, reward, done).
        """
        dx, dy = ACTIONS[action_index]
        
        self.player_x += dx * CELL_WIDTH
        self.player_y += dy * CELL_HEIGHT

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
            reward = 100  # High reward for eating the circle
            done = True
            self.score += 1

        next_state_tensor = get_state_vector(self.player_x, self.player_y, self.target_x, self.target_y)

        return next_state_tensor, reward, done

    def draw(self):
        """Draw everything on the screen"""
        screen.fill(BLACK)
        pygame.draw.rect(screen, RED, (self.player_x, self.player_y, SQUARE_SIZE, SQUARE_SIZE))
        pygame.draw.circle(screen, BLUE, (self.target_x+CIRCLE_RADIUS, self.target_y+CIRCLE_RADIUS), CIRCLE_RADIUS)
        pygame.draw.rect(screen, WHITE, (self.wall_x, self.wall_y, SQUARE_SIZE, SQUARE_SIZE))
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
    num_episodes = 5000 # Number of episodes from the training
    epsilon = EPSILON_START # Resete epsilon for the training

# --- Main loop for the training ---
if num_episodes > 0:
    game = Game()
    num_episodes = 5000
    episodes_rewards = []
    moving_avg_rewards = deque(maxlen=100)
    avg_rewards_plot = []
    epsilon = EPSILON_START

    print("Initiating train with DQN...")
    for episode in range(num_episodes):
        game.reset()
        current_state_tensor = get_state_vector(game.player_x, game.player_y, game.target_x, game.target_y)
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

    MODEL_SAVE_PATH = 'dqn_agent_model.pth' # Nombre del archivo para guardar
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
epsilon = 0.01 

game = Game()
running = True

start_time = pygame.time.get_ticks()
time_limit_ms = 10 * 1000 

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    current_state_tensor = get_state_vector(game.player_x, game.player_y, game.target_x, game.target_y)
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