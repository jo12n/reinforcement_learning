import pygame
import random

WIDTH, HEIGHT = 600, 600
GRID_SIZE = 10  # Size of the cell
CELL_WIDTH = WIDTH // GRID_SIZE
CELL_HEIGHT = HEIGHT // GRID_SIZE
FPS = 60
BLACK = "black"
WHITE = "white"
WALL_SIZE = GRID_SIZE

# pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True

def wall_generator(nowall_x, nowall_y):
    wall1_x = nowall_x
    wall1_y = 0
    height_wall1 = nowall_y
    wall2_x = nowall_x
    wall2_y = nowall_y + WALL_SIZE
    height_wall2 = HEIGHT - nowall_y - WALL_SIZE
    return wall1_x, wall1_y, wall2_x, wall2_y+1, height_wall1-1, height_wall2

def get_random_position(orientation, init_x, init_y):
    """Generate a random position in the grid."""
    if orientation == 0:
        x = random.randint(init_x, CELL_WIDTH*GRID_SIZE-WALL_SIZE)
        y = init_y
    else:
        x = init_x
        y = random.randint(init_y, CELL_HEIGHT*GRID_SIZE-WALL_SIZE)
    return x, y

class Maze:
    def __init__(self):
        self.layers = 1
        self.wall1_x, self.wall1_y = get_random_position(1, 0, WALL_SIZE)
        self.wall2_x, self.wall2_y = get_random_position(1, WIDTH-GRID_SIZE, 0)
    def draw(self):
        """Draw everything on the screen"""
        screen.fill(BLACK)
        top_x, top_y = 0, 0
        right_x, right_y = WIDTH-GRID_SIZE, 0
        down_x, down_y = WALL_SIZE, HEIGHT-GRID_SIZE
        left_x, left_y = 0, WALL_SIZE
        pygame.draw.rect(screen, WHITE, (top_x, top_y, CELL_WIDTH*GRID_SIZE-WALL_SIZE, WALL_SIZE))
        pygame.draw.rect(screen, WHITE, (right_x, right_y, WALL_SIZE, CELL_HEIGHT*GRID_SIZE-GRID_SIZE))
        pygame.draw.rect(screen, WHITE, (down_x, down_y, CELL_HEIGHT*GRID_SIZE-GRID_SIZE, WALL_SIZE))
        pygame.draw.rect(screen, WHITE, (left_x, left_y, WALL_SIZE, CELL_HEIGHT*GRID_SIZE))
        pygame.draw.rect(screen, BLACK, (self.wall1_x, self.wall1_y, WALL_SIZE, WALL_SIZE))
        pygame.draw.rect(screen, BLACK, (self.wall2_x, self.wall2_y, WALL_SIZE, WALL_SIZE))

        pygame.display.flip()

maze = Maze()

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("black")
    maze.draw()

    # RENDER YOUR GAME HERE

    # flip() the display to put your work on screen
    pygame.display.flip()

    clock.tick(FPS)  # limits FPS to 60

pygame.quit()