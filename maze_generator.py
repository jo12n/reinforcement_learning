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

def get_random_position(orientation, init_x, init_y, lenght):
    """Generate a random position in the grid."""
    if orientation == 0:
        final_x = lenght//GRID_SIZE
        x = init_x+random.randint(0, final_x)*GRID_SIZE
        y = init_y
    else:
        final_y = (lenght)//GRID_SIZE
        x = init_x
        y = init_y+random.randint(0, final_y)*GRID_SIZE
    return x, y

class Maze:
    def __init__(self, layers):
        self.wall1_x, self.wall1_y = get_random_position(1, 0, WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*WALL_SIZE)
        self.wall2_x, self.wall2_y = get_random_position(1, WIDTH-GRID_SIZE, WALL_SIZE,CELL_WIDTH*GRID_SIZE-2*WALL_SIZE)
        print(CELL_WIDTH*GRID_SIZE-2*WALL_SIZE)
        self.layers = layers
        self.space_between = 4
        self.top_x, self.top_y = 0, 0
        self.right_x, self.right_y = WIDTH-GRID_SIZE, 0
        self.down_x, self.down_y = WALL_SIZE, HEIGHT-GRID_SIZE
        self.left_x, self.left_y = 0, WALL_SIZE
        self.nowalls = [(self.wall1_x,self.wall1_y),(self.wall2_x,self.wall2_y)]
        self.walls_h = []
        self.walls_v = []
        top_back = [0,0]
        down_back = [0,0]
        left_back = [0,self.wall1_y]
        right_back = [0,self.wall2_y]
        for i in range(1,self.layers):
            if random.randint(0,2) == 0:
                new_top = get_random_position(0, self.top_x+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, self.top_y+self.space_between*i*WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                if top_back[0] == i-1:
                    while abs(new_top[0]-top_back[1]) < 2*WALL_SIZE:
                        new_top = get_random_position(0, self.top_x+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, self.top_y+self.space_between*i*WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                self.nowalls.append(new_top)
                top_back = [i,new_top[0]]
            else:
                new_left = get_random_position(1, self.left_x+self.space_between*i*WALL_SIZE, self.left_y+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                if left_back[0] == i-1:
                    while abs(new_left[1]-left_back[1]) < 2*WALL_SIZE:
                        new_left = get_random_position(1, self.left_x+self.space_between*i*WALL_SIZE, self.left_y+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                self.nowalls.append(new_left)
                left_back = [i,new_left[1]]
            if random.randint(0,2) == 0:
                new_down = get_random_position(0, self.down_x+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, self.down_y-self.space_between*i*WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                if down_back[0] == i-1:
                    while abs(new_down[0]-left_back[1]) < 2*WALL_SIZE:
                        new_down = get_random_position(0, self.down_x+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, self.down_y-self.space_between*i*WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                self.nowalls.append(new_down)
                down_back = [i,new_down[0]]
            else:
                new_right = get_random_position(1, self.right_x-self.space_between*i*WALL_SIZE, self.right_y+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                if right_back[0] == i-1:
                    while abs(new_right[1]-right_back[1]) < 2*WALL_SIZE:
                        new_right = get_random_position(1, self.right_x-self.space_between*i*WALL_SIZE, self.right_y+self.space_between*(i+1)*WALL_SIZE+WALL_SIZE, CELL_WIDTH*GRID_SIZE-2*self.space_between*(i+1)*WALL_SIZE-2*WALL_SIZE)
                self.nowalls.append(new_right)
                right_back = [i,new_right[1]]

    def draw(self):
        """Draw everything on the screen"""
        screen.fill(BLACK)
        
        for i in range(self.layers):
            lenght_wall = CELL_WIDTH*GRID_SIZE-2*self.space_between*i*WALL_SIZE-WALL_SIZE 
            pygame.draw.rect(screen, WHITE, (self.top_x+self.space_between*i*WALL_SIZE, self.top_y+self.space_between*i*WALL_SIZE, lenght_wall, WALL_SIZE))
            pygame.draw.rect(screen, WHITE, (self.right_x-self.space_between*i*WALL_SIZE, self.right_y+self.space_between*i*WALL_SIZE, WALL_SIZE, lenght_wall))
            pygame.draw.rect(screen, WHITE, (self.down_x+self.space_between*i*WALL_SIZE, self.down_y-self.space_between*i*WALL_SIZE, lenght_wall, WALL_SIZE))
            pygame.draw.rect(screen, WHITE, (self.left_x+self.space_between*i*WALL_SIZE, self.left_y+self.space_between*i*WALL_SIZE, WALL_SIZE, lenght_wall))
        for no_wall in self.nowalls:
            pygame.draw.rect(screen, BLACK, (no_wall[0], no_wall[1], WALL_SIZE, WALL_SIZE))
        for i in range(self.layers-1):
            top_right = [WIDTH-((i+1)*self.space_between+1)*WALL_SIZE,WALL_SIZE*(1+i*self.space_between)]
            down_left = [((i+1)*self.space_between+1)*WALL_SIZE-WALL_SIZE,HEIGHT-WALL_SIZE*(1+(i+1)*self.space_between)+WALL_SIZE]
            pygame.draw.rect(screen, WHITE, (top_right[0], top_right[1], WALL_SIZE, 3*WALL_SIZE))
            pygame.draw.rect(screen, WHITE, (down_left[0], down_left[1], WALL_SIZE, 3*WALL_SIZE))

        pygame.display.flip()

maze = Maze(7)

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