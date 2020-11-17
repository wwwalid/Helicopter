import pygame as pg
import numpy as np
import random
import neat
import os

# Colors
black       = (0,0,0)
white       = (255,255,255)
red         = (255, 0, 0)
green       = (0, 255, 0)
darkgreen   = (0, 200, 0)
yellow      = (255, 255, 0)
orange      = (255, 153, 51)
blue        = (0, 0, 255)
purple      = (128, 0, 128)

# Create window
xmax = 1200  # pixels
ymax = 800  # pixels
scr = pg.display.set_mode((xmax, ymax))
pg.display.set_caption('Helicopter')

gen = 0
best = []

class Wall:
    WIDTH = xmax
    HEIGHT = 100
    COLOR = green

    def __init__(self, y):
        self.y = y
        self.rect = pg.Rect(0, self.y, self.WIDTH, self.HEIGHT)
    def draw(self, scr):
        pg.draw.rect(scr, self.COLOR, self.rect)

class Obstacle:
    LENGTH = ymax / 4
    WIDTH = xmax / 30
    COLOR = green
    vx = -2000

    def __init__(self, x):
        self.x = x
        self.y = random.randrange(self.LENGTH/2 + Wall.HEIGHT, ymax - self.LENGTH * 1.5 - Wall.HEIGHT)
        self.y2 = self.y + self.LENGTH
        self.rect = pg.Rect(self.x, 0, self.WIDTH, self.LENGTH)
        self.rect2 = pg.Rect(self.x, 0, self.WIDTH, self.LENGTH)

    def move(self, dt):
        self.x += self.vx * dt
        self.rect.left = self.x
        self.rect.height = self.y
        self.rect.top = 0
        self.rect.bottom = self.y


        self.rect2.left = self.x
        self.rect2.height = ymax - self.y
        self.rect2.top = self.y + self.LENGTH

    def draw(self, scr):
        pg.draw.rect(scr, self.COLOR, self.rect)
        pg.draw.rect(scr, self.COLOR, self.rect2)
        if self.rect.right <= 0:
            self.x = xmax
            self.y = random.randrange(self.LENGTH/4 + Wall.HEIGHT, ymax - self.LENGTH * 1.25 - Wall.HEIGHT)

class Helicopter:
    # Helicopter image
    x = xmax / 3
    y = ymax / 2
    vy = 0
    heli_img = pg.image.load("helicopter.png").convert()
    heli_img.set_colorkey(white)


    def __init__(self):
        self.rect = self.heli_img.get_rect()
        self.rect.center = self.x, self.y
        self.passed = False

    def move(self, climb, descend):
        if climb and not descend:
            self.y -= 10
            self.rect.center = self.x, self.y
        elif descend and not climb:
            self.y += 10
            self.rect.center = self.x, self.y
        else:
            pass


    def draw(self, scr):
        scr.blit(self.heli_img, self.rect)

def hitwall(helicopter, ground, ceiling):
    if pg.Rect.colliderect(helicopter.rect, ground.rect) or \
        pg.Rect.colliderect(helicopter.rect, ceiling.rect):
        helicopter.vy = 0
        return True
    return False

def hitobstacle (helicopter, obstacle):
    if pg.Rect.colliderect(helicopter.rect, obstacle.rect) or pg.Rect.colliderect(helicopter.rect, obstacle.rect2):
        helicopter.vy = 0
        return True
    return False



def write_text(txt, x_pos, y_pos, color, scr, font=32):
    font = pg.font.Font('Minecraft.ttf', font)
    text = font.render(txt, True, color)
    Rect = text.get_rect()
    Rect.center = (x_pos, y_pos)
    scr.blit(text, Rect)


def draw_window(scr, ceiling, ground, obstacle, helicopters, score, gen, best):
    # Draw stuff
    scr.fill(black)
    ceiling.draw(scr)
    ground.draw(scr)
    obstacle.draw(scr)
    for helicopter in helicopters:
        helicopter.draw(scr)
    # helicopter.draw(scr)
    write_text(f"Score: {score}", xmax / 10, ymax - Wall.HEIGHT / 2, white, scr, font=32)
    write_text(f"Generation: {gen}", xmax * 9 / 10, ymax - Wall.HEIGHT / 2, white, scr, font=32)
    write_text(f"Alive: {len(helicopters)}", xmax * 2 / 3, ymax - Wall.HEIGHT / 2, white, scr, font=32)
    if len(best) > 0:
        write_text(f"Best: {max(best)}", xmax / 3, ymax - Wall.HEIGHT / 2, white, scr, font=32)


    # Update screen
    pg.display.flip()

def passed(helicopter, obstacle):
    if not helicopter.passed and helicopter.rect.left > obstacle.x:
        helicopter.passed = True
        return True
    return False


def eval_genomes(genomes, config):
    global gen
    gen += 1
    nets = []
    ge = []
    helis = []

    # initialize pygame
    pg.init()

    # Clock
    clock = pg.time.Clock()

    # Create objects
    ceiling = Wall(0)
    ground = Wall(ymax - Wall.HEIGHT)
    obstacle = Obstacle(xmax)

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        helis.append(Helicopter())
        g.fitness = 0
        ge.append(g)

    running = True
    climb = False
    descend = False
    score = 0
    while running:
        score += 1
        # 30 fps
        clock.tick(120)
        dt = clock.get_fps() / 1000

        # Event pump
        pg.event.pump()



        if len(helis) == 0:
            running = False
            best.append(score)

        for x, heli in enumerate(helis):
            heli.move(climb, descend)
            ge[x].fitness += 1

            # Horizontal distance to the obstacle
            horz_dist = obstacle.rect.left - heli.rect.right
            # Vertical distance from top of obstacle to bottom of helicopter
            vert_dist1 = obstacle.rect.top - heli.rect.bottom
            # Vertical distance from bottom of obstacle to top of helicopter
            vert_dist2 = heli.rect.top - obstacle.rect.bottom
            # Vertical distance from bottom of heli to the ground
            ground_dist = np.abs(ground.rect.top - heli.rect.bottom)
            # Vertical distance from top of heli to the ceiling
            ceilingdist = np.abs(heli.rect.top - ceiling.rect.bottom)
            # Space between walls and obstacle
            space1 = np.abs(obstacle.rect.top - ceiling.rect.bottom)
            space2 = np.abs(obstacle.rect.bottom - ground.rect.top)
            # Vertical velocity znd vertical position
            # heli.vy, heli.y

            # 8 inputs
            inputs = [horz_dist, vert_dist1, vert_dist2, ground_dist, ceilingdist, heli.y, space1, space2]
            # inputs = [ground_dist, ceilingdist, heli.rect.centerx, heli.rect.centery, heli.rect.centery]

            # 2 output (climb or descend)
            output = nets[x].activate(inputs)

            climb = False
            descend = False
            if output[0] > 0.5:
                climb = True
            if output[1] > 0.5:
                descend = True

            if (not climb and not descend):
                ge[x].fitness -= 1

            # Increase fitness if heli passed the obstacle
            if passed(heli, obstacle):
                ge[x].fitness += 20
            if obstacle.rect.right <= 0:
                heli.passed = False
            if obstacle.x < heli.x:
                climb = False
                descend = False

            if hitwall(heli, ground, ceiling):
                ge[x].fitness -= 12
                helis.pop(x)
                nets.pop(x)
                ge.pop(x)

            if hitobstacle(heli, obstacle):
                ge[x].fitness -= 10
                helis.pop(x)
                nets.pop(x)
                ge.pop(x)


        # ================================================================================
        # Events
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
                pg.quit()

            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    running = False
                    pg.quit()
        # ================================================================================

        # Draw window
        obstacle.move(dt)
        draw_window(scr, ceiling, ground, obstacle, helis, score, gen, best)





def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # play for x generations
    winner = p.run(eval_genomes, 1000)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)