import pygame, sys, math, board
from pygame.locals import *
from snake import Snake
from NN import Agent
import numpy as np
import time
from random import seed
from random import randint
seed(1)
pygame.init()

DISPLAYSURF = pygame.display.set_mode((400, 400))
GAME_WIDTH, GAME_HEIGHT = pygame.display.get_surface().get_size()

NUM_CELLS = (5, 5)
WHITE=(255,255,255)
BLUE=(0,0,255)
BLACK=(0,0,0)
SPEED = 50

pygame.display.set_caption('Quiridor')
board = board.board(NUM_CELLS, GAME_HEIGHT, GAME_WIDTH)
snake = Snake(1, WHITE, board.cells[0][0], board)
brain = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, input_dim=[16], learning_rate=0.003)



scores = []
eps_history = []
n_games = 100
score = 0


def get_cell_value(cell):
    if not cell:
        return 1
    if cell.occupied:
        return 2
    elif cell.has_food:
        return 3
    else:
        return 0

def get_state():
    cells = []
    x = snake.pos.x
    y = snake.pos.y

    cells.append(get_cell_value(board.get_cell(x, y - 1)))
    cells.append(get_cell_value(board.get_cell(x, y - 2)))
    cells.append(get_cell_value(board.get_cell(x, y - 3)))
    cells.append(get_cell_value(board.get_cell(x, y - 4)))
    cells.append(get_cell_value(board.get_cell(x, y + 1)))
    cells.append(get_cell_value(board.get_cell(x, y + 2)))
    cells.append(get_cell_value(board.get_cell(x, y + 3)))
    cells.append(get_cell_value(board.get_cell(x, y + 4)))
    cells.append(get_cell_value(board.get_cell(x - 1, y)))
    cells.append(get_cell_value(board.get_cell(x - 2, y)))
    cells.append(get_cell_value(board.get_cell(x - 3, y)))
    cells.append(get_cell_value(board.get_cell(x - 4, y)))
    cells.append(get_cell_value(board.get_cell(x + 1, y)))
    cells.append(get_cell_value(board.get_cell(x + 2, y)))
    cells.append(get_cell_value(board.get_cell(x + 3, y)))
    cells.append(get_cell_value(board.get_cell(x + 4, y)))

    return cells



for n in range(n_games):
    if n % 10 == 0 and n > 0:
        avg_score = np.mean(scores[max(0, n-10):(n+1)])
        print('episode', n, 'score', score, 'average score %.3f' % avg_score,
                'epsilon %.3f' % brain.epsilon)
    else:
        print('episode', n, 'score', score)

    board.reset_board()
    snake.dir = -1
    score = 0
    eps_history.append(brain.epsilon)
    snake.pos = board.get_cell(2,2)
    snake.body = []
    state = get_state()
    reward = 0
    done = False
    board.cells[randint(0, NUM_CELLS[0] - 1)][randint(0, NUM_CELLS[1] - 1)].has_food = True

    while not done: # main game loop
        time.sleep (SPEED / 1000.0);
        reward = 0
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            SPEED -= 50
            SPEED = max(0, SPEED)
        elif keys[pygame.K_DOWN]:
            SPEED += 50

        action = brain.choose_action(state)

        if action == 0:
            snake.dir = 1
        elif action == 1:
            snake.dir = 2
        elif action == 2:
            snake.dir = 3
        elif action == 3:
            snake.dir = 4

        if(snake.dir == 1):
            move_cell = board.get_cell(snake.pos.x, snake.pos.y - 1)
            if move_cell:
                if not snake.move(move_cell):
                    done = True
            else:
                done = True

        elif(snake.dir == 2):
            move_cell = board.get_cell(snake.pos.x + 1, snake.pos.y)
            if move_cell:
                if not snake.move(move_cell):
                    done = True
            else:
                done = True
        elif(snake.dir == 3):
            move_cell = board.get_cell(snake.pos.x, snake.pos.y + 1)
            if move_cell:
                if not snake.move(move_cell):
                    done = True
            else:
                done = True

        elif(snake.dir == 4):
            move_cell = board.get_cell(snake.pos.x - 1, snake.pos.y)
            if move_cell:
                if not snake.move(move_cell):
                    done = True
            else:
                done = True


        if snake.pos.has_food:
            snake.pos.has_food = False
            snake.add_part()
            board.cells[randint(0, NUM_CELLS[0] - 1)][randint(0, NUM_CELLS[1] - 1)].has_food = True
            reward = 1000

        new_state = get_state()
        score += reward
        brain.store_transition(state, action, reward, new_state, done)
        brain.learn()
        state = new_state
        
        if done:
            reward = -100
            scores.append(score)

        DISPLAYSURF.fill(WHITE)
        #print(get_mouse_pos_index())
        board.draw(DISPLAYSURF)
        snake.draw(DISPLAYSURF)
        pygame.display.update()
