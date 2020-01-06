import pygame, math

class Snake():
    def __init__(self, id, color, pos, board):
        self.id = id
        self.pos = pos
        self.color = color
        self.dir = -1
        self.body = []
        self.board = board

    def move(self, newPos):

        if(newPos.occupied):
            return False

        self.pos.occupied = False
        old_pos = self.pos
        self.pos = newPos
        self.pos.occupied = True

        for part in self.body:
            part.pos.occupied = False
            temp = part.pos
            part.pos = old_pos
            part.pos.occupied = True
            old_pos = temp

        return True


    def add_part(self):
        if len(self.body) > 0:
            last_part = self.body[len(self.body) - 1]
        else:
            last_part = self
        self.body.append(SnakePart(self.board.get_cell(last_part.pos.x, last_part.pos.y)))



    def draw(self, surface):
        return
        pygame.draw.circle(surface, self.color, (self.pos.x * 8 + 4,self.pos.y * 8 + 4), 4)
        for part in self.body:
            part.draw(surface)

class SnakePart():
    def __init__(self, pos):
        self.pos = pos

    def move(self, newPos):
        self.pos = newPos

    def draw(self, surface):
        pass
        #pygame.draw.circle(surface, (255, 0, 0), (self.pos.x * 8 + 4,self.pos.y * 8 + 4), 4)
