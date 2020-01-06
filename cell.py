class cell():
    def __init__(self, x, y, g_cost = 1, h_cost = 1):
        self.x = x
        self.y = y
        self.has_food = False
        self.g_cost = g_cost
        self.h_cost = h_cost
        self.passable = True
        self.parent = None
        self.checked = False
        self.occupied = False

    def set_passable(self, passable):
        self.passable = passable

    def set_blocked_side(self, dir):
        #0 N, 1 E
        self.blocked[dir] = True

    def f_cost(self):
        return self.g_cost + self.h_cost;

    def __repr__(self):
        if self.occupied:
            return 'X'
        else:
            return '0'

    def __str__(self):
        if self.occupied:
            return 'X'
        else:
            return '0'
