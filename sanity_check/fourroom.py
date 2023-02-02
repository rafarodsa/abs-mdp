from ssl import VerifyFlags
from tkinter import VERTICAL
from visgrid.gridworld import GridWorld
import matplotlib.pyplot as plt

class FourRoom(GridWorld):
    def __init__(self, rows=5, cols=5):
        super().__init__(rows, cols)
        bottom, top = cols//2, cols//2
        left, right = rows//2, rows//2 + 1
        x_o, y_o = 0, 0
        self._horizontal_doors = []
        self._vertical_doors = []
        for horizontal_wall in [(bottom, left), (cols, right)]:
            x_max = horizontal_wall[0]
            for x in range(x_o, x_max):
                row = horizontal_wall[1] * 2
                col = x * 2 + 1
                self._grid[row, col] = 1
            door_row = horizontal_wall[1] * 2
            door_col = ((x_o+x_max)//2)*2 + 1
            self._grid[door_row, door_col] = 0
            self._horizontal_doors.append((horizontal_wall[1], (x_o+x_max)//2))
            x_o += horizontal_wall[0]
            
        
        for vertical_wall in [(bottom, left), (top, rows)]:
            y_max = vertical_wall[1]
            for y in range(y_o, y_max):
                row = y * 2 + 1
                col = vertical_wall[0] * 2
                self._grid[row, col] = 1
            door_row = ((y_o + y_max)//2)*2 + 1
            door_col = vertical_wall[0] * 2
            self._vertical_doors.append(((y_o + y_max)//2, vertical_wall[0]))
            self._grid[door_row, door_col] = 0.
            y_o += vertical_wall[1]

    def get_doors(self):
        return self._horizontal_doors, self._vertical_doors


if __name__=="__main__":

    env = FourRoom(5,5)
    pl = env.plot()
    plt.show()