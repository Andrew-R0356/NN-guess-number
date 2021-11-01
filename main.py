import tkinter as tk
import torch

from nnModel import Model

GRID_WIDTH = 280
GRID_HEIGHT = 280
SIZE = 10
BLACK = 'black'
WHITE = 'white'

class Space():
    def __init__(self, row, col, canvas):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.canvas = canvas
        self.pixels = [0 for _ in range(28 * 28)]
        
        self.create_space()
        
    def create_space(self):
        for i in range(self.rows):
            for j in range(self.cols):
                x0 = i * SIZE
                y0 = j * SIZE
                x1 = x0 + SIZE
                y1 = y0 + SIZE
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=BLACK, tags='pixel', outline="")
                self.pixels[i * 28 + j] = 0
                
    def draw(self, master):
        self.master = master
        self.master.bind("<B1-Motion>", self.on_motion)
        
    def bbox(self):
        x, y = self.master.winfo_x(), self.master.winfo_y()
        return x, y, x + self.master.winfo_width(), y + self.master.winfo_height()

    def on_motion(self, event):
        bbox = self.bbox()
        if bbox[0] <= event.x <= bbox[2] and bbox[1] <= event.y <= bbox[3]:
            x0 = (event.x // SIZE) * SIZE
            y0 = (event.y // SIZE) * SIZE
            x1 = x0 + SIZE
            y1 = y0 + SIZE
            self.master.create_rectangle(x0, y0, x1, y1, 
                                         fill=WHITE, 
                                         tags='painted',
                                         outline="")
            x = event.x // 10
            y = event.y // 10
            if 0 < y * 28 + x < 28 * 28:
                self.pixels[y * 28 + x] = 1
    
    def guess_number(self):
        labels = list(range(0, 10))
        inputs = torch.Tensor(self.pixels).reshape(1, 1, 28, 28).to('cuda')
        model = torch.load('model.pt').to('cuda')
        model.eval()
        with torch.no_grad():
            pred = model(inputs)
            predicted = labels[pred[0].argmax(0)]
            print('I think the number is {}'.format(predicted))

            
window = tk.Tk()
window.title("Guess number")
window.resizable(False, False)
frame = tk.Frame(window)
frame.pack()
canvas = tk.Canvas(frame, bg=WHITE, 
                width=GRID_WIDTH, 
                height=GRID_HEIGHT, 
                highlightthickness=0)
canvas.pack()

space = Space(28, 28, canvas)
space.draw(canvas)

reset_btn = tk.Button(frame, text='Reset', 
                    font=('consolas', 15), 
                    command=space.create_space,
                    width=12,
                    height=1)
reset_btn.pack(side=tk.LEFT)

guess_btn = tk.Button(frame, text='Guess', 
                    font=('consolas', 15), 
                    command=space.guess_number,
                    width=12,
                    height=1)
guess_btn.pack(side=tk.RIGHT)

window.mainloop()














