import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2

screen_size = (512, 512)

image = Image.new("L", screen_size, 0)  
draw = ImageDraw.Draw(image) 

last_x, last_y = None, None

def draw_line(event):
    global last_x, last_y
    canvas.create_line(last_x, last_y, event.x, event.y, fill="white", width=15)
    
    draw.line([last_x, last_y, event.x, event.y], fill=255, width=15)  
    
    last_x, last_y = event.x, event.y

def set_last_xy(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

def clear_canvas():
    global image, draw
    canvas.delete("all")
    canvas.create_rectangle(0, 0, screen_size[0], screen_size[1], fill="black")
    
    image = Image.new("L", screen_size, 0)
    draw = ImageDraw.Draw(image) 

def save_image():
    np_img = np.array(image)
    img = process_single_digit(np_img)
    
    img = Image.fromarray(img)
    img.save("drawing.png")
    print("Image saved as 'drawing.png'")

def process_single_digit(image):
    resized_digit = cv2.resize(image, (28, 28))
    return resized_digit

root = tk.Tk()
root.title("Drawing Numbers")

canvas = tk.Canvas(root, width=screen_size[0], height=screen_size[1], bg="black")
canvas.pack()

canvas.bind("<Button-1>", set_last_xy)
canvas.bind("<B1-Motion>", draw_line)

btn_clear = tk.Button(root, text="Clear", command=clear_canvas)
btn_clear.pack(side=tk.LEFT, padx=10, pady=10)

btn_save = tk.Button(root, text="Save", command=save_image)
btn_save.pack(side=tk.RIGHT, padx=10, pady=10)

clear_canvas()

root.mainloop()
