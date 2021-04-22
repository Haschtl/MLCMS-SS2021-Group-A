#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from math import floor
from collections import deque

from PIL import Image
from PIL import ImageTk
from PIL import ImageDraw

from tkinter import Tk, Button, Label, Frame, Menu, Canvas, NW, LEFT
from tkinter.filedialog import asksaveasfilename, askopenfilename
import tkinter.messagebox
from tkinter import simpledialog


stored_position = (0, 0)  # for rectangle drawing
colour_mapping = [
    ["Target", "yellow", [255, 255, 0], 2],
    ["Obstacle", "purple", [128, 0, 128], -1],
    ["Pedestrian", "red", [255, 0, 0], 1],
    ["Empty", "white", [255, 255, 255], 0]
]


def colour_chosen(draw_window, canvas, colour, text):
    """
    Select drawing color (ped, obstacle, target, empty)
    """
    if canvas.data.image != None:
        canvas.data.drawColour = colour


def colour2feature(color):
    """
    Converts color to the feature it represents
    """
    for mapping in colour_mapping:
        if color == mapping[1]:
            return mapping[0]
    return colour_mapping[-1][0]


def draw_dot(event, canvas, text):
    """
    Draw a dot (leftclick/move)
    """
    global stored_position
    if canvas.data.drawOn == True:
        x = (event.x-canvas.data.imageTopX)*canvas.data.imageScale
        y = (event.y-canvas.data.imageTopY)*canvas.data.imageScale
        stored_position = (x, y)
        draw = ImageDraw.Draw(canvas.data.image)
        draw.point((x, y), fill=canvas.data.drawColour)
        # save(canvas)
        canvas.data.undoQueue.append(canvas.data.image.copy())
        canvas.data.imageForTk = make_image_for_tk(canvas)

        counters = counter(canvas)
        text.configure(text="P: {}, O: {}, T: {}".format(
            counters["pedestrians"], counters["obstacles"], counters["targets"]))

        draw_image(canvas)


def draw_rect(event, canvas, text):
    """
    Draw a rectangle (leftclick + rightclick)
    """
    global stored_position
    if canvas.data.drawOn == True:
        x = (event.x-canvas.data.imageTopX)*canvas.data.imageScale
        y = (event.y-canvas.data.imageTopY)*canvas.data.imageScale
        draw = ImageDraw.Draw(canvas.data.image)
        draw.rectangle([(x, y), stored_position], fill=canvas.data.drawColour)
        # save(canvas)
        canvas.data.undoQueue.append(canvas.data.image.copy())
        canvas.data.imageForTk = make_image_for_tk(canvas)

        counters = counter(canvas)
        text.configure(text="P: {}, O: {}, T: {}".format(
            counters["pedestrians"], counters["obstacles"], counters["targets"]))

        draw_image(canvas)


def write_position(event, canvas, text):
    """
    Write the current cursor position in sidebar
    """
    global stored_position
    if canvas.data.image != None:
        # image_width = canvas.data.image.size[0]
        image_height = canvas.data.image.size[1]
        x = (event.x-canvas.data.imageTopX)*canvas.data.imageScale
        y = image_height-(event.y-canvas.data.imageTopY)*canvas.data.imageScale
        text.configure(text="Position: {}:{}\nSelected: {}\nLast draw: {}:{}".format(
            floor(x), floor(y), colour2feature(canvas.data.drawColour), floor(stored_position[0]), floor(image_height-stored_position[1])))


def reset(canvas):
    """
    Reset the canvas.data.imageForTk
    """
    canvas.data.colourPopToHappen = False
    canvas.data.cropPopToHappen = False
    canvas.data.drawOn = False
    # change back to original image
    if canvas.data.image != None:
        canvas.data.image = canvas.data.originalImage.copy()
        # save(canvas)
        canvas.data.undoQueue.append(canvas.data.image.copy())
        canvas.data.imageForTk = make_image_for_tk(canvas)
        draw_image(canvas)


def key_pressed(canvas, event):
    """
    Handler for key-presses
    """
    if event.keysym == "z":
        undo(canvas)
    elif event.keysym == "y":
        redo(canvas)


def undo(canvas):
    """
    Undo last step
    """
    if len(canvas.data.undoQueue) > 0:
        last_image = canvas.data.undoQueue.pop()
        canvas.data.redoQueue.appendleft(last_image)
    if len(canvas.data.undoQueue) > 0:
        canvas.data.image = canvas.data.undoQueue[-1]
    canvas.data.imageForTk = make_image_for_tk(canvas)
    draw_image(canvas)


def redo(canvas):
    """
    Redo last undone step
    """
    if len(canvas.data.redoQueue) > 0:
        canvas.data.image = canvas.data.redoQueue[0]
    if len(canvas.data.redoQueue) > 0:
        last_image = canvas.data.redoQueue.popleft()
        canvas.data.undoQueue.append(last_image)
    canvas.data.imageForTk = make_image_for_tk(canvas)
    draw_image(canvas)


def new_image(canvas):
    """
    Create a new empty image
    """
    w = simpledialog.askinteger("Scenario Width", "Please enter the width of your scenario",
                                parent=canvas,
                                minvalue=1, maxvalue=100)
    if w is None:
        return
    h = simpledialog.askinteger("Scenario Height", "Please enter the height of your scenario",
                                parent=canvas,
                                minvalue=1, maxvalue=100)
    if h is None:
        return
    im = Image.new(mode="RGB", size=(w, h), color="white")
    canvas.data.image = im
    canvas.data.imageLocation = "new.ppm"
    canvas.data.originalImage = im.copy()
    canvas.data.undoQueue.append(im.copy())
    canvas.data.imageSize = im.size
    canvas.data.imageForTk = make_image_for_tk(canvas)
    draw_image(canvas)


def image_to_array(image):
    """
    Convert a PIL image to numpy array
    """
    image_width = image.size[0]
    image_height = image.size[1]
    array = np.array(image)
    # for mapping in colour_mapping:
    #     array[array==mapping[2]]=mapping[3]
    mapped_array = np.zeros((image_width, image_height))
    for x, row in enumerate(mapped_array):
        for y, cell in enumerate(row):
            for mapping in colour_mapping:
                if np.array_equal(array[x][y], mapping[2]):
                    mapped_array[x][y] = mapping[3]
    mapped_array = np.flip(mapped_array, 1)
    return mapped_array


def array_to_image(array):
    """
    Convert a numpy array to PIL image
    """
    mapped_array = np.zeros((*array.shape, 3))
    for x, row in enumerate(array):
        for y, cell in enumerate(row):
            for mapping in colour_mapping:
                if array[x][y] == mapping[3]:
                    mapped_array[x][y] = mapping[2]

    mapped_array = np.flip(mapped_array, 0)
    return Image.fromarray(np.uint8(mapped_array))


def save_as(canvas):
    """
    Save the scenario as numpy array
    """
    if canvas.data.image != None:
        filename = asksaveasfilename(defaultextension=".npy")
        array = image_to_array(canvas.data.image)
        np.save(filename, array)


def open_image(canvas):
    """
    Open image with filedialog
    """
    image_name = askopenfilename()
    if image_name.endswith(".npy"):
        canvas.data.imageLocation = image_name
        im_data = np.load(image_name)
        im = array_to_image(im_data)
    elif any([image_name.endswith(typ) for typ in ['jpeg', 'bmp', 'png', 'tiff']]):
        canvas.data.imageLocation = image_name
        im = Image.open(image_name)
    else:
        tkinter.messagebox.showinfo(title="Scenario File",
                                    message="Choose an Image/Npy File!", parent=canvas.data.mainWindow)
        return
    canvas.data.image = im
    canvas.data.originalImage = im.copy()
    canvas.data.undoQueue.append(im.copy())
    canvas.data.imageSize = im.size  # Original Image dimensions
    canvas.data.imageForTk = make_image_for_tk(canvas)
    draw_image(canvas)


def counter(canvas):
    """
    Count colored pixels in canvas (peds, obstacles, targets)
    """
    im = canvas.data.image
    counters = {"pedestrians": 0, "obstacles": 0, "targets": 0}
    if im != None:
        for x in range(im.width):
            for y in range(im.height):
                if im.getpixel((x, y)) == (255, 0, 0):
                    counters["pedestrians"] += 1
                elif im.getpixel((x, y)) == (128, 0, 128):
                    counters["obstacles"] += 1
                elif im.getpixel((x, y)) == (255, 255, 0):
                    counters["targets"] += 1
    return counters


def make_image_for_tk(canvas):
    """
    Creates ImageTk for rendering
    """
    im = canvas.data.image
    if im != None:
        image_width = canvas.data.image.size[0]
        image_height = canvas.data.image.size[1]
        if image_width > image_height:
            resized_image = im.resize((canvas.data.width,
                                       int(round(float(image_height)*canvas.data.width/image_width))), Image.NONE)
            canvas.data.imageScale = float(image_width)/canvas.data.width
        else:
            resized_image = im.resize((int(round(float(image_width)*canvas.data.height/image_height)),
                                      canvas.data.height), Image.NONE)
            canvas.data.imageScale = float(image_height)/canvas.data.height
        canvas.data.resizedIm = resized_image
        return ImageTk.PhotoImage(resized_image)


def draw_image(canvas):
    """
    Draw data from canvas.data.imageForTk in canvas
    """
    if canvas.data.image != None:
        # make canvas and image center
        canvas.create_image(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0,
                            canvas.data.height/2.0 -
                            canvas.data.resizedIm.size[1]/2.0,
                            anchor=NW, image=canvas.data.imageForTk)
        canvas.data.imageTopX = int(
            round(canvas.data.width/2.0-canvas.data.resizedIm.size[0]/2.0))
        canvas.data.imageTopY = int(
            round(canvas.data.height/2.0-canvas.data.resizedIm.size[1]/2.0))


def init(root, canvas):
    '''
    Initialize scenario editor 
    '''
    init_buttons(root, canvas)
    init_menu(root, canvas)
    canvas.data.image = None
    canvas.data.angleSelected = None
    canvas.data.rotateWindowClose = False
    canvas.data.brightnessWindowClose = False
    canvas.data.brightnessLevel = None
    canvas.data.histWindowClose = False
    canvas.data.solarizeWindowClose = False
    canvas.data.posterizeWindowClose = False
    canvas.data.colourPopToHappen = False
    canvas.data.cropPopToHappen = False
    canvas.data.endCrop = False
    canvas.data.drawOn = True

    canvas.data.undoQueue = deque([], 10)
    canvas.data.redoQueue = deque([], 10)
    canvas.pack()
    init_image(canvas)


def init_buttons(root, canvas):
    '''
    Initialize sidebar buttons and mouse-clicks
    '''
    background_color = "white"
    button_width = 14
    button_height = 2
    toolkit_frame = Frame(root)
    row_idx = 0
    feature_buttons = []
    text = Label(toolkit_frame, text="P:0, O:0, T:0")
    for _ in colour_mapping:
        colour = colour_mapping[row_idx][1]
        button = Button(toolkit_frame, text=colour_mapping[row_idx][0], bg=colour, width=button_width, height=button_height,
                        command=lambda x=colour: colour_chosen(toolkit_frame, canvas, x, text))
        button.grid(row=row_idx, column=0)
        feature_buttons.append(button)
        row_idx += 1

    label1 = Label(toolkit_frame, text="Leftclick: Draw dots")
    label1.grid(row=row_idx, column=0)

    label2 = Label(toolkit_frame, text="Rightclick: Draw rects")
    label2.grid(row=row_idx+1, column=0)

    text.grid(row=row_idx+2, column=0)

    canvas.data.drawColour = "white"

    text2 = Label(toolkit_frame, text="-")
    text2.grid(row=row_idx+3, column=0)

    reset_button = Button(toolkit_frame, text="Reset",
                          background=background_color, width=button_width,
                          height=button_height, command=lambda: reset(canvas))
    reset_button.grid(row=row_idx+4, column=0)

    canvas.bind("<Motion>", lambda event: write_position(event, canvas, text2))
    canvas.bind("<B1-Motion>",
                lambda event: draw_dot(event, canvas, text))
    canvas.bind("<Button-1>",
                lambda event: draw_dot(event, canvas, text))
    canvas.bind("<Button-2>",
                lambda event: draw_rect(event, canvas, text))
    canvas.bind("<Button-3>",
                lambda event: draw_rect(event, canvas, text))

    toolkit_frame.pack(side=LEFT)


def init_menu(root, canvas):
    """
    Initialize top menu
    """
    menubar = Menu(root)
    menubar.add_command(label="New", command=lambda: new_image(canvas))
    menubar.add_command(label="Open", command=lambda: open_image(canvas))
    menubar.add_command(label="Save", command=lambda: save_as(canvas))
    menubar.add_command(label="Undo (Z)", command=lambda: undo(canvas))
    menubar.add_command(label="Redo (Y)", command=lambda: redo(canvas))
    root.config(menu=menubar)


def init_image(canvas):
    """
    Initialize canvas
    """
    im = Image.new(mode="RGB", size=(50, 50), color="white")
    canvas.data.image = im
    canvas.data.imageLocation = "new.ppm"
    canvas.data.originalImage = im.copy()
    canvas.data.undoQueue.append(im.copy())
    canvas.data.imageSize = im.size
    canvas.data.imageForTk = make_image_for_tk(canvas)
    draw_image(canvas)


def main():
    # create root and canvas
    root = Tk()
    root.title("Scenario Editor")
    canvas_width = 800
    canvas_height = 800
    canvas = Canvas(root, width=canvas_width, height=canvas_height,
                    background="gray")

    # Setup canvas data and init
    class Struct:
        pass
    canvas.data = Struct()
    canvas.data.width = canvas_width
    canvas.data.height = canvas_height
    canvas.data.mainWindow = root
    init(root, canvas)
    root.bind("<Key>", lambda event: key_pressed(canvas, event))
    root.mainloop()


if __name__ == "__main__":
    main()
