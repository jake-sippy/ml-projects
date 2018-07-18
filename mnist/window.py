import tkinter as tk
import skimage.io
import skimage.transform
import numpy as np
import pickle
import sklearn

class App(tk.Frame):
    def __init__(self, master=None, classifier=None):
        super().__init__(master)
        self.classifier = classifier
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        # drawing canvas
        self.canvas = tk.Canvas(self, bg="white", confine=True, height=100, width=100)
        self.canvas.pack(side="left")
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<B1-Motion>", self.on_clicked_move)

        self.buttons = tk.Frame(self)
        self.buttons.pack()

        # Digit estimation
        self.digit = tk.Text(self.buttons)
        self.digit["state"] = "disabled"
        self.digit["height"] = 1
        self.digit["width"] = 1
        self.digit.pack(side="top")

        # Clear button
        self.clear_button = tk.Button(self.buttons)
        self.clear_button["text"] = "Clear\n"
        self.clear_button["command"] = self.clear_screen
        self.clear_button.pack()

        # Quit Button
        self.quit = tk.Button(self.buttons, text="QUIT", fg="red", command=root.destroy)
        self.quit.pack(side="bottom")

    def clear_screen(self):
        self.canvas.delete("all")
        self.digit["state"] = "normal"
        self.digit.delete(1.0, "end")
        self.digit["state"] = "disabled"

    def on_click(self, event):
        self.draw_dot(event.x, event.y)

    def on_clicked_move(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, width=2)
        self.draw_line(self.prev_x, self.prev_y, event.x, event.y)


    def draw_dot(self, x, y):
        x1, y1 = (x - 1), (y - 1)
        x2, y2 = (x + 1), (y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, width=2)
        self.prev_x = x
        self.prev_y = y

    def draw_line(self, x1, y1, x2, y2):
        self.canvas.create_line(x1, y1, x2, y2, width=4)
        self.prev_x = x2
        self.prev_y = y2

    def on_release(self, event):
        # write canvas to file
        self.canvas.postscript(file="tmp_canvas.eps",
                               colormode="gray",
                               width=100,
                               height=100,
                               pagewidth=99,
                               pageheight=99)
        # read the file
        data = skimage.io.imread("tmp_canvas.eps", as_grey=True)
        # transform to match mnist size, range, and shape
        scaled_img = skimage.transform.resize(data, (28, 28), 
                mode="constant")
        scaled_img = (scaled_img*255).astype(dtype="uint8")
        scaled_img = scaled_img.reshape((28*28)).reshape(1, -1)
        scaled_img = skimage.util.invert(scaled_img)
        prediction = self.classifier.predict(scaled_img)
        self.digit["state"] = "normal"
        self.digit.delete(1.0, "end")
        self.digit.insert("end", str(prediction[0]))
        self.digit["state"] = "disabled"



root = tk.Tk()
root.attributes('-type','dialog')
clf = pickle.load(open("clf.save", "rb"))
app = App(master=root, classifier=clf)
app.mainloop()
