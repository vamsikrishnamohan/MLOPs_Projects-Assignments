import argparse
import requests
import numpy as np
from PIL import Image,ImageDraw
import tkinter as tk
from tkinter import messagebox

# Function to predict the digit using the API
def predict(image_vector, api_url):
    # Convert the image vector to bytes
    image_bytes = (image_vector * 255).astype(np.uint8).tobytes()
    
    # Send the image to the FastAPI endpoint
    files = {"file": image_bytes}
    response = requests.post(api_url, files=files)
    
    # Display the prediction result
    if response.status_code == 200:
        result = response.json().get("predicted_digit")
        messagebox.showinfo("Result", f"Digit: {result}")
    else:
        messagebox.showerror("Error", "Failed to get prediction from the API.")

# Drawing application class
class DrawingApp:
    def __init__(self, root, api_url):
        self.root = root
        self.api_url = api_url
        self.root.title("My Drawing Canvas 28x28")

        # Canvas settings
        self.canvas_size = 680  # Canvas size in pixels
        self.image_size = 28  # Image size for vectorization
        self.brush_size = 20  # Size of the white brush

        # Canvas for drawing
        self.canvas = tk.Canvas(root, bg="black", width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()

        # Creation of the image and the object for drawing
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        # Action buttons
        self.button_frame = tk.Frame(root)
        self.button_frame.pack()
        
        self.predict_button = tk.Button(self.button_frame, text="  Predict The Digit  ", command=self.predict_image)
        self.predict_button.pack(side="left")

        self.clear_button = tk.Button(self.button_frame, text="  Erase  ", command=self.clear_canvas)
        self.clear_button.pack(side="right")

        # Drawing event
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        # Draw on the screen and on the image
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        
        # Draw on the canvas (screen) with a white brush
        self.canvas.create_oval(x1, y1, x2, y2, fill="yellow", outline="yellow")

        # Draw on the 28x28 image for vectorization
        scaled_x1, scaled_y1 = (x1 * self.image_size // self.canvas_size), (y1 * self.image_size // self.canvas_size)
        scaled_x2, scaled_y2 = (x2 * self.image_size // self.canvas_size), (y2 * self.image_size // self.canvas_size)
        self.draw.ellipse([scaled_x1, scaled_y1, scaled_x2, scaled_y2], fill="yellow")

    def predict_image(self):
        # Convert the image to a vector and normalize the values (0 to 1)
        image_data = np.array(self.image).reshape(1, -1) / 255.0
        predict(image_data, self.api_url)

    def clear_canvas(self):
        # Clears the canvas and creates a new black image
        self.canvas.delete("all")
        self.image = Image.new("L", (self.image_size, self.image_size), "black")
        self.draw = ImageDraw.Draw(self.image)

# Parse command-line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Handwritten Digit Classifier UI")
    parser.add_argument("--api_url", type=str, required=True, help="URL of the FastAPI prediction endpoint")
    return parser.parse_args()

# Application initialization
if __name__ == "__main__":
    args = parse_arguments()
    root = tk.Tk()
    root.tk.call('tk','scaling',4.0)
    app = DrawingApp(root, args.api_url)
    root.mainloop()