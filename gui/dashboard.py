import tkinter as tk
from tkinter import Label
import requests

def update_prediction():
    response = requests.get("http://JETSON_IP:5000/prediction")
    prediction = response.json().get("value", "N/A")
    label.config(text=f"Landmine Detected: {prediction}")
    root.after(1000, update_prediction)

root = tk.Tk()
root.title("Landmine Detector")
label = Label(root, text="Initializing...", font=("Arial", 24))
label.pack()

root.after(1000, update_prediction)
root.mainloop()
