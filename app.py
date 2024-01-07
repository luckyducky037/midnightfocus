import ssl

ssl._create_default_https_context = ssl._create_unverified_context

import tkinter as tk
import customtkinter as ctk

import torch
import numpy as np

import cv2
from PIL import Image, ImageTk

import pygame
pygame.mixer.init()
import random

def play_audio(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

app = tk.Tk()
app.geometry("600x480")

app.title("MidnightFocus")
app.configure(bg="black")
ctk.set_appearance_mode("dark")

vidFrame = tk.Frame(height=400, width=400)
vidFrame.pack()
vid = ctk.CTkLabel(vidFrame, text = '')
vid.pack()

counter = 0

counterLabel = ctk.CTkLabel(app, text = str(counter), height = 40, width = 120, font = ("monospace", 20), text_color = "white", fg_color = "red")
counterLabel.pack(pady=10)

def reset_counter():
    global counter
    counter = 0

moyai_path = "/Users/vicki/aiproj/midnightfocus/moyai.png"
moyai_image = Image.open(moyai_path)
moyai_photo = ImageTk.PhotoImage(moyai_image)

image_label = tk.Label(vidFrame, image=moyai_photo, bg="black")
image_label.pack()
image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

def erase_overlay():
    image_label.place_forget()

erase_overlay()    

resetButton = ctk.CTkButton(app, text = "Reset Counter", command = reset_counter, height = 40, width = 120, font = ("monospace", 20), text_color = "white", fg_color = "red")
resetButton.pack()

model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights.pt', force_reload=True)

cap = cv2.VideoCapture(0)
def detect():
    global counter
    ret, frame = cap.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame)
    img = np.squeeze(results.render())
    
    if len(results.xywh[0]) > 0:
        dconf = results.xywh[0][0][4]
        dclass = results.xywh[0][0][5]
        
        if dconf.item() > 0.85 and dclass.item() != 15.0:
            image_label.pack()
            image_label.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
            filechoice = random.choice([1, 2, 3])
            file_path = "/Users/vicki/aiproj/midnightfocus/audio/{}.mp3".format(filechoice)
            play_audio(file_path)
            print("Wake up")
            print("Confidence: " + str(dconf.item()))
            counter += 1
            app.after(1000, erase_overlay)
            
    imgarr = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(imgarr)
    vid.imgtk = imgtk
    vid.configure(image=imgtk)
    vidFrame.after(10, detect)
    counterLabel.configure(text=str(counter))

detect()
app.mainloop()
