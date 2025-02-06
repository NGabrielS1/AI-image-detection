import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageFont, ImageDraw
import threading
import sys
import os
import shutil
from typing import Union
import numpy as np

ctk.set_appearance_mode("dark")

class App(ctk.CTk):
    #variables
    file = 0
    files = []
    width = 800
    height = 600

    def __init__(self):
        super().__init__()

        # window settings
        self.title("AI Image Detector")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(0, 0)

        # widgets
        # images
        current_path = os.path.dirname(os.path.realpath(__file__))
        self.upload_img = ctk.CTkImage(Image.open(current_path+"/upload-button.png"))
        self.bg_image_pil = Image.open(current_path+"/background.png")
        self.bg_img = ctk.CTkImage(self.bg_image_pil,size=(self.width, self.height))
        # fonts
        LS_font = current_path+"/LeagueSpartan-Bold.ttf"
        M_font = current_path+"/Montserrat-Medium.ttf"

        # background image
        self.bg_image_label = ctk.CTkLabel(master=self, image=self.bg_img, text="")
        self.bg_image_label.grid(column=0,row=0)

        # title page
        self.welcome_label = ctk.CTkLabel(master=self)
        self.welcome_label.configure(image=self.transparent(55, 90, widget=self.welcome_label, text_func=True, text="WELCOME TO", font=M_font))
        self.welcome_label.place(x=55,y=90)

        self.title_label = ctk.CTkLabel(master=self)
        self.title_label.configure(image=self.transparent(55, 122, widget=self.title_label, text_func=True, text="AI IMAGE \nDETECTOR", font=LS_font, font_size=98))
        self.title_label.place(x=55,y=122)

        self.start_btn = ctk.CTkButton(master=self, width=200, height=50, text="", corner_radius=50, fg_color="#0fb568", hover_color="#0fb568", bg_color="black", text_color="black", font=("Free Sans", 20), \
            image=self.transparent(55, 450, text_func=True, text="START NOW", font=M_font, font_size=17, bg_color=(15, 181, 104),pad=(17,0), color=(0,0,0)))
        self.start_btn.place(x=55,y=450)

        # self.welcome_label.destroy()
        # self.title_label.destroy()
        # self.start_btn.destroy()

    # helper functions
    def transparent(self, x, y, widget=None, text_func=False, text=None, font=None, color=None, font_size=None, width=None, height=None, bg_color=None, pad=None):
        if color == None:
            color = (255,255,255)
        if font_size == None:
            font_size = 22
        if pad == None:
            pad = (0,0)
        
        width=width
        height=height

        # draw text
        if text_func:
            width = len(text)*font_size
            height = font_size+5
            fill = color
            text = text
            font = font
            font_size = font_size
            font = ImageFont.truetype(font=font, size=font_size)
            # multiline
            text = text.split("\n")
            height *= len(text)
            if bg_color is None:
                image = self.bg_image_pil.crop((x, y, x+width, y+height)).convert("RGBA")
            else:
                image = Image.new(mode="RGBA", size=(width, height), color=bg_color)
            draw = ImageDraw.Draw(image)
            draw.fontmode = "L"
            for i, line in enumerate(text):
                draw.text(xy=(5+pad[0], 5+pad[1] + (font_size*i)), text=line, font=font, fill=fill, anchor="lt")
            if widget != None:
                widget.configure(text="")

        image = ctk.CTkImage(image, size=(width,height))
        return image

    # def get_images(self):
    #     self.file = 0
    #     self.files = filedialog.askopenfiles(filetypes=[("Image Files", "*.jpg *.jpeg *.pgm")])
    #     print(self.files[0])
    #     self.file_name.configure(text=f"{self.files[self.file].name}")
    #     self.file_counter_label.configure(text=f"Files: {len(self.files)}")
    #     threading.Thread(target=self.process_image, args=(self.files[self.file].name,)).start()

    # def next_file(self):
    #     if self.file < len(self.files)-1:
    #         self.file += 1
    #         self.file_name.configure(text=f"{self.files[self.file].name}")
    #         self.ai_label.configure(text="...", text_color="blue")
    #         self.file_counter_label.configure(text=f"Files: {len(self.files) - self.file}")
    #         threading.Thread(target=self.process_image, args=(self.files[self.file].name,)).start()
    #     else:
    #         self.files = []
    #         self.file = 0
    #         self.file_name.configure(text="No File")
    #         self.file_counter_label.configure(text=f"Files: {len(self.files)}")
    #         self.image_label.configure(image=None)

    # def process_image(self, file):
    #     image = Image.open(file)
    #     image.resize((256,256))
    #     self.image_label.configure(image=ctk.CTkImage(image,size=(self.image_label.winfo_width(),self.image_label.winfo_width())))

# Run application
if __name__ == "__main__":
    # create app
    app = App()

    # # on resize
    # def resize(event):
    #     app.title_label.configure(font=("Arial", app.winfo_height() * 0.03))
    #     app.file_counter_label.configure(font=("Arial", app.winfo_height() * 0.03))
    #     app.ai_label.configure(font=("Arial", app.winfo_height() * 0.05))
    #     app.upload_btn.grid(ipady=app.winfo_height() * 0.05)

    # app.bind('<Configure>', resize)

    # Run application
    app.mainloop()