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
        LS_font = current_path+"/LeagueSpartan-Regular.ttf"
        M_font = current_path+"/Montserrat-Regular.ttf"

        # background image
        self.bg_image_label = ctk.CTkLabel(master=self, image=self.bg_img, text="")
        self.bg_image_label.grid(column=0,row=0)

        # title page
        self.welcome_label = ctk.CTkLabel(master=self, text="WELCOME TO TKINTER", fg_color="transparent")
        self.welcome_label.configure(image=self.transparent(0, 0, self.welcome_label, text=True, font=M_font))
        self.welcome_label.place(x=0,y=0)

    # helper functions
    def transparent(self, x, y, widget, text=False, font=None, color=(255,255,255), font_size = 22):
        self.update_idletasks()
        width = widget.winfo_reqwidth()
        height = widget.winfo_reqheight()
        image = self.bg_image_pil.crop((x, y, x+(width*1), y+(height*1))).convert("RGBA")

        if text:
            fill = color
            text = widget.cget("text")
            font = font
            font_size = font_size
            font = ImageFont.truetype(font=font, size=font_size-2)
            ImageDraw.Draw(image).text(xy=(5, 5), text=text, font=font, fill=fill, anchor='lt')
            image = image.resize((width, height), Image.LANCZOS)
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