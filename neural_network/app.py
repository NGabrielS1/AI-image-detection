import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image

# Parent window
app = ctk.CTk()

# Window settings
app.title("AI Image Detector")
app.geometry("800x600")
ctk.set_appearance_mode("dark")

#variables
file = 0

# function
def get_images():
    global file, files
    file = 0
    files = filedialog.askopenfiles()
    file_name.configure(text=f"{files[file].name}")
    file_counter_label.configure(text=f"Files: {len(files)}")

def next_file():
    global file, files
    if file < len(files)-1:
        file += 1
        file_name.configure(text=f"{files[file].name}")
        ai_label.configure(text="...", text_color="blue")
        file_counter_label.configure(text=f"Files: {len(files) - file}")
    else:
        files = []
        file = 0
        file_name.configure(text="No File")
        file_counter_label.configure(text=f"Files: {len(files)}")


# images
upload_img = ctk.CTkImage(Image.open("neural_network/upload-button.png").resize((1000,1000), Image.Resampling.LANCZOS))

# frame
side_frame = ctk.CTkFrame(master=app, corner_radius=0)
side_frame.place(x=0, y=0, relwidth=0.2, relheight=1)
side_frame.columnconfigure((0,1,2), weight=1, uniform=True)
side_frame.rowconfigure((0,1,2,3), weight=1, uniform=False)

image_frame = ctk.CTkFrame(master=app, fg_color="transparent")
image_frame.place(relx=0.2, y=0, relwidth=0.8, relheight=1)
 
# side frame widgets
title_label = ctk.CTkLabel(master=side_frame, text="AI Image Detector")
title_label.grid(row=0, column=0, columnspan=3, sticky="nwes", padx = 5, pady = (10, 0))

upload_btn = ctk.CTkButton(master=side_frame, text="UPLOAD", image=upload_img, command=get_images)
upload_btn.grid(column=0, row=1, columnspan=3, sticky="nwe", padx=20)

file_counter_label = ctk.CTkLabel(master=side_frame, text="Files: 0")
file_counter_label.grid(row=2, column=0, columnspan=3, sticky="nw", padx = 10)

# image frame widgets
file_name = ctk.CTkLabel(master=image_frame, text="No File")
file_name.place(anchor="center", relx=0.5, rely=0.65)

ai_label = ctk.CTkLabel(master=image_frame, text="...", text_color="blue")
ai_label.place(anchor="center", relx=0.5, rely=0.7)

next_button = ctk.CTkButton(master=image_frame, text="Next", command=next_file)
next_button.place(anchor="center", relx=0.5, rely=0.8, relwidth=0.2, relheight=0.1)

# on resize
def resize(event):
    title_label.configure(font=("Arial", app.winfo_height() * 0.03))
    file_counter_label.configure(font=("Arial", app.winfo_height() * 0.03))
    file_name.configure(font=("Arial", app.winfo_height() * 0.05))
    ai_label.configure(font=("Arial", app.winfo_height() * 0.05))
    upload_btn.grid(ipady=app.winfo_height() * 0.05)

side_frame.bind('<Configure>', resize)

# Run application
app.mainloop()