import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image
import threading
import os

ctk.set_appearance_mode("dark")

class App(ctk.CTk):
    #variables
    file = 0
    files = []
    width = 800
    height = 500

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

        # frames
        self.side_frame = ctk.CTkFrame(master=self, corner_radius=0)
        self.side_frame.place(x=0, y=0, relwidth=0.2, relheight=1)
        self.side_frame.columnconfigure((0,1,2), weight=1, uniform=True)
        self.side_frame.rowconfigure((0,1,2,3), weight=1, uniform=False)

        self.image_frame = ctk.CTkFrame(master=self, fg_color="transparent")
        self.image_frame.place(relx=0.2, y=0, relwidth=0.8, relheight=1)
        
        # side frame widgets
        self.title_label = ctk.CTkLabel(master=self.side_frame, text="AI Image Detector")
        self.title_label.grid(row=0, column=0, columnspan=3, sticky="nwes", padx = 5, pady = (10, 0))

        self.upload_btn = ctk.CTkButton(master=self.side_frame, text="UPLOAD", image=self.upload_img, command=self.get_images)
        self.upload_btn.grid(column=0, row=1, columnspan=3, sticky="nwe", padx=20)

        self.file_counter_label = ctk.CTkLabel(master=self.side_frame, text="Files: 0")
        self.file_counter_label.grid(row=2, column=0, columnspan=3, sticky="nw", padx = 10)

        # image frame widgets
        self.image_label = ctk.CTkLabel(master=self.image_frame, text="")
        self.image_label.place(anchor="center", relx=0.5, rely=0.3, relheight = 0.5, relwidth = 0.5)

        self.file_name = ctk.CTkLabel(master=self.image_frame, text="No File", font=("Arial", 20))
        self.file_name.place(anchor="center", relx=0.5, rely=0.65)

        self.ai_label = ctk.CTkLabel(master=self.image_frame, text="...", text_color="blue")
        self.ai_label.place(anchor="center", relx=0.5, rely=0.7)

        self.next_btn = ctk.CTkButton(master=self.image_frame, text="Next", command=self.next_file)
        self.next_btn.place(anchor="center", relx=0.5, rely=0.8, relwidth=0.2, relheight=0.1)

    # functions
    def get_images(self):
        self.file = 0
        self.files = filedialog.askopenfiles(filetypes=[("Image Files", "*.jpg *.jpeg *.pgm")])
        print(self.files[0])
        self.file_name.configure(text=f"{self.files[self.file].name}")
        self.file_counter_label.configure(text=f"Files: {len(self.files)}")
        threading.Thread(target=self.process_image, args=(self.files[self.file].name,)).start()

    def next_file(self):
        if self.file < len(self.files)-1:
            self.file += 1
            self.file_name.configure(text=f"{self.files[self.file].name}")
            self.ai_label.configure(text="...", text_color="blue")
            self.file_counter_label.configure(text=f"Files: {len(self.files) - self.file}")
            threading.Thread(target=self.process_image, args=(self.files[self.file].name,)).start()
        else:
            self.files = []
            self.file = 0
            self.file_name.configure(text="No File")
            self.file_counter_label.configure(text=f"Files: {len(self.files)}")
            self.image_label.configure(image=None)

    def process_image(self, file):
        image = Image.open(file)
        image.resize((256,256))
        self.image_label.configure(image=ctk.CTkImage(image,size=(self.image_label.winfo_width(),self.image_label.winfo_width())))

# Run application
if __name__ == "__main__":
    # create app
    app = App()

    # on resize
    def resize(event):
        app.title_label.configure(font=("Arial", app.winfo_height() * 0.03))
        app.file_counter_label.configure(font=("Arial", app.winfo_height() * 0.03))
        app.ai_label.configure(font=("Arial", app.winfo_height() * 0.05))
        app.upload_btn.grid(ipady=app.winfo_height() * 0.05)

    app.side_frame.bind('<Configure>', resize)

    # Run application
    app.mainloop()