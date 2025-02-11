import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageFont, ImageDraw
import threading
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from torchvision.models import ResNet34_Weights
import torch
import torch.nn as nn
import torch.nn.functional as F

ctk.set_appearance_mode("dark")

# find device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset class
class CreateDataset(Dataset):
    def __init__(self,imageFolderDataset):
        self.imageFolderDataset = imageFolderDataset
        # data transformation   
        self.transform = transforms.Compose([
            transforms.Resize((100,100)),
            transforms.ToTensor()
        ])
        
    def __getitem__(self,index):
        img_tuple = self.imageFolderDataset.imgs[index]

        img = Image.open(img_tuple[0])
        img = self.transform(img)
       
        return img, img_tuple[1]
    
    # len function
    def __len__(self):
        return len(self.imageFolderDataset.imgs)

# Model Class
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # load ResNet34(transfer learning)
        self.resnet34 = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        self.resnet34.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    
    def forward_once(self, X):
        y = self.resnet34(X)

        return y

    def forward(self, X1, X2):
        y1 = self.forward_once(X1)
        y2 = self.forward_once(X2)

        return y1, y2

# Main app
class App(ctk.CTk):
    #variables
    file = 0
    files = []
    width = 800
    height = 600
    analyzing = False
    model = SiameseNetwork().to(device)
    model.load_state_dict(torch.load(("AI_DETECTOR_SIAMESE.pt"),map_location=torch.device('cpu')))
    transform = transforms.Compose([
            transforms.Resize((100,100)),
            transforms.ToTensor()
    ])

    def __init__(self):
        super().__init__()

        # window settings
        self.title("AI Image Detector")
        self.geometry(f"{self.width}x{self.height}")
        self.resizable(0, 0)

        # widgets
        current_path = os.path.dirname(os.path.realpath(__file__))
        #dataset
        dataset = CreateDataset(datasets.ImageFolder(root=current_path+"/app_data/"))
        self.app_data = DataLoader(dataset, shuffle=False, batch_size=1)
        # images
        self.upload_img = ctk.CTkImage(Image.open(current_path+"/upload-button.png"), size=(300, 75))
        self.bg_image_pil = Image.open(current_path+"/background.png")
        self.bg_img = ctk.CTkImage(self.bg_image_pil,size=(self.width, self.height))
        self.page_img_pil = Image.open(current_path+"/page.png")
        self.page_img = ctk.CTkImage(self.page_img_pil, size=(self.width, self.height))
        self.placeholder_image = ctk.CTkImage(self.page_img_pil.crop((256.5, 135, 256.5+287, 135+287)).convert("RGBA"), size=(287,287))
        next_img = self.page_img_pil.crop((300, 450, 300+200, 450+60)).convert("RGBA")
        next_img = Image.alpha_composite(next_img, Image.open(current_path+"/next.png").convert("RGBA"))
        self.next_img = ctk.CTkImage(next_img, size=(200,60))

        # fonts
        self.LS_font = current_path+"/LeagueSpartan-Bold.ttf"
        self.M_font = current_path+"/Montserrat-Medium.ttf"

        # background image
        self.bg_image_label = ctk.CTkLabel(master=self, image=self.bg_img, text="")
        self.bg_image_label.grid(column=0,row=0)

        # title page
        self.welcome_label = ctk.CTkLabel(master=self)
        self.welcome_label.configure(image=self.transparent(55, 90, widget=self.welcome_label, text_func=True, text="WELCOME TO", font=self.M_font))
        self.welcome_label.place(x=55,y=90)

        self.title_label = ctk.CTkLabel(master=self)
        self.title_label.configure(image=self.transparent(55, 122, widget=self.title_label, text_func=True, text="AI IMAGE \nDETECTOR", font=self.LS_font, font_size=98))
        self.title_label.place(x=55,y=122)

        self.start_btn = ctk.CTkButton(master=self, width=200, height=50, text="", corner_radius=50, fg_color="#0fb568", hover=False, bg_color="black", command=self.next_page, 
            image=self.transparent(55, 450, text_func=True, text="START NOW", font=self.M_font, font_size=17, bg_color=(15, 181, 104),pad=(17,0), color=(0,0,0)))
        self.start_btn.place(x=55,y=450)

        # ai page
        self.footer = ctk.CTkFrame(self, width=self.width, height=75, fg_color="#171717")
        self.footer.pack_propagate(False)
        self.footer.columnconfigure((0,1), weight=1, uniform=True)
        self.footer.rowconfigure(0, weight=1)

        self.upload_btn = ctk.CTkButton(self.footer, height=75, width=350, image=self.upload_img, text="", fg_color="#171717", corner_radius=0, anchor="center", command=self.get_images)

        self.status_label1 = ctk.CTkLabel(self.footer, height=75, text="", fg_color="transparent", corner_radius=0)
        self.status_label1.configure(image=self.transparent(0,0, text_func=True, text="STATUS", font=self.M_font, font_size=30, bg_color=(23, 23, 23)))

        self.image_label = ctk.CTkLabel(self, height=287, width=287, text="", image=self.placeholder_image)

        self.next_btn = ctk.CTkLabel(self, height=60, width=200, image=self.next_img, text="", fg_color="transparent", corner_radius=0)

        self.title_label2 = ctk.CTkLabel(master=self)
        self.title_label2.configure(image=self.transparent(120, 60, widget=self.title_label2, text_func=True, text="AI IMAGE \nDETECTOR", font=self.LS_font, font_size=25))

        self.counter_label1 = ctk.CTkLabel(self, text="", height=25,
            image=self.transparent(450, 90, text_func=True, text="FILES DETECTED: ", font=self.M_font, font_size=20))

        self.counter_label2 = ctk.CTkLabel(self, text="", height=25,
            image=self.transparent(640, 90, text_func=True, text="0", font=self.M_font, font_size=20, color=(255,0,0)))

    # helper functions
    def transparent(self, x, y, widget=None, text_func=False, text=None, font=None, color=None, font_size=None, width=None, height=None, bg_color=None, pad=None, anchor=None):
        if color == None:
            color = (255,255,255)
        if font_size == None:
            font_size = 22
        if pad == None:
            pad = (0,0)
        if anchor == None:
            anchor="lt"
        
        width=width
        height=height

        # draw text
        if text_func:
            width = len(text)*font_size
            if height == None:
                height = font_size+5
            fill = color
            text = text
            font = font
            font_size = font_size
            font = ImageFont.truetype(font=font, size=font_size)
            # multiline
            text = text.split("\n")
            if len(text) > 1:
                lengths =[]
                for line in text:
                    lengths.append(len(line))
                width = max(lengths)*font_size*2/3
            height *= len(text)
            if bg_color is None:
                image = self.bg_image_pil.crop((x, y, x+width, y+height)).convert("RGBA")
            else:
                image = Image.new(mode="RGBA", size=(width, height), color=bg_color)
            draw = ImageDraw.Draw(image)
            draw.fontmode = "L"
            for i, line in enumerate(text):
                draw.text(xy=(5+pad[0], 5+pad[1] + (font_size*i)), text=line, font=font, fill=fill, anchor=anchor)
            if widget != None:
                widget.configure(text="")

        image = ctk.CTkImage(image, size=(width,height))
        return image

    # image functions
    def get_images(self):
        if not self.analyzing:
            self.file = 0
            self.files = filedialog.askopenfiles(filetypes=[("Image Files", "*.jpg *.jpeg *.pgm")])
            print(self.files[0])
            self.status_label1.configure(image=self.transparent(0,0, text_func=True, text="...", color=(224, 255, 255), font=self.M_font, font_size=30, bg_color=(23, 23, 23), height=10))
            self.counter_label2.configure(image=self.transparent(630, 90, text_func=True, text=f"{len(self.files)}", font=self.M_font, font_size=20, color=(255,0,0)))
            threading.Thread(target=self.process_image, args=(self.files[self.file].name,)).start()

    def next_file(self, event=None):
        if not self.analyzing:
            if self.file < len(self.files)-1:
                self.file += 1
                self.status_label1.configure(image=self.transparent(0,0, text_func=True, text="...", color=(224, 255, 255), font=self.M_font, font_size=30, bg_color=(23, 23, 23), height=10))
                self.counter_label2.configure(image=self.transparent(630, 90, text_func=True, text=f"{len(self.files)-self.file}", font=self.M_font, font_size=20, color=(255,0,0)))
                threading.Thread(target=self.process_image, args=(self.files[self.file].name,)).start()
            else:
                self.files = []
                self.file = 0
                self.counter_label2.configure(image=self.transparent(630, 90, text_func=True, text=f"{len(self.files)}", font=self.M_font, font_size=20, color=(255,0,0)))
                self.image_label.configure(image=self.placeholder_image)
                self.status_label1.configure(image=self.transparent(0,0, text_func=True, text="STATUS", font=self.M_font, font_size=30, bg_color=(23, 23, 23)))

    def process_image(self, file):
        status = ""
        image = Image.open(file)
        image.resize((287,287))
        self.image_label.configure(image=ctk.CTkImage(image,size=(self.image_label.winfo_width(),self.image_label.winfo_width())))
        self.analyzing = True

        # prepare image
        X0 = self.transform(image)
        X0 = X0.unsqueeze(0)

        # get predictions
        self.model.eval()
        with torch.no_grad():
            for X1, img_class in self.app_data:
                X0, X1 = X0.to(device), X1.to(device)
                output1, output2 = self.model(X0, X1)
                euclidean_distance = F.pairwise_distance(output1, output2)
                print(euclidean_distance.item())
                if euclidean_distance.item() < 1:
                    status = "AI"
                    break
                else:
                    status = "REAL"
        if status == "AI":
            self.status_label1.configure(image=self.transparent(0,0, text_func=True, text="AI", color=(255,0,0), font=self.M_font, font_size=30, bg_color=(23, 23, 23)))
        else:
            self.status_label1.configure(image=self.transparent(0,0, text_func=True, text="HUMAN", color=(0,255,0), font=self.M_font, font_size=30, bg_color=(23, 23, 23)))
        print(status)
        
        self.analyzing = False

    def next_page(self):
        self.bg_image_label.configure(image=self.page_img)

        self.welcome_label.destroy()
        self.title_label.destroy()
        self.start_btn.destroy()

        self.footer.place(x=0, y=self.height-75)
        self.upload_btn.pack(side="left", fill="both")
        self.status_label1.pack(side="right", fill="both")
        self.image_label.place(x=260,y=135)
        self.next_btn.place(x=300,y=450)
        self.title_label2.place(x=120,y=60)
        self.counter_label1.place(x=450, y=90)
        self.counter_label2.place(x=640, y=90)

# Run application
if __name__ == "__main__":
    # create app
    app = App()

    # custom button
    app.next_btn.bind("<ButtonPress-1>", app.next_file)

    # Run application
    app.mainloop()