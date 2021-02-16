from google_images_download import google_images_download   #importing the library

response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"카스 캔맥주","limit":20,"print_urls":True, "output_directory":"C:/data/"}   #creating list of arguments
paths = response.download(arguments)   #passing the arguments to the function
print(paths)   #printing absolute paths of the downloaded images


# arguments = "format": "jpg" .png 등 확장자만 지정
'''
import os
from PIL import Image

img_dir = r"path/to/downloads/directory"
for filename in os.listdir(img_dir):
    try :
        with Image.open(img_dir + "/" + filename) as im:
             print('ok')
    except :
        print(img_dir + "/" + filename)
        os.remove(img_dir + "/" + filename)'''