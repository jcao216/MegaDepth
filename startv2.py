import os
from PIL import Image



new_dir = "final_frames/"
prod_en = 0  ##enables the demo.py file to be executed in a loop
cut_en = 0   ## enables cropping of files within a directory to be cropped to 384 by 384

try:
    os.mkdir(new_dir)
except OSError:
    print("Directory final frames could not be created or already has been created!")

""" os.rename('interpolated_frames/'+"00001000.png", 'demo.jpg')  ## take the interpolated frame out of its own directory and rename it to demo.jpg so that demo.py does not need to handle different filenames each time
os.system("python demo.py")
os.rename('demo.jpg','interpolated_frames/'+"00001000.png") """

if(prod_en):
    file_ct = 1
    for filer in os.listdir("interpolated_frames/"):
        print(">>> Generating depth frame #{}. <<<".format(file_ct))
        os.rename('interpolated_frames/'+filer, 'demo.jpg')  ## take the interpolated frame out of its own directory and rename it to demo.jpg so that demo.py does not need to handle different filenames each time
        os.system("python demo.py")
        os.rename('demo.jpg','interpolated_frames/'+filer)  ## put original interpolated frame back
        os.rename('demo.png','final_frames/'+'interpolated_frame'+str(file_ct)+'.png')  ##put file into final_frames directory
        file_ct += 1
    """     print(">>> Autoencoding depth frame #{}. Please hold. <<<".format(file_ct))
        autoencode("demo.png",filer)
        print(">>> Finished processing depth frame #{}. <<<".format(file_ct)) """

if(cut_en):
    try:
        os.mkdir("final_frames/cropped")
    except OSError:
        print("Directory final_frames/cropped could not be created or already has been created!")
    file_ct = 1
    for filer in os.listdir(new_dir):
        print(">>>Trimming depth image to 384 by 384 <<<")
        img_uncut = Image.open(new_dir + filer)
        # w,h = img_uncut.size
        img_cut = img_uncut.crop((0,0,384,384))
        img_cut.save("final_frames/cropped/trimmed_" + filer) ## save file
        print(">> Cropped image #{}/{}. <<".format(file_ct,os.listdir(new_dir)))
        file_ct += 1