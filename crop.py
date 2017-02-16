from PIL import Image
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--namebase", required=True,
	help="names of images in folder mus be like /namebase/smth,/namebase/smth etc ")
ap.add_argument("-p","--path",required=False,help="path to folder containing images")
#ap.add_argument("-c","--count",required=True,help="number of images in folder")
#ap.add_argument("-f","--format",required=False,help="format of images (only this format will bee proceeded) ")
ap.add_argument("-s","--savepath",required=False,help="path where to save cropped images (specify if it is needed to save)")
args = vars(ap.parse_args())


#print(args)
if args["path"] is None:
	args["path"] = "./"
if args["path"][0:4] != "home":
	args["path"] = "./" + args["path"]

if args["savepath"] is not None:

	if args["savepath"][0:4] != "home":
		args["savepath"] = "./" + args["savepath"]

#if args["format"] is None:
#	args["format"] = "JPG"


#for i in range(0,int(args["count"])):
#	Namefil = args["path"] + args["namebase"]+str(i)+"."+args["format"]
#	if not os.path.isfile(Namefil):
#		print ("No file "+Namefil)
#		continue
k=0

print("Working with",args["path"])

for filename in os.listdir(args["path"]):
    if filename.startswith(args["namebase"]): 
        #print(os.path.join(directory, filename))
        im = Image.open(os.path.join(args["path"], filename))
        im=im.crop((500,500,2100,1700))
        #im.show()
        #break
        k+=1
        if args["savepath"] is not None:
        	im.save(args["savepath"]+"/img"+str(k)+".png")
    else:
        continue

