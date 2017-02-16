from PIL import Image
import argparse
import os


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--namebase", required=True,
	help="names of images in folder mus be like /namebase/smth,/namebase/smth etc ")
ap.add_argument("-p","--path",required=False,help="path to folder containing images")
ap.add_argument("-r","--rotations",required=True,help="list of degrees to turn on")
ap.add_argument("-e","--expand",required=False,help="wether to extand rotated images to original size")
ap.add_argument("--stopat",required=False,help="num of images to be turned")
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

if args["expand"] is None:
    args["expand"]=False
#if args["format"] is None:
#	args["format"] = "JPG"

args["rotations"] = [float(i) for i in args["rotations"].split(" ")]
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
        fullname = os.path.join(args["path"], filename)
        print("Processing ",fullname)
        im = Image.open(fullname)
        #im=im.crop((500,500,2100,1700))
        ims=[]
        for rot in args["rotations"]:
            print("ROT",rot)
            ims.append(im.rotate(rot,expand = args["expand"]))
        #for df in ims:
        #    df.show()

        #im.show()
        #break
        k+=1
        l=0
        #ims[0].show()
        #break
        if args["savepath"] is not None:
            for pic in ims:
                l+=1
                #pic.show()
                #break        
                im.save(args["savepath"]+"/Rotatedimg"+str(k)+"_"+str(l)+".png")
        if args["stopat"] is not None:
            if k>= int(args["stopat"]):
                break
    else:
        continue

