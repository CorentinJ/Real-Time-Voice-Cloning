import ffmpeg
import os
import argparse
from multiprocessing import Pool, freeze_support
from pathlib import Path
from functools import partial

parser = argparse.ArgumentParser(description='Dataset args')
parser.add_argument("-d", "--dataset_path", type=Path, help="Path to your root dataset folder")

#this file probably sucks but it worked when I used it, idk about now :D

def ffmpeg_convert(filei, dspath, dir1, dir2):
    file, i = filei
    fullname = dspath + dir1 + "\\" + dir2 + "\\" + file
    if fullname.split(".")[1] == "opus":
        cleandirname = str(i)
        while len(cleandirname) < 3:
            cleandirname = "0" + cleandirname
        wavname = dspath + dir1 + "\\" + dir2 + "\\" + "utterance-" + cleandirname + ".wav"
        # print()
        # print(fullname, "to", wavname)
        txtname = fullname.replace("opus", "txt")
        txtoutname = wavname.replace("wav", "txt")
        # print(txtname, txtoutname)
        # print()
        (
            ffmpeg  # the ffmpeg part, works like a charm but very slowly
                .input(fullname)
                .output(wavname)
                .run()
        )
        # print(fullname)
        os.remove(fullname)  # be careful
        os.rename(txtname, txtoutname)

def convert_opuses(dspath):
    for dir1 in os.listdir(dspath):
        for dir2 in os.listdir(dspath+dir1):
            pool = Pool()
            pool.map(partial(ffmpeg_convert, dspath, dir1, dir2), enumerate(os.listdir(dspath+dir1+"\\"+dir2)))

def rename_secondary_dir(insecdir):
    j=0
    for indirq in os.listdir(insecdir):
        inbefore = insecdir+"\\"+indirq
        s = str(j)
        while len(s) < 3:
            s = "0"+s
        newpath = insecdir+"\\"+"book-"+s
        j+=1
        os.rename(inbefore, newpath)
        # print("\t"+inbefore, "to", newpath)

def restore_structure(rpath):
    i=0
    for dirq in os.listdir(rpath):
        s = str(i)
        while len(s) < 3:
            s = "0"+s
        newpath = "speaker-"+s
        i+=1
        os.rename(rpath+dirq, rpath+newpath)
        rename_secondary_dir(rpath + "\\" + newpath)

if __name__=="__main__":
    freeze_support()
    args = parser.parse_args()
    restore_structure(args.dataset_path)
    convert_opuses(args.dataset_path)
