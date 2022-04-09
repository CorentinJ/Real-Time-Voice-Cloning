import datetime
import time

import ffmpeg
import os
import argparse
from multiprocessing import Pool, freeze_support
from functools import partial


parser = argparse.ArgumentParser(description='Dataset args')
parser.add_argument("-d", "--dataset_path", type=str, help="Path to your root dataset folder")
parser.add_argument("-m", "--multiproc", type=bool, help="Use multiprocessing?", default=True)  # speed up 2x-4x

test_run = False

def ffmpeg_convert(dspath, dir1, dir2, filei):
    i, file = filei
    fullname = dspath + dir1 + "\\" + dir2 + "\\" + file
    if fullname.split(".")[1] == "opus":
        cleandirname = str(i)
        while len(cleandirname) < 3:
            cleandirname = "0" + cleandirname
        wavname = dspath + dir1 + "\\" + dir2 + "\\" + "utterance-" + cleandirname + ".wav"
        txtname = fullname.replace("opus", "txt")
        txtoutname = wavname.replace("wav", "txt")
        if test_run:
            print(fullname, wavname)
            print(txtname, txtoutname)
        else:
            (
                ffmpeg  # the ffmpeg part, works like a charm but very slowly
                    .input(fullname)
                    .output(wavname, loglevel="quiet")
                    .run()
            )
            os.remove(fullname)  # be careful
            os.rename(txtname, txtoutname)

def convert_opuses(dspath):
    times = []
    for dir1 in os.listdir(dspath):
        if dir1 == "speaker-014" or dir1 == "speaker-015":
            tim = time.time()
            for dir2 in os.listdir(dspath+dir1):
                if args.multiproc:
                    pool = Pool()
                    pool.map(partial(ffmpeg_convert, dspath, dir1, dir2), enumerate(os.listdir(dspath+dir1+"\\"+dir2)))
                else:
                    for i, qdir in enumerate(os.listdir(dspath+dir1+"\\"+dir2)):
                        ffmpeg_convert(dspath, dir1, dir2, (i, qdir))

            times.append(time.time() - tim)
            print("time per dir {}: ".format(str(dir1)), datetime.timedelta(seconds=round(time.time() - tim)))
    print("overall time: ", datetime.timedelta(seconds=round(sum(times))))

def rename_secondary_dir(insecdir):
    j=0
    for indirq in os.listdir(insecdir):
        inbefore = insecdir+"\\"+indirq
        s = str(j)
        while len(s) < 3:
            s = "0"+s
        newpath = insecdir+"\\"+"book-"+s
        j+=1
        if test_run:
            print(inbefore, newpath)
        else:
            os.rename(inbefore, newpath)

def restore_structure(rpath):
    i=0
    for dirq in os.listdir(rpath):
        s = str(i)
        while len(s) < 3:
            s = "0"+s
        newpath = "speaker-"+s
        i+=1
        if test_run:
            print(rpath+dirq, rpath+newpath)
        else:
            os.rename(rpath+dirq, rpath+newpath)
        rename_secondary_dir(rpath + newpath)

if __name__=="__main__":
    freeze_support()
    args = parser.parse_args()
    pat = args.dataset_path if args.dataset_path[-1] == "\\" or "/" else args.dataset_path+"\\"
    print("Working with path: ", pat)
    if "speaker-000" in os.listdir(pat):
        print("structure already restored previously, skipping")
    else:
        print("restoring structure..")
        restore_structure(pat)
    print("converting opuses..")
    convert_opuses(pat)
