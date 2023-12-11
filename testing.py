import time
import numpy as np
import io
import os
import shutil
from PIL import Image
import cv2
import saverloader
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import ffmpeg
import pandas as pd

random.seed(125)
np.random.seed(125)
    
def calculateSFromFrames():
    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    print('Filenames was originally %d frames' % len(filenames), end = '')

    ## get S
    frames=len(filenames)
    frm_done=False
    check = 0
    while not frm_done:
      if frames < 160:
        print('Not enough frames. Reload video as a bigger clip or with higher fps.')
        frm_done=True
      if frames > 3299:
        print('Too many frames. Reload video as a smaller clip or with lower fps.')
        frm_done=True
      for n in range(1,30):
        y = frames/(n)
        if y == int(y) and y<110 and y>80: 
          S = int(y)
          filenames=filenames[:frames]
          print(', it is now %d, divided by %d gives an S of %d' %(frames, n, S))
          frm_done=True
          check=1
          break
      if check == 0:
        frames = frames-2
        #print('Adjusting frame set. Frames:', frames)
    
def setupPoints():

    ## choose hyps
    B = 1
    N = 3			# number of points to track

    xy0 = torch.ones((B, N, 2), dtype=torch.float32, device='cuda')
    
    ## choose points of interest
    xs = [408, 410, 412]
    ys = [220, 222, 224]
    for n in range(N):
      xy0[:,0+n:1+n,:1] = xs[n]
      xy0[:,0+n:1+n,1:] = ys[n]
    print(xy0)
  # print('\n Just one part of xy0:', xy0[:,1,:])		# returns tensor([[410., 222.]], device='cuda:0')
    return xy0

def userInput():
    n = int(input("Enter the number of points to track: "))
    xy = [int(input("Enter x coordinate of centroid: ")), int(input("Enter y coordinate of centroid: "))]
    xy_dec = [int(input("Enter desired shift for each x coordinate: ")), int(input("Enter desired shift for each y coordinate: "))]
    print('n =',n,'\nx, y =',xy,'\nx_dec, y_dec =', xy_dec)
    xy0 = setupPoints()

def concatenate():
    os.chdir(r'C:\Users\imcbr\OneDrive\Documents\CoOp\CSM_Monaco\WorkingFol\gifs_chain_demo')
    comm = 'ffmpeg -hide_banner -loglevel error -i \"concat:'
    vid = glob.glob('*.gif')
    file_temp = []
    for f in vid:
        file = 'temp' + str(vid.index(f) + 1) + '.ts'
        os.system('ffmpeg -hide_banner -loglevel error -i ' + f + ' -f mpegts ' + file)
        file_temp.append(file)
 #  print(file_temp)
    for f in file_temp:
        comm += f
        if file_temp.index(f) != len(file_temp)-1:
          comm += "|"
        else:
          comm += "\" -pix_fmt rgb24 run.gif"
    os.system(comm)

def reEvalPointsForNextRun():
    N=5
    decale = [5, 5]

    xy0_raw = torch.ones((1, N, 2), dtype=torch.float32, device='cuda')
    xs = [458.6214, 453.8610, 461.2572, 464.3767, 464.3767] 
    ys = [251.8424, 249.0949, 254.7653, 254.9706, 254.9706]
    for n in range(N):
      xy0_raw[:,0+n:1+n,:1] = xs[n]
      xy0_raw[:,0+n:1+n,1:] = ys[n]
    print('The created tensor is:', xy0_raw,'with size', xy0_raw.size())
    samp = np.zeros((N,2))
    count = 0

 #  find out what the biggest difference between values is and therefore if you need to go through longer code
    for n in range (N):
        if (abs(xy0_raw[:,n:n+1,:] - xy0_raw[:,n+1:n+2,:]) < [element * 2 for element in decale]).all():
            samp[count, 0] = xy0_raw[:,n:n+1,:1]
            samp[count, 1] = xy0_raw[:,n:n+1,1:]
            count += 1
        else:
        count


 #  q1 = np.quantile(samp, 0.25, axis=0)
 #  q3 = np.quantile(samp, 0.75, axis=0)
 #  IQR = q3-q1
 #  print('IQR=q3-q1:', IQR,'=',q3,'-',q1)
    print('medians are:', np.median(samp, axis=0))
    print('Orig samp:\n', samp)
    adjust = [element * 2 for element in decale]
    print('Adjust\n', adjust)
    for n in range(N):
        if (samp[n,:] < (np.median(samp, axis=0) - adjust)).any():
            samp[n] = np.median(samp, axis=0) + decale
            print('Correcting number', n+1, 'by adding', decale)
        elif (samp[n,:] > (np.median(samp, axis=0) + adjust)).any():
            samp[n] = np.median(samp, axis=0) - decale
            print('Correcting number', n+1, 'by subtracting', decale)
    for n in range(N):
        xy0_raw[:,n:n+1,:1] = samp[n,0]
        xy0_raw[:,n:n+1,1:] = samp[n,1]
    print('Checked next xy0 as.... ', xy0_raw)
    return xy0_raw

def main():
    reEvalPointsForNextRun()


if __name__ == '__main__':
    main()
