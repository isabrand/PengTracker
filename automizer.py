# Python packages/modules to use
import time
import numpy as np
import io
import os
import shutil
from PIL import Image
import cv2
import imageio.v2 as imageio
from nets.pips import Pips
import utils.improc
import random
import glob
from utils.basic import print_, print_stats
import torch
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import datetime
import sys
import ffmpeg
from matplotlib import pyplot as plt 

random.seed(125)
np.random.seed(125)

def setPoints(B, N, target, decale):
  # Takes user input and calculates the points they want to track, saving to and passing a Tensor
    x = target[0]
    y = target[1]
    if N%2==0:					# even numbers will be augmented by 1 for evenness
      N = N+1
    xplt = np.arange(0,N)
    yplt = np.arange(0,N)
    xplt[(N//2)] = x
    yplt[(N//2)] = y
    for n in range(1, (N//2)+1):
      countup = [x + n*decale[0], y + n*decale[1]]
      xplt[(N//2)+n] = countup[0]
      yplt[(N//2)+n] = countup[1]
      countdown = [x - n*decale[0], y - n*decale[1]]
      xplt[(N//2)-n] = countdown[0]
      yplt[(N//2)-n] = countdown[1]

    xy0 = torch.ones((B, N, 2), dtype=torch.float32, device='cuda')
    for n in range(N):
      xy0[:,0+n:1+n,:1] = xplt[n]
      xy0[:,0+n:1+n,1:] = yplt[n]
    print('--> tracking.... --> ', xy0)
    plt.plot(xplt,yplt,"ob") 
    plt.show(block=False) 
    return xy0

def getS(N, filenames):
    print('\nFilenames was originally %d frames' % len(filenames), end = '')
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
          print(', now %d, divided by %d gives a frame count (S)  of %d' %(frames, n, S), '\n')
          frm_done=True
          check=1
          break
      if check == 0:
        frames = frames-2
        #print('Adjusting frame set. Frames:', frames)
    return S

def run_model(model, rgbs, N, sw, xy0, fn, decale):
    rgbs = rgbs.cuda().float() # B, S, C, H, W

    B, S, C, H, W = rgbs.shape
    rgbs_ = rgbs.reshape(B*S, C, H, W)
    H_, W_ = 480, 854
    rgbs_ = F.interpolate(rgbs_, (H_, W_), mode='bilinear')
    H, W = H_, W_
    rgbs = rgbs_.reshape(B, S, C, H, W)
    _, S, C, H, W = rgbs.shape

    trajs_e = torch.zeros((B, S, N, 2), dtype=torch.float32, device='cuda')
    new_xy0 = torch.ones((B, N, 2), dtype=torch.float32, device='cuda')

    for n in range(N):
        xy0_rn = xy0[:,n,:]
        print('working on point (kp) %d/%d' % (n+1, N), 'with xy0 of', xy0_rn)
        cur_frame = 0
        done = False
        traj_e = torch.zeros((B, S, 2), dtype=torch.float32, device='cuda')
     #  print('Starting in traj_e with this val:', xy0_rn[:,:])
        traj_e[:,0] = xy0_rn[:,:] # B, 1, 2  						# set first position
        feat_init = None
        while not done:
            frmst = 8
            end_frame = cur_frame + frmst

            rgb_seq = rgbs[:,cur_frame:end_frame]
            S_local = rgb_seq.shape[1]
            rgb_seq = torch.cat([rgb_seq, rgb_seq[:,-1].unsqueeze(1).repeat(1,frmst-S_local,1,1,1)], dim=1)

            outs = model(traj_e[:,cur_frame].reshape(1, -1, 2), rgb_seq, iters=6, feat_init=feat_init, return_feat=True)
            preds = outs[0]
            vis = outs[2] # B, S, 1
            feat_init = outs[3]
            
            vis = torch.sigmoid(vis) # visibility confidence
            xys = preds[-1].reshape(1, frmst, 2)
            traj_e[:,cur_frame:end_frame] = xys[:,:S_local]

            found_skip = False
            thr = 0.9
            si_last = frmst-1 # last frame we are willing to take
            si_earliest = 1 # earliest frame we are willing to take
            si = si_last
            while not found_skip:
                if vis[0,si] > thr:
                    found_skip = True
                else:
                    si -= 1
                if si == si_earliest:
                    #print('si is si_earliest... decreasing thresh')
                    thr -= 0.02
                    si = si_last
            #print('found skip at frame %d, where we have' % si, vis[0,si].detach().item())

            cur_frame = cur_frame + si

            if cur_frame >= S:
                done = True
        trajs_e[:,:,n] = traj_e
    
    pad = 50
    rgbs = F.pad(rgbs.reshape(B*S, 3, H, W), (pad, pad, pad, pad), 'constant', 0).reshape(B, S, 3, H+pad*2, W+pad*2)
  # print('Trajs_e without the pad is: \n', trajs_e[:,0:3,:,:], '...')
    trajs_e = trajs_e + pad
  # print('Trajs_e with the pad is: \n', trajs_e[:,0:3,:,:], '...\nAnd has a size of ', trajs_e.size())

    prep_rgbs = utils.improc.preprocess_color(rgbs)
    gray_rgbs = torch.mean(prep_rgbs, dim=2, keepdim=True).repeat(1, 1, 3, 1, 1)
    
    if sw is not None and sw.save_this:
       # for n in range(N):						# used to be a for loop from here until out_fn, now does it all at once
       #    print('Visualizing point (kp) %d/%d' % (n+1, N))
       #    print('Input tensor trajs_e has size ', trajs_e.size(),' and looks like', trajs_e[0:1])
        kp_vis = sw.summ_traj2ds_on_rgbs('video_%d/kp_%d_trajs_e_on_rgbs' % (sw.global_step, n), trajs_e[0:1], gray_rgbs[0:1,:S], cmap='spring', linewidth=1)

        # write to disk, in case that's more convenient
        kp_list = list(kp_vis.unbind(1))
        kp_list = [kp[0].permute(1,2,0).cpu().numpy() for kp in kp_list]
        kp_list = [Image.fromarray(kp) for kp in kp_list]
            
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb', trajs_e[0:1], prep_rgbs[0:1,0], cmap='spring')
        sw.summ_traj2ds_on_rgb('outputs/trajs_e_on_rgb2', trajs_e[0:1], torch.mean(prep_rgbs[0:1], dim=1), cmap='spring')

        out_fn = './chain_out_%d.gif' % sw.global_step
        kp_list[0].save(os.path.join('gifs_chain_demo', out_fn), save_all=True, append_images=kp_list[1:])
        print('end frame', fn, '\n')                
        print('SAVED %s' % out_fn)

    ret = trajs_e-pad
    new_xy0 = ret[:,S-1:S,:,:]
  # print('new_xy pulled from trajs_e - pad: ', new_xy)
    new_xy0 = new_xy0.reshape(1, N, 2)
  # print('reshaped new_xy: ', new_xy) 
  # print('Returning the trajs_e - pad as.... ', ret[:,0:3,:,:])
    print('Unchecked next xy0 as.... ', new_xy0)
    return ret, new_xy0

def reEvalPointsForNextRun(B, N, S, decale, xy0_raw):
    print('xy0_raw is:', xy0_raw)
    samp = np.zeros((N,2))
    for n in range(N):
        samp[n, 0] = xy0_raw[:,n:n+1,:1]
        samp[n, 1] = xy0_raw[:,n:n+1,1:]
  # print('medians are:', np.median(samp, axis=0))
  # print('Orig samp:\n', samp)
    adjust = [(1+((element*N)//2)) for element in decale]
  # print('Adjust\n', adjust)
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

def concatenate():
  # Basically breaks down ffmpeg command and does it through python instead
    os.chdir(r'C:\Users\imcbr\OneDrive\Documents\CoOp\CSM_Monaco\WorkingFol\gifs_chain_demo')
    comm = 'ffmpeg -hide_banner -loglevel error -i \"concat:'
    vid = glob.glob('*.gif')
    file_temp = []
    for f in vid:
        file = 'temp' + str(vid.index(f) + 1) + '.ts'
        os.system('ffmpeg -hide_banner -loglevel error -i ' + f + ' -f mpegts ' + file)
        file_temp.append(file)
    for f in file_temp:
        comm += f
        if file_temp.index(f) != len(file_temp)-1:
          comm += "|"
        else:
          comm += "\" -pix_fmt rgb24 run.gif"
    os.system(comm)
    os.system('run.gif')
    os.chdir(r'C:\Users\imcbr\OneDrive\Documents\CoOp\CSM_Monaco\WorkingFol')
