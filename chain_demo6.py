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

# other python files that me/Adam have made
import saverloader
import automizer

random.seed(125)
np.random.seed(125)

def main():    # IDEA: chain together pips from longer sequence given user specs, return some visualizations

    exp_name = '00' # (exp_name is used for logging notes that correspond to different runs)
    init_dir = 'reference_model'

  # Initialize main variables
    B = 1
    N = int(input("Enter the number of points to track: "))
    target = [int(input("Enter x coordinate of the starting point: ")), int(input("Enter y coordinate of the starting point: "))]
    decale = [int(input("Enter desired shift for each x coordinate: ")), int(input("Enter desired shift for each y coordinate: "))]
    xy0 = automizer.setPoints(B, N, target, decale)
    filenames = glob.glob('./demo_images/*.jpg')
    filenames = sorted(filenames)
    S = automizer.getS(N, filenames)

  # Set up folder to receive output videos
    if (os.path.exists('gifs_chain_demo') == True):
      shutil.rmtree('gifs_chain_demo')
    os.makedirs('gifs_chain_demo', exist_ok=True)

  # Autogenerate a name
    model_name = "%02d_%d_%d" % (B, S, N)
    model_name += "_%s" % exp_name
    model_date = datetime.datetime.now().strftime('%H.%M.%S')
    model_name = model_name + '_' + model_date
    print('model :', model_name)

  # Initialize the model evaluation method
    model = Pips(stride=4).cuda()
    parameters = list(model.parameters())
  # print('the model is: ', model)
  # print('Parameters are: ', parameters)
    print('init_dir is: ', init_dir)
    if init_dir:
        _ = saverloader.load(init_dir, model)
    model.eval()
    max_iters = len(filenames)//(S)			# determines number of runs i.e. number of output files there will be

    log_freq = 1 						# when to produce visualizations 
    log_dir = 'logs_chain_demo'
    writer_t = SummaryWriter(log_dir + '/' + model_name + '/t', max_queue=10, flush_secs=60)

    global_step = 0
    read_start_time = time.time()

    while global_step < max_iters:
        global_step += 1
        sw_t = utils.improc.Summ_writer(writer=writer_t, global_step=global_step, log_freq=log_freq,
            fps=12, scalar_freq=int(log_freq/2), just_gif=True)

        try:
            rgbs = []
            for s in range(S):
                fn = filenames[(global_step-1)*(S-12)+s]		# rewind next start frame a bit
                if s==0:
                    print('--------------------------------\nstart frame', fn)
                im = imageio.imread(fn)
                im = im.astype(np.uint8)
                rgbs.append(torch.from_numpy(im).permute(2,0,1))
            rgbs = torch.stack(rgbs, dim=0).unsqueeze(0) # 1, S, C, H, W

            read_time = time.time()-read_start_time
            iter_start_time = time.time()

            with torch.no_grad():
                res = automizer.run_model(model, rgbs, N, sw_t, xy0, fn, decale)
                trajs_e = res[0]
                xy0_raw = res[1]
                xy0 = automizer.reEvalPointsForNextRun(B, N, S, decale, xy0_raw)

            iter_time = time.time()-iter_start_time
            print('model : %s; step %06d/%d; read time %.2f; iter time %.2f' % (
                model_name, global_step, max_iters, read_time, iter_time))
        except FileNotFoundError as e:
            print('error', e)

    writer_t.close()
    automizer.concatenate()

if __name__ == '__main__':
    main()
