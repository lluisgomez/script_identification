from scipy import misc
from skimage import color
import numpy as np

img_height = 40
win_size = 32
win_step = 8

fname = 'orig/train-list.txt'

with open(fname) as f:
  content = f.readlines()

for line in content:
  gt = line.rstrip().split(' ')
  filename = 'orig/'+gt[0]
  label = gt[1]


  img = misc.imread(filename)
  img = color.rgb2gray(img)
  (h,w) = img.shape
  img = misc.imresize(img,(img_height,w*img_height/h))
  (h,w) = img.shape

  count = 0
  for x in range(0,w-win_size+1,win_step):
    if x < w-h:
      #patch = img[:,x:x+h]
      #patch = misc.imresize(patch,(win_size,win_size))
      #patch_filename = 'patches/train/'+filename.split('/')[2].split('.')[0] + '_' + str(count) + '.jpg'
      #misc.imsave(patch_filename, patch)
      #print 'data/'+patch_filename+' '+label
      count = count+1
    for y in range(0,h-win_size+1,win_step):
      
      #patch = img[y:y+win_size,x:x+win_size]
      #patch_filename = 'patches/train/'+filename.split('/')[2].split('.')[0] + '_' + str(count) + '.jpg'
      #misc.imsave(patch_filename, patch)
      #print 'data/'+patch_filename+' '+label
      count = count+1

  patch_filename_root = 'data/patches/train/'+filename.split('/')[2].split('.')[0] + '_'
  num_patches = count;
  while num_patches < 10: # this is just to take into account images with less than 10 patches
    num_patches = num_patches*2;

  for i in range(num_patches-10):
    choice = np.random.choice(num_patches, 10)%count
    print patch_filename_root+str(choice[0])+'.jpg '+patch_filename_root+str(choice[1])+'.jpg '+patch_filename_root+str(choice[2])+'.jpg '+patch_filename_root+str(choice[3])+'.jpg '+patch_filename_root+str(choice[4])+'.jpg '+patch_filename_root+str(choice[5])+'.jpg '+patch_filename_root+str(choice[6])+'.jpg '+patch_filename_root+str(choice[7])+'.jpg '+patch_filename_root+str(choice[8])+'.jpg '+patch_filename_root+str(choice[9])+'.jpg '+label
