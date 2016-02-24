from scipy import misc
from skimage import color

img_height = 40
win_size = 32
win_step = 8

fname = 'orig/test-list.txt'

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
      patch = img[:,x:x+h]
      patch = misc.imresize(patch,(win_size,win_size))
      patch_filename = 'patches/test/'+filename.split('/')[2].split('.')[0] + '_' + str(count) + '.jpg'
      misc.imsave(patch_filename, patch)
      print 'data/'+patch_filename+' '+label
      count = count+1

    for y in range(0,h-win_size+1,win_step):
      patch = img[y:y+win_size,x:x+win_size]
      patch_filename = 'patches/test/'+filename.split('/')[2].split('.')[0] + '_' + str(count) + '.jpg'
      misc.imsave(patch_filename, patch)
      print 'data/'+patch_filename+' '+label
      count = count+1
