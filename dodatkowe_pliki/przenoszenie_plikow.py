import os
import random
import shutil

source = '../data/images'
dest = '../data/validate/images'

files = os.listdir(source)
no_of_files = 10

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)

# train 44
# test 9
# validate 10