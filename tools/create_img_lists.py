import os

f = open("../models/train.txt", 'w')
dataset_basepath = "../data"
for p1 in os.listdir(dataset_basepath):
  image = os.path.abspath(dataset_basepath + '/' + p1)
  f.write(image + '\n')
f.close()
