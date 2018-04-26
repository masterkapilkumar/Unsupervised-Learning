import numpy as np
import sys

x = np.load(sys.argv[1])

f=open("submit.csv",'w')
f.write("ID,CATEGORY\n")

for i,pred in enumerate(x):
    f.write(str(i)+","+str(pred)+'\n')
f.close()