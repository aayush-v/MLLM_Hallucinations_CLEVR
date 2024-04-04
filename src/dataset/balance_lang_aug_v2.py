import numpy as np
import json

path = ''
yes = []
no = []

fp = open(path,)
objects = json.load(fp)

for obj in objects:
    if obj['ground_truth'].lower() == 'yes':
        yes.append()
    elif obj['ground_truth'].lower() == 'no':
        no.append()

small = min(yes, no)

print(small)


