import glob
import os
fs = glob.glob('*.pickle')
for f in fs:
    if f.startswith('trial_573100'):
        n = f.removeprefix('trial_573100')
        n = n.split('_')[0]
        if int(n) < 256:
            print(f)
            os.remove(f)