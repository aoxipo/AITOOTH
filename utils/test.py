import sys
import os
import cv2
import numpy as np

from fid import calculate_fid
from cdc import calculate_cdc


res_dir = '/path/to/val_result_frame'
ref_dir = '/path/to/val_gt'

print('Calculating FID...')
fid = calculate_fid(res_dir, ref_dir)
print('Calculating CDC...')
cdc = calculate_cdc(res_dir)

print(f'FID: {fid}')
print(f'CDC: {cdc}')