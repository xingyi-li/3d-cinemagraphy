import os
import argparse
import cv2
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputdir', type=str, required=True, help='the path to the input directory')
    args = parser.parse_args()

    img = cv2.imread(os.path.join(args.inputdir, 'label.png'))
    b, g, r = cv2.split(img)
    b, g, r = (b > 0).astype(np.int64), (g > 0).astype(np.int64), (r > 0).astype(np.int64)
    mask = cv2.bitwise_or(b, g)
    mask = cv2.bitwise_or(mask, r)
    mask = mask * 255
    cv2.imwrite(os.path.join(args.inputdir, 'mask.png'), mask)
    print(f"Generated mask for {os.path.join(args.inputdir, 'label.png')}")
