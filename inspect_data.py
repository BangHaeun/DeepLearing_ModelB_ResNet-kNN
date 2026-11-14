"""zip을 푼 폴더 구조 확인용. 사용: python inspect_data.py /content/MIMII_project_preprocessed"""
import os
import sys

import numpy as np

root = sys.argv[1]
for d, subdirs, files in sorted(os.walk(root)):
    depth = d[len(root):].count(os.sep)
    npy = [f for f in files if f.endswith(".npy")]
    other = [f for f in files if not f.endswith(".npy")]
    line = "  " * depth + os.path.basename(d) + "/"
    if npy:
        shp = np.load(os.path.join(d, npy[0]), mmap_mode="r").shape
        line += f"   [.npy {len(npy)}개, 예시 shape={shp}]"
    if other:
        line += f"   [기타 {len(other)}개: {other[:2]}]"
    print(line)
    if depth >= 5:
        subdirs[:] = []
