import numpy as np
from PIL import Image
from pathlib import Path
from commonutils import tools
# Read village dem data
village_path = "../static/datasets"
village_data_dir = Path(village_path).glob("*.tif")

tl = tools.Tools
datalines = []
for f in village_data_dir:
    # Get Center of the data
    data = np.asarray(Image.open(str(f)))
    [px,py],_ = tl.coordinateTransformOfPoint(str(f), data.shape[0] / 2, data.shape[1] / 2)
    datalines.append([f.stem,px,py])


# Write into txt
np.savetxt("village_metadata.txt",datalines,delimiter=" ",fmt = "%s")