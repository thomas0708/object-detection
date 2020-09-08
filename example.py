import glob
import xml.etree.ElementTree as ET

import numpy as np
import time
# import sys
# print(sys.version)

from kmeans import kmeans, mini_batch_kmeanspp, avg_iou


# ANNOTATIONS_PATH = "../data_BCCD/train/annotation" # cell
# ANNOTATIONS_PATH = "../data/train/annotation" # crop
ANNOTATIONS_PATH = "../data_rsod/train/annotation" # sensing
CLUSTERS = 9

def load_dataset(path):
	dataset = []
	for xml_file in glob.glob("{}/*xml".format(path)):
		tree = ET.parse(xml_file)

		height = int(tree.findtext("./size/height"))
		width = int(tree.findtext("./size/width"))

		for obj in tree.iter("object"):
			xmin = float(obj.findtext("bndbox/xmin")) / width
			ymin = float(obj.findtext("bndbox/ymin")) / height
			xmax = float(obj.findtext("bndbox/xmax")) / width
			ymax = float(obj.findtext("bndbox/ymax")) / height

			dataset.append([xmax - xmin, ymax - ymin])

	return np.array(dataset)

data = load_dataset(ANNOTATIONS_PATH)
start = time.time()
# out = kmeans(data, k=CLUSTERS)
out = mini_batch_kmeanspp(data, k=CLUSTERS, batch_size=50, num_iter=10, replacement=True)
end = time.time()
print("Time: {:.2f}".format(end - start))
print("Accuracy: {:.2f}%".format(avg_iou(data, out) * 100))
# print("Boxes:\n {}".format(out))
# print("x:", out[:, 0]*416)
# print("y:", out[:, 1]*416)
x = out[:, 0]*416
y = out[:, 1]*416
print(x)
print(y)

# ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
# print("Ratios:\n {}".format(sorted(ratios)))