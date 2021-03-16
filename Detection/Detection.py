import cv2
import os

directory = "C:\\Users\\Moxis\\Downloads\\camera00\\00"

for files in os.listdir(directory):
	if files.endswith(".jp2"):
		img = cv2.imread(os.path.join(directory,files))
		cv2.imshow(files,img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
