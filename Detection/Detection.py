import cv2
import os
import matplotlib.pyplot as plt

directory = "C:\\Users\\Moxis\\Downloads\\BelgiumTSC_Testing\\Testing"

labels = []
files = []
count = -1
for x in os.listdir(directory):
	count += 1
	if x.isdigit():
		labels.append(x[3:5])
		files.append(0)
		for y in os.listdir(os.path.join(directory, x)):
			if y.endswith("ppm"):
				files[count] += 1

print(len(labels))
print(len(files))
assert len(labels) == len(files)

plt.bar(labels, files)
plt.xlabel("Label name")
plt.ylabel("Images count")
plt.xticks(rotation=90, fontsize=6)
plt.show()

'''
for folder in os.listdir(directory):
	if os.path.isdir(os.path.join(directory,folder)):
		for files in os.listdir(os.path.join(directory, folder)):
			if files.endswith(".ppm"):
				print(os.path.join(directory, folder, files))
				img = cv2.imread(os.path.join(directory,folder, files))
				cv2.imshow(files, img)
				cv2.waitKey(0)
				cv2.destroyAllWindows()
'''
