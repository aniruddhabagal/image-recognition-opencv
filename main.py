import cv2
from matplotlib import pyplot as plt

img = cv2.imread("image.png")

# image_src = ""

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# object_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
object_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


foundObjects = object_cascade.detectMultiScale(img_gray, minSize=(30,30))

for (x, y, w, h) in foundObjects:
    cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 5)

plt.imshow(img_rgb)
plt.show()