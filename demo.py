import cv2
from PIL import Image
import numpy as np

x = cv2.imread('/Users/nmy/STUDY/lab/works/moving_frame/1MNT_scene37_0030.jpg')
y = cv2.imread('/Users/nmy/STUDY/lab/works/moving_frame/1MNT_scene37_0032.jpg')
x_xdog = cv2.imread('/Users/nmy/1MNT_scene37_0030.jpg')
y_xdog = cv2.imread('/Users/nmy/1MNT_scene37_0032.jpg')
x = cv2.resize(x, (512, 512))
y = cv2.resize(y, (512, 512))
x_xdog = cv2.resize(x_xdog, (512, 512))
y_xdog = cv2.resize(y_xdog, (512, 512))

logit = cv2.absdiff(x_xdog, y_xdog)
cv2.imwrite('./pics/save_xdog.jpg', logit)
logit = cv2.absdiff(x, y)
logit = cv2.cvtColor(logit, cv2.COLOR_RGB2GRAY)
# cv2.imshow('hh', logit)
# cv2.waitKey(0)
retval, logit = cv2.threshold(logit, 15, 255, cv2.THRESH_BINARY)
print(logit)
cv2.imwrite('./pics/save_cv.jpg', logit)

region1 = cv2.bitwise_and(x, x, mask=~logit)
cv2.imwrite('./pics/region1.jpg', region1)
background1 = cv2.bitwise_and(x, x, mask=logit)
cv2.imwrite('./pics/background1.jpg', background1)

region2 = cv2.bitwise_and(y, y, mask=~logit)
cv2.imwrite('./pics/region2.jpg', region2)
background2 = cv2.bitwise_and(y, y, mask=logit)
cv2.imwrite('./pics/background2.jpg', background2)
# cv2.imshow('hh', logit)
# cv2.waitKey(0)
