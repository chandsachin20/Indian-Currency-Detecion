#!/usr/bin/python


from utils import *
from matplotlib import pyplot as plt

import subprocess
from gtts import gTTS

# image = read_img('files/500_1.jpg')
# orig = image
# orig = resize_img(orig, 0.5)
# img = resize_img(img, 0.6)

# img = img_to_gray(img)
# img = canny_edge(img, 720, 350)
# img = canny_edge(img, 270, 390)

# img = laplacian_edge(img)

# img = find_contours(img)
# img = img_to_neg(img)
# img = binary_thresh(img, 85)
# img = close(img)
# img = adaptive_thresh(img)
# img = sobel_edge(img, 'v')
# img = sobel_edge2(img)
# img = median_blur(img)
# img = binary_thresh(img, 106)
# img = dilate_img(img)
# img = binary_thresh(img, 120)

# img = foo_convolution(img)
# histogram(img)
# fourier(img)
# img = harris_edge(img)

# display('image',img)

# show the original image and the edge detected image
# cv2.imshow("Image", image)
# cv2.imshow("Edged", edged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# find the contours in the edged image, keeping only the
# largest ones, and initialize the screen contour

# must define here

max_val = 8
max_pt = -1
max_kp = 0

orb = cv2.ORB_create()
# orb is an alternative to SIFT

#test_img = read_img('files/test_100_2.jpg')
#test_img = read_img('files/test_50_2.jpg')
test_img = read_img('files/test_20_2.jpg')
#test_img = read_img('files/test_100_3.jpg')
# test_img = read_img('files/test_20_4.jpg')

# resizing must be dynamic
original = resize_img(test_img, 0.4)
display('original', original)

# keypoints and descriptors
# (kp1, des1) = orb.detectAndCompute(test_img, None)
(kp1, des1) = orb.detectAndCompute(test_img, None)

training_set = ['files/20.jpg', 'files/50.jpg', 'files/100.jpg', 'files/500.jpg']

for i in range(0, len(training_set)):
	# train image
	train_img = cv2.imread(training_set[i])

	(kp2, des2) = orb.detectAndCompute(train_img, None)

	# brute force matcher
	bf = cv2.BFMatcher()
	all_matches = bf.knnMatch(des1, des2, k=2)

	good = []
	# give an arbitrary number -> 0.789
	# if good -> append to list of good matches
	for (m, n) in all_matches:
		if m.distance < 0.789 * n.distance:
			good.append([m])

	if len(good) > max_val:
		max_val = len(good)
		max_pt = i
		max_kp = kp2

	print(i, ' ', training_set[i], ' ', len(good))

if max_val != 8:
	print(training_set[max_pt])
	print('good matches ', max_val)

	train_img = cv2.imread(training_set[max_pt])
	img3 = cv2.drawMatchesKnn(test_img, kp1, train_img, max_kp, good, 4)
	
	note = str(training_set[max_pt])[6:-4]
	print('\nDetected denomination: Rs. ', note)

	audio_file = 'audio/' + note + '.mp3'

	# audio_file = "value.mp3
	# tts = gTTS(text=speech_out, lang="en")
	# tts.save(audio_file)
	return_code = subprocess.call(["afplay", audio_file])

	(plt.imshow(img3), plt.show())
else:
	print('No Matches')