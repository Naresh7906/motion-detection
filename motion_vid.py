import cv2
from datetime import datetime

static_back = None

video = cv2.VideoCapture('video.mp4')

curr_frame_time = datetime.now()

fps = 30

while True:
	itrr_frame_time = datetime.now()
	check, frame = video.read()
 
	motion = 0

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if static_back is None:
		static_back = gray
		continue

	if (itrr_frame_time-curr_frame_time).total_seconds() >= 0.1:
		static_back = gray
		curr_frame_time = datetime.now()

	diff_frame = cv2.absdiff(static_back, gray)

	thresh_frame = cv2.threshold(diff_frame, 90, 255, cv2.THRESH_BINARY)[1]

	#cnts,herch = cv2.findContours(thresh_frame.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts, hierarchy = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	for contour in cnts:
		if cv2.contourArea(contour) <500 or cv2.contourArea(contour) > 1000:
			continue
		motion = 1

		(x, y, w, h) = cv2.boundingRect(contour)

		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


	cv2.imshow("Gray Frame", gray)

	cv2.imshow("Difference Frame", diff_frame)

	cv2.imshow("Threshold Frame", thresh_frame)

	cv2.imshow("Color Frame", frame)


	key = cv2.waitKey(1)
	if key == ord(' '):
		break


video.release()

cv2.destroyAllWindows()
