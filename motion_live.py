import cv2
from datetime import datetime
import yt_dlp

static_back = None

url = "https://www.youtube.com/watch?v=1fiF7B6VkCk"
ydl_opts = {'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'}
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(url, download=False)
    video_url = info_dict.get('url', None)

video = cv2.VideoCapture(video_url)

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

    if (itrr_frame_time - curr_frame_time).total_seconds() >= 0.04:
        static_back = gray
        curr_frame_time = datetime.now()

    diff_frame = cv2.absdiff(static_back, gray)

    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]

    cnts, hierarchy = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 100:
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
