from mylib import config, thread
from mylib.mailer import Mailer
from mylib.detection import detect_people
from imutils.video import VideoStream, FPS
from scipy.spatial import distance as dist
import numpy as np
import argparse, imutils, cv2, os, time, schedule
import threading
import socketserver
import http.server

PORT = 8080

pageData = "<!DOCTYPE>" + \
            "<html>" + \
            "  <head>" + \
            "    <title>Social Distancer</title>" + \
            "     <style>body { background-color: black; text-align: center; } #image { height: 100% }</style>" + \
            "  </head>" + \
            "  <body>" + \
            "<img id=\"image\"/>"+ \
            " <script type=\"text/javascript\">var image = document.getElementById('image');function refresh() {image.src = \"/image?\" + new Date().getTime();image.onload= function(){setTimeout(refresh, 30);}}refresh();</script>   "+ \
            "  </body>" + \
            "</html>"
# pageData = "<!DOCTYPE>" + \
#             "<html>" + \
#             "  <head>" + \
#             "    <title>Social Distancer</title>" + \
#             "     <style>html { background: url(/image) no-repeat center center fixed; background-size: contain; background-color: black; transition: background-image 1s ease-in-out;}</style>" + \
#             " <script type=\"text/javascript\">var image = document.documentElement;function refresh() {image.style.backgroundImage = \"url(/image?\" + new Date().getTime() + \")\"};setInterval(refresh, 1000);</script>   "+ \
#             "  </head>" + \
#             "  <body>" + \
#             "  </body>" + \
#             "</html>"

            # " <script type=\"text/javascript\">var image = document.documentElement;function refresh() {image.style.backgroundImage = \"url(/image?\" + new Date().getTime() + \");image.onload= function(){setTimeout(refresh, 30);}}refresh();</script>   "+ \
            #"<img id=\"image\" />"+ \
            # " <script type=\"text/javascript\">var image = document.getElementById('image');function refresh() {image.src = \"/image?\" + new Date().getTime();image.onload= function(){setTimeout(refresh, 30);}}refresh();</script>   "+ \

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):

        if self.path == '/':
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(bytes(pageData, "utf8"))
        elif self.path.startswith('/image'):
            self.send_response(200)
            self.send_header("Content-type", "image/jpeg")
            self.end_headers()

            # ret, frame = cap.read()
            # _, jpg = cv2.imencode(".jpg", frame)

            self.wfile.write(jpg)
        else:
            self.send_response(404)
# from web import MyHandler

#----------------------------Parse req. arguments------------------------------#
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, default="",
        help="path to (optional) input video file")
ap.add_argument("-o", "--output", type=str, default="",
        help="path to (optional) output video file")
ap.add_argument("-d", "--display", type=int, default=1,
        help="whether or not output frame should be displayed")
args = vars(ap.parse_args())
#------------------------------------------------------------------------------#

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([config.MODEL_PATH, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([config.MODEL_PATH, "yolov3.weights"])
configPath = os.path.sep.join([config.MODEL_PATH, "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# check if we are going to use GPU
if config.USE_GPU:
    # set CUDA as the preferable backend and target
        print("")
        print("[INFO] Looking for GPU")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# determine only the *output* layer names that we need from YOLO
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# if a video path was not supplied, grab a reference to the camera
print("[INFO] Starting the live stream..")
# vs = cv2.VideoCapture(config.url)
cap = thread.ThreadingClass(config.url)
#vs.set(cv2.CAP_PROP_BUFFERSIZE, 1)

time.sleep(2)

# otherwise, grab a reference to the video file
# start the FPS counter
fps = FPS().start()


class FrameThread (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.isRunning = True

    def run(self):
        global jpg, cap

        while self.isRunning:
            frame = cap.read()

            # resize the frame and then detect people (and only people) in it
            frame = imutils.resize(frame, width=1300)
            results = detect_people(frame, net, ln,
                    personIdx=LABELS.index("person"))

            # initialize the set of indexes that violate the max/min social distance limits
            serious = set()
            abnormal = set()

            # ensure there are *at least* two people detections (required in
            # order to compute our pairwise distance maps)
            if len(results) >= 2:
                # extract all centroids from the results and compute the
                    # Euclidean distances between all pairs of the centroids
                    centroids = np.array([r[2] for r in results])
                    D = dist.cdist(centroids, centroids, metric="euclidean")

                    # loop over the upper triangular of the distance matrix
                    for i in range(0, D.shape[0]):
                        for j in range(i + 1, D.shape[1]):
                            # check to see if the distance between any two
                                    # centroid pairs is less than the configured number of pixels
                                    if D[i, j] < config.MIN_DISTANCE:
                                        # update our violation set with the indexes of the centroid pairs
                                            serious.add(i)
                                            serious.add(j)
                    # update our abnormal set if the centroid distance is below max distance limit
                                    if (D[i, j] < config.MAX_DISTANCE) and not serious:
                                        abnormal.add(i)
                                        abnormal.add(j)

            # loop over the results
            for (i, (prob, bbox, centroid)) in enumerate(results):
                # extract the bounding box and centroid coordinates, then
                    # initialize the color of the annotation
                    (startX, startY, endX, endY) = bbox
                    (cX, cY) = centroid
                    color = (0, 255, 0)

                    # if the index pair exists within the violation/abnormal sets, then update the color
                    if i in serious:
                        color = (0, 0, 255)
                    elif i in abnormal:
                        color = (0, 255, 255) #orange = (0, 165, 255)

                    # draw (1) a bounding box around the person and (2) the
                    # centroid coordinates of the person,
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    cv2.circle(frame, (cX, cY), 5, color, 2)

            text = "Violations: {}".format(len(serious))
            cv2.putText(frame, text, (10, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2)

            fps.update()
            fps.stop()
            text = "FPS: {:.2f}".format(fps.fps())
            cv2.putText(frame, text, (frame.shape[1]-150, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 255, 255), 2)

            _, jpg = cv2.imencode(".jpg", frame)

        print("Quit thread")

frame_thread = FrameThread()
frame_thread.start()

with socketserver.TCPServer(("", PORT), MyHandler) as httpd:
    print("Serving at port ", PORT)
    try:
        httpd.serve_forever()
    except:
        pass

# quit the app
print('Server is stopped')
frame_thread.isRunning = False
frame_thread.join()
cap.release()


fps.stop()
print("===========================")
print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))
