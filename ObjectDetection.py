import cv2
import numpy as np

# Yolo 로드
net = cv2.dnn.readNet("model/608/yolov3.weights", "model/608/yolov3.cfg")

classes = []
with open("model/608/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 동영상은 시작부터 종료될때까지 프레임을 지속적으로 받아야 하기 때문에 while문을 계속 반복함
while True:

    # 웹캠의 영상은 이미지파일과 유사한 많은 프레임들이 빠르게 움직이면서 보여줌
    # 웹캠의 영상으로부터 프레임 1개마다 읽어오기 위해 사용
    ret, img = cam.read() # 프레임 1개마다 읽기

    # 웹캠 영상으로부터 프레임(이미지)를 잘 받았으면 실행함
    if ret is True:

        # 이미지 가져오기
        # img = cv2.imread("image/test.jpg")
        # img = cv2.resize(img, None, fx=0.4, fy=0.4)
        height, width, channels = img.shape

        # Detecting objects
        # blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        blob = cv2.dnn.blobFromImage(img, 1/256, (608, 608), (0, 0, 0), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y + 30), font, 3, color, 3)


        # 수정된 웹캠 프레임 출력
        cv2.imshow("Object Detection", img)

    # 입력받는 것 대기하기, 작성안하면, 결과창이 바로 닫힘
    if cv2.waitKey(1) > 0:
        break

# 사용이 완료된 웹캠을 해지하기
cam.release()

# 모든 창 닫기
cv2.destroyAllWindows()

