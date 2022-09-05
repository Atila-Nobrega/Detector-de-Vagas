import cv2



COLORS = [(69,139,0),(10,10,255)]

class_names = []
with open("coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

net = cv2.dnn.readNet("yolov4-custom_last.weights", "yolov4-custom.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(256,256), scale=1/255)


image = cv2.imread('estacionamento1.jpg')

classes, scores, boxes = model.detect(image, 0.1, 0.2)

vazio = 0
ocupado = 0 
for(classid, score, box) in zip(classes, scores, boxes):
    if(score > 0.2):
        if(classid==0):
            vazio += 1
        else:
            ocupado += 1

        color = COLORS[int(classid) % len(COLORS)]

        label = f'{class_names[classid]}'

        cv2.rectangle(image, box, color, 2)

        cv2.putText(image, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

vazioqnt = "Numero de Vagas: " + str(vazio)
ocupadoqnt = "Numero de Ocupados: " + str(ocupado)

cv2.putText(image, vazioqnt, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)
cv2.putText(image, ocupadoqnt, (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2)

cv2.imshow('Image', image)
cv2.waitKey(0)
