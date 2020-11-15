from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
from DLWP.preprocessing.aspectawarepreprocessor import AspectAwarePreprocessor
app = AspectAwarePreprocessor(224, 224)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def load_models():
    model = load_model('output/best.hdf5')
    model._make_predict_function() 
    print('model loaded') # just to keep track in your server
    return model
model = load_models()

def get_label(path):
    frame = cv2.imread(path)
    # print(frame)
    frameClone = frame.copy()
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2frame)
    # frameClone = app.preprocess(frameClone)
    rects = detector.detectMultiScale(frame, scaleFactor=1.1,minNeighbors=5, minSize=(200, 200),flags=cv2.CASCADE_SCALE_IMAGE)
    # print("here3")
    for (fX, fY, fW, fH) in rects:
        # extract the ROI of the face from the framescale image,
        # resize it to a fixed 28x28 pixels, and then prepare the
        # ROI for classification via the CNN
        roi = frame[fY:fY + fH, fX:fX + fW]
        print("ROI:", roi)
        # roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        # roi = cv2.resize(roi, (224, 224))
        roi = img_to_array(roi)

        roi = app.preprocess(roi)
        roi = roi.astype("float")/255.0
        roi = np.expand_dims(roi, axis=0)
        (mask, no_mask) = model.predict(roi)[0]
        label = "Mask {}".format(mask*100) if mask > no_mask else "No Mask {}".format(no_mask*100)
#         color = (0, 255, 0) if mask > no_mask else (0, 0, 255)

    return label

    #         cv2.putText(frameClone, label, (fX, fY - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    #         cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),color, 2)

    #         cv2.imshow("Face", frameClone)

    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #         break
# camera.release()
# cv2.destroyAllWindows()
