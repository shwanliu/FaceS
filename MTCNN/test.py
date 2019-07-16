import cv2
import sys
sys.path.append('..')
from Detector.detect import create_mtcnn_net, MtcnnDetector

if __name__=="__main__":

    pnet =create_mtcnn_net(p_model='./model_path/Pnet_epoch_10.pt', use_cuda=True)

    mtcnn_detector = MtcnnDetector(pnet,rnet=None, onet=None, min_face_size=32,threshold=[0.4, 0.7, 0.7])
    img = cv2.imread('./test.jpg')
    img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    bboxs, landmarks = mtcnn_detector.detect_face(img,use_cuda=True)
    for bbox in bboxs:
        cv2.rectangle(img,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(255,0,0),1)
        cv2.imwrite("result.jpg",img)