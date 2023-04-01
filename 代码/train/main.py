import os
import os.path
from xml.etree.ElementTree import parse, Element
import cv2
import time

path = "./Annotations"
imgpath= "./Images"
files = os.listdir(path)
for xmlFile in files:
    xml_path = os.path.join(path, xmlFile)
    dom = parse(xml_path)
    root = dom.getroot()
    img_path = os.path.join(imgpath, root.find('filename').text)
    img = cv2.imread(img_path)
    for obj in root.iter('object'):
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        newimg = img[ymin:ymax, xmin:xmax]
        savepath = './' + obj.find('name').text + '/temp' + str(time.time())+'.png'
        cv2.imwrite(savepath, newimg)
        time.sleep(0.001)
