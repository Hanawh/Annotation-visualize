import os
import xml.etree.ElementTree as ET
from matplotlib import colors
import cv2
import numpy as np
import webcolors
import random
# please pre-defines before u use this script

PRE_DEFINE_CATEGORIES =  ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                        'bus', 'car','cat', 'chair', 'cow', 'diningtable',
                        'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
                        'sheep', 'sofa', 'train','tvmonitor']

STANDARD_COLORS = [
    'LawnGreen', 'Chartreuse', 'Aqua','Beige', 'Azure','BlanchedAlmond','Bisque',
    'Aquamarine', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'AliceBlue', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def from_colorname_to_bgr(color):
    rgb_color=webcolors.name_to_rgb(color)
    result=(rgb_color.blue,rgb_color.green,rgb_color.red)
    return result

# process the custom dataset json file which is coco fomat
class Dataset():
    def __init__(self, xml_dir, image_dir, output_dir):
        self.xml_dir = xml_dir
        self.image_dir = image_dir
        self.output_dir = output_dir
        self.cat2label = {cat: i+1  for i, cat in enumerate(PRE_DEFINE_CATEGORIES)} 
        self.image_ids = os.listdir(xml_dir)
    
    def len(self):
        return len(self.image_ids)

    def getitem(self, idx):
        img, file_name = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        return sample, file_name
    
    def load_image(self, image_index):
        image_info = self.image_ids[image_index].replace('xml','jpg')
        path = os.path.join(self.image_dir, image_info)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        return img, image_info
    
    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.image_ids[image_index]
        xml_path = os.path.join(self.xml_dir, annotations_ids)
        annotations = np.zeros((0, 5))
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            difficult = int(obj.find('difficult').text) # useless in this script
            bnd_box = obj.find('bndbox')
            bbox = [
                int(bnd_box.find('xmin').text),
                int(bnd_box.find('ymin').text),
                int(bnd_box.find('xmax').text),
                int(bnd_box.find('ymax').text)
            ]
            annotation = np.zeros((1, 5))
            annotation[0, :4] = bbox
            annotation[0, 4] = self.cat2label[name] - 1
            annotations = np.append(annotations, annotation, axis=0)
        return annotations
    
    def drawBox(self, image_index):
        sample, file_name = self.getitem(image_index)
        img, annots = sample['img'], sample['annot']
        for annot in annots:
            x1, y1, x2, y2 = annot[:4]
            class_name = PRE_DEFINE_CATEGORIES[int(annot[4])]
            self.plot_one_box(img, [x1, y1, x2, y2], label=class_name, color=from_colorname_to_bgr(STANDARD_COLORS[int(annot[4])]))
        cv2.imwrite(f'{self.output_dir}/{file_name}', img)

    
    def plot_one_box(self, img, coord, label, color):
        tl = int(round(0.001 * max(img.shape[0:2]))) # box font thickness
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        
        tf = max(tl - 2, 1)  # class font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0]+15, c1[1] - t_size[1] -3
        cv2.rectangle(img, c1, c2 , color, -1)  # filled
        cv2.putText(img, '{}'.format(label), (c1[0],c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.FONT_HERSHEY_SIMPLEX)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Visualize COCO format annotation."
    )
    parser.add_argument("xml_root", help="Directory path to xml.", type=str)
    parser.add_argument("image_root", help="Directory path to images.", type=str)
    parser.add_argument("random_number", help="How many imgs randomly selected.", type=int)
    parser.add_argument("--output_dir", default="visualize/", help="Directory path to store images.", type=str)
    args = parser.parse_args()

    mydata = Dataset(args.xml_root, args.image_root, args.output_dir)
    if os.path.exists(args.output_dir) is None:
        os.makedirs(args.output_dir)
    
    length = mydata.len()
    index = random.sample(list(np.arange(0, length)), args.random_number)
    for ind in index:
        mydata.drawBox(ind)

    

