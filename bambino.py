""" Bambino is a birth record handwriting box extraction tool.

This program provides a framework for extracting desired handwriting boxes
from ancient italian birth records. The program functions by classifying 
handwriting boxes by classifying their preceding keywords based on 
geometric positioning.

Typical usage example:

    # Use Bambino to return a virtual spreadsheet in 'out/out.html'
    bambi = Bambino(record_images, ground_truth_csv, model_path)
    bambi.run()

"""

from ultralytics import YOLO    # Yolo CNN Model
import numpy as np              # Numeric Python Package
import matplotlib.pyplot as plt # Plotting Package
import cv2                      # Open CV2 Package
import pandas as pd             # Pandas Data Package


class Bambino:
    """Bambino is a birth record handwriting box extraction tool."""
    keywords_csv = None
    model_path = None
    ground_truth_csv = None
    record_datas = []
    record_results = []
    record_images = []

    def __init__(self, record_images, ground_truth_csv, model_path):
        self.model_path = model_path
        self.ground_truth_csv = ground_truth_csv
        # If single image, convert to list
        for image in record_images:
            self.record_images.append(image)

    def run(self, overlap_threshold=15):
        """ Run bambino on image set.
        
        Outputs all files to out/ directory.
            - bbox0.jpg ... bboxN.jpg
            - out.csv
            - out.html
            - bboxes.csv
        """
        print("Bambino!")
        self.extract_bboxes(confidence=0.01, output=True)
        self.associate_keywords(threshold=overlap_threshold, output=True)
        self.generate_cropped()
        self.gen_virtual_spreadsheets(input_image=self.record_images[0], input_csv="out/out.csv", output_file="out/out")
    
    
    def extract_bboxes(self, confidence=0.5, output=False, output_dir=None):
        """ Extract bounding boxes from record images using pre-trained model.
        
        Outputs all files to out/ directory.
            - bboxes.csv
        """
        # Run model inference on image
        model = YOLO(self.model_path)
        for image in self.record_images:
            self.record_results.append(model.predict(image, conf=confidence, save=False, save_crop=False, show_labels=True))

        # Compile bboxes into list
        data = []
        for record_result in self.record_results:
            for result in record_result:
                for boxes in result.boxes:
                    box = boxes.xyxy[0].tolist()
                    cls = boxes.cls
                    conf = boxes.conf
                    cls_name = model.names[int(cls)]
                    x1 = int(box[0])
                    y1 = int(box[1])
                    x2 = int(box[2])
                    y2 = int(box[3])
                    data.append([cls_name, conf.item(), "NULL", 0,x1, y1, x2, y2])

        # Store as member variable
        data = pd.DataFrame(data, columns=['class', 'confidence', 'keyword', 'overlap_perc', 'x1', 'y1', 'x2', 'y2'])
        self.record_datas.append(data)
        if output:
            data.to_csv(output_dir, index=0)

    def associate_keywords(self, threshold, output=False):
        """ Associate bounding boxes from csv file by geometric overlap percent.
        
        Inputs from out/ directory.
            - bboxes.csv

        Outputs all files to out/ directory.
            - bboxes.csv
        """
        # Read in truth csv
        truth_bboxes = pd.read_csv(self.ground_truth_csv)

        # Loop through truths, compare each bbox with record bbox for overlap
        keyword_count = 0
        for index_record in range(len(self.record_datas)):
            record_data = self.record_datas[index_record]
            data = []
            for index_truth, row_truth in truth_bboxes.iterrows():
                keyword = row_truth['class']
                truth_x1 = row_truth['x1']
                truth_y1 = row_truth['y1']
                truth_x2 = row_truth['x2']
                truth_y2 = row_truth['y2']

                for idx, row in record_data.iterrows():
                    if(row['class']!='Text'):
                        bbox_x1 = row['x1']
                        bbox_y1 = row['y1']
                        bbox_x2 = row['x2']
                        bbox_y2 = row['y2']
                        overlap_percent = self.intersection_percent(truth_x1, truth_y1, truth_x2, truth_y2, bbox_x1, bbox_y1, bbox_x2, bbox_y2)

                        if overlap_percent > threshold:
                            keyword_count += 1
                            data.append([row['class'], row['confidence'], keyword, overlap_percent, row['x1'], row['y1'], row['x2'], row['y2']])

            record_data = pd.DataFrame(data, columns=['class', 'confidence', 'keyword', 'overlap_perc', 'x1', 'y1', 'x2', 'y2'])
            self.record_datas[index_record] = record_data

            # Write bboxes to .csv
            if output:
                self.record_datas[index_record].to_csv("out/bboxes.csv", index=0)

        print("\n\nDetected", keyword_count, "keywords of interest in image batch.")

    @staticmethod
    def intersection_percent(x1, y1, x2, y2, x3, y3, x4, y4):
        """ Static method for calculating rectangle overlap percent."""
        intersection_x1 = max(x1, x3)
        intersection_y1 = max(y1, y3)
        intersection_x2 = min(x2, x4)
        intersection_y2 = min(y2, y4)

        if intersection_x1 < intersection_x2 and intersection_y1 < intersection_y2:
            width = intersection_x2 - intersection_x1
            height = intersection_y2 - intersection_y1
            area = width * height
            overlap_percentage = (area / ((x2 - x1) * (y2 - y1))) * 100
            return overlap_percentage
        else:
            return 0

    def generate_cropped(self):
        """ Generates cropped handwriting images from record images.
    
        Outputs all files to out/ directory.
            - bbox0.jpg ... bboxN.jpg
            - out.csv
        """
        data = pd.read_csv("out/bboxes.csv")
        img = cv2.imread(self.record_images[0])

        out_data = []

        intdex = 0
        print("\nKeyword\t| Overlap %")
        print("-------------------")
        for(index, row) in data.iterrows():
            cls = row['keyword']
            conf = row['confidence']
            overlap_per = row['overlap_perc']

            x1 = row['x1']
            y1 = row['y1']
            x2 = row['x2']
            y2 = row['y2']

            # Crop Bounding Box
            crop_img = img[y1:y2, x1:x2]

            print(row['keyword'] + "\t| " + str("{:.3f}".format(overlap_per)))
            image_name =  "bbox" + str(index) + ".jpg"
            image_path = "out/" + image_name
            image_filename = image_path
            cv2.imwrite(image_filename, crop_img)

            out_data.append([cls, overlap_per, image_name]);

        out_data = pd.DataFrame(out_data, columns=['keyword', 'overlap_perc', 'image'])
        out_data.to_csv("out/out.csv")
        index = 0
        index += 1

    def gen_virtual_spreadsheets(self, input_image, input_csv, output_file=None):
        """ Generates cropped handwriting images from record images.
    
        Inputs from out/ directory.
            - out.csv

        Outputs all files to out/ directory.
            - out.html
        """

        data = pd.read_csv(input_csv);

        header = "<!DOCTYPE html><html><head><style>table {font-family: arial, sans-serif;border-collapse: collapse;width: 100%;}td, th {border: 1px solid #dddddd; text-align: left; padding: 8px;}tr:nth-child(even) {background-color: #dddddd;}</style></head><body>"

        title ="<h2>Virtual Spreadsheet for " + input_image +"</h2>"

        table_header = "<table><tr><th>Keyword</th><th>Handwriting</th><th>Overlap Percent</th></tr>"

        rows = ""
        for index, row in data.iterrows():
            keyword = row['keyword']
            image_path = row['image']
            overlap = row['overlap_perc']
            row = "<tr><td>" + keyword +"</td><td><img src=\"" + image_path + "\"></td><td>" + str(overlap) +"</td></tr>"
            rows += row

        footer = "</table></body></html>"

        html = header + title + table_header + rows + footer
        with open(output_file+".html", "w") as file:
            file.write(html)

    """ NON-FUNCTIONAL, STILL IMPLEMENTING
    def visualize_nearest_handwriting_bbox(self, image, csv, location=0):

        pairs = pd.read_csv(csv)

        img = cv2.imread(image)

        for index, row in pairs.iterrows():
            keyword = row['keyword']
            keyword_x1 = row['keyword_x1']
            keyword_y1 = row['keyword_y1']
            keyword_x2 = row['keyword_x2']
            keyword_y2 = row['keyword_y2']
            handwriting_x1 = row['handwriting_x1']
            handwriting_y1 = row['handwriting_y1']
            handwriting_x2 = row['handwriting_x2']
            handwriting_y2 = row['handwriting_y2']

            keyword_crop = img[keyword_y1:keyword_y2, keyword_x1:keyword_x2]
            handwriting_crop = img[handwriting_y1:handwriting_y2, handwriting_x1:handwriting_x2]

            print(keyword)
            plt.figure().set_figheight(2)
            plt.subplot(1, 2, 1)
            plt.axis('off')
            plt.imshow(keyword_crop[:,:,::-1])
            plt.subplot(1, 2, 2)
            plt.imshow(handwriting_crop[:,:,::-1])
            plt.axis('off')
            plt.show()
    def gen_composite(self, image_path, bbox_csv_path, class_select=None, colors=False, view_composite=False, view_boxes=False):

        # Generate composite image canvas
        img = cv2.imread(image_path)
        canvas = np.ones((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 255

        # Color Code
        if(cls.item() == 0 and colors == True):
            clr_img  = np.full((crop_img.shape[0] ,crop_img.shape[1] ,3), (255,0,0), np.uint8)
            crop_img  = cv2.addWeighted(crop_img, 0.8, clr_img, 0.2, 0)
        elif(cls.item() == 1 and colors == True):
            clr_img  = np.full((crop_img.shape[0] ,crop_img.shape[1] ,3), (0,255,0), np.uint8)
            crop_img  = cv2.addWeighted(crop_img, 0.8, clr_img, 0.2, 0)

        # Crop Bounding Box
        crop_img = img[y1:y2, x1:x2]

        # Class Select
        if(cls.item() == class_select and class_select != None or class_select == None):
            canvas[y1:y2, x1:x2] = crop_img
            data.append([cls_name, conf.item(), x1, y1, x2, y2])

            # Plot Bounding Boxes
            if view_bbox:
                plt.figure()
                plt.imshow(crop_img[:,:,::-1])
                plt.axis('off')
                plt.show()


            # Plot Composite
            if view_composite:
                fig = plt.figure()
                plt.imshow(canvas[:,:,::-1])
                plt.axis('off')
                plt.show()

            #(w, h) = crop_img.shape[:2]

            #aspect_ratio = w/h

            #crop_img_resize = cv2.resize(crop_img, (int(100/aspect_ratio), 100))

            #data.append([keyword, area, img_x1, img_y1, img_x2, img_y2])

            #plt.figure().set_figheight(2)
            #plt.imshow(crop_img_resize[:,:,::-1])
            #plt.axis('off')
            #plt.show()
    """    