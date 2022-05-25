import numpy as np
import PIL
from PIL import Image
import os

image_list = []

"""
    def __init__(self, data_filename="datasettt"):
        self.labels = ["apple", "pear", "neither"]

        # Data should be stored in /home/username/data/
        self.path = os.path.join(os.environ["HOME"], "data")
        self.roi_classification_size = 28
        self.collected_ROIs = []

        self.data_filename = data_filename

        fake_image = np.zeros((self.roi_classification_size, self.roi_classification_size, 3))
        self.model.predict(np.asarray([fake_image]))

    def resize(self, rois):
        new_rois = []
        for roi in rois:
            new_rois.append(cv2.resize(roi, (self.roi_classification_size, self.roi_classification_size)))
        return np.array(new_rois)

    # @TODO by student
    # Update function to get all the ROIs, should return either 1 ROI or the label of the ROI
    def update(self, image):
        # Convert the image to HSV.
        hsv_image = self.to_hsv(image)

        # Filter the HSV image such that it contains only the red signs.
        filtered_image = self.filter_hsv(hsv_image)

        # Post process the image to remove noise.
        processed_image = self.post_process(filtered_image)

        # Get the ROIs and optionally the largest ROI index.
        rois = self.get_rois(processed_image, image)

        # Filter the ROIs based on their width and height
        filtered_rois = self.filter_rois(rois)

        # Resize all the filtered ROIs to a constant size
        final_rois = self.resize(filtered_rois)

        if self.data_collection_mode == True:
            # Save the ROIs in the list self.collected_ROIs (so they can be saved when you close the program)
            for roi in final_rois:
                self.collected_ROIs.append(roi)

        if self.data_collection_mode == False:
            for roi in final_rois:
                # cv2.imshow("sign", roi)
                # cv2.waitKey(1)
                label = self.predict(roi)
                # Return either 1 ROI, or return the label of that ROI
                return label
        return 5
"""

def save_data(filename):
    data = image_list
    project_root = os.path.dirname(os.path.dirname(__file__))
    output_path = os.path.join(project_root, filename)

    # Check if path exists, else create it.
    if not os.path.exists("trainingdata"):
        os.mkdir("trainingdata")

    # Save the numpy array in /home/student/data/
    np.save(os.path.join("trainingdata", filename + ".npy".format(filename)), data)
    print("Data has been saved in " + "trainingdata/" + filename)

if __name__ == '__main__':
    filename = input("Please specify the filename for the training data: ")
    image = Image.open("images/trebekthumb.jpg")
    resized_image = image.resize((100, 100))
    image_array = np.asarray(resized_image)
    image_list.append(image_array)
    save_data(filename)
