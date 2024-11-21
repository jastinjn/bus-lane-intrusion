from inference import get_model
import supervision as sv
import numpy as np

def filter_detection_by_mask(detection, mask):

  num_objects = len(detection)
  filter_mask = np.empty(num_objects, dtype='bool')
  for idx in range(num_objects):
    bbox = detection.xyxy[idx].astype(int)
    filter_mask[idx] = np.any(mask[bbox[1]:bbox[3], bbox[0]:bbox[2]])

  return detection[filter_mask]

def filter_detection_in_zone(detection, mask, thresh=0.5):

  num_objects = len(detection)
  filter_mask = np.empty(num_objects, dtype='bool')
  for idx in range(num_objects):
    object_center_x = int((detection.xyxy[idx][0] + detection.xyxy[idx][2])/2)
    object_bottom_y = int(detection.xyxy[idx][3])

    non_zero_indices = np.nonzero(mask[object_bottom_y])[0]

    if non_zero_indices.shape[0]:
      zone_width = non_zero_indices[-1] - non_zero_indices[0]
      zone_center = (non_zero_indices[-1] + non_zero_indices[0]) // 2

      filter_mask[idx] = np.abs(zone_center-object_center_x) < (thresh * zone_width)

  return detection[filter_mask]

def license_from_ocr(ocr_result, conf_thresh=0.5):
  license_number = ""

  for pred in ocr_result:
    if pred[2] < conf_thresh:
      return ""
    license_number = license_number + pred[1]

  license_number = license_number.replace(' ','')
  license_number = license_number.upper()

  return license_number

class RoboflowSegmentationModel:
  def __init__(self, model_name):
    self.model = get_model(model_name)
    self.mask_annotator = sv.MaskAnnotator(opacity=0.25)
    self.label_annotator = sv.LabelAnnotator(text_position=sv.Position.BOTTOM_CENTER)

  def predict(self, image, verbose=False):
    return self.model.infer(image)

  def get_detection(self, prediction):
    return sv.Detections.from_inference(prediction)

  def annotate_image(self, img, detections):

    labels_road = [
        f"{class_id} {confidence:.2f}"
        for class_id, confidence in zip(detections.data["class_name"], detections.confidence)
    ]

    annotated_image = self.mask_annotator.annotate(scene=img, detections=detections)
    annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels_road)

    return annotated_image

class RoboflowObjectDetectionModel:
  def __init__(self, model_name):
    self.model = get_model(model_name)
    self.bounding_box_annotator = sv.BoxAnnotator()
    self.label_annotator = sv.LabelAnnotator(text_scale=1, text_position=sv.Position.TOP_CENTER)
    self.label_annotator_bottom = sv.LabelAnnotator(text_scale=1, text_position=sv.Position.BOTTOM_CENTER)

  def predict(self, image, verbose=False):
    return self.model.infer(image)

  def get_detection(self, prediction):
    return sv.Detections.from_inference(prediction)

  def annotate_image(self, img, detections, bottom_labels=None):

    labels_objs = [ f"{class_id} {confidence:.2f}" for class_id, confidence in zip(detections.data["class_name"], detections.confidence) ]

    annotated_image = self.bounding_box_annotator.annotate(scene=img, detections=detections)
    annotated_image = self.label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels_objs)

    if bottom_labels is not None:
      annotated_image = self.label_annotator_bottom.annotate(scene=annotated_image, detections=detections, labels=bottom_labels)

    return annotated_image

