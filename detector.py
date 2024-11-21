import easyocr
from utils import RoboflowObjectDetectionModel, RoboflowSegmentationModel, filter_detection_by_mask, filter_detection_in_zone, license_from_ocr
import supervision as sv
import numpy as np

class VehicleIntrusionDetector:
  def __init__(self, max_fps=30):
    self.vehicle_model = RoboflowObjectDetectionModel("yolov8n-640")
    self.plate_model = RoboflowObjectDetectionModel("license-plate-recognition-rxg4e/6")
    self.lane_model = RoboflowSegmentationModel("sg-bus-lanes/2")
    self.reader = easyocr.Reader(['en'])
    self.allowlist = "ABCDEFGHJKLMNPQRSTUVWXYZ1234567890"

    self.tracker = sv.ByteTrack()
    self.tracker.reset()
    self.tracked_objects = {}
    self.fps = max_fps

  # public methods
  def infer(self, image):
    vehicle_results = self.vehicle_model.predict(image)
    lane_results = self.lane_model.predict(image)

    return list(zip(vehicle_results, lane_results))

  def process_inference(self, image, vehicle_results, lane_results):
    vehicle_detections, plate_detections, license_numbers, lane_detections = self.__get_detections(image, vehicle_results, lane_results)
    vehicle_detections = self.__track_detections(vehicle_detections, license_numbers)
    annotated_image = self.__annotate_frame(image, vehicle_detections, plate_detections, license_numbers, lane_detections)

    return annotated_image, self.tracked_objects

  ## private methods
  def __get_detections(self, image, vehicle_results, lane_results):

    vehicle_detections = self.vehicle_model.get_detection(vehicle_results)
    lane_detections = self.lane_model.get_detection(lane_results)

    # filter vehicle detection
    vehicle_mask = np.isin(vehicle_detections.class_id, [2, 5, 7])
    vehicle_detections.class_id[vehicle_mask] = 2
    vehicle_detections.data["class_name"][vehicle_mask] = "car"
    vehicle_detections = vehicle_detections[vehicle_mask]
    vehicle_detections = vehicle_detections[vehicle_detections.confidence > 0.25]

    # filter lane detections to only consider bus lane
    lane_detections = lane_detections[lane_detections.class_id == 0]

    # filter vehicles by bus lane region
    lane_mask = lane_detections.mask
    if lane_mask is None or lane_mask.shape[0] == 0:
      return vehicle_detections[vehicle_detections.class_id == -1], [], [], lane_detections

    lane_mask = np.any(lane_mask, axis=0)

    vehicle_detections = filter_detection_by_mask(vehicle_detections, lane_mask)
    vehicle_detections = filter_detection_in_zone(vehicle_detections, lane_mask)

    # detect license plates and numbers
    plate_detections, license_numbers = self.__extract_license_plates(image, vehicle_detections)

    return vehicle_detections, plate_detections, license_numbers, lane_detections

  def __extract_license_plates(self, image, vehicle_detections):
    plate_detections = []
    license_numbers = []
    for idx in range(vehicle_detections.xyxy.shape[0]):
      # detect license plate bounding box
      bbox = vehicle_detections.xyxy[idx].astype(int)
      vehicle_roi = image[bbox[1]:bbox[3],bbox[0]:bbox[2]]

      plate_results = self.plate_model.predict(vehicle_roi)[0]
      plate_detection = self.plate_model.get_detection(plate_results)

      plate_detection = plate_detection[plate_detection.confidence > 0.25]
      plate_detection.xyxy[:,[0,2]] += bbox[0]
      plate_detection.xyxy[:,[1,3]] += bbox[1]

      if len(plate_detection) <= 1:
        plate_detections.append(plate_detection)
      else:
        plate_detections.append(plate_detection[0])

      # extract license plate number with ocr
      license_number = ""
      if len(plate_detection):
        plate_bbox = plate_detection.xyxy[0].astype(int)
        plate_roi = image[plate_bbox[1]:plate_bbox[3],plate_bbox[0]:plate_bbox[2]]

        min_pixel_width = 60
        if plate_roi.shape[1] > min_pixel_width:
          ocr_result = self.reader.readtext(plate_roi, allowlist=self.allowlist)
          license_number = license_from_ocr(ocr_result, conf_thresh=0.75)

      license_numbers.append(license_number)

    plate_detections = sv.Detections.merge(plate_detections)

    return plate_detections, license_numbers

  def __track_detections(self, detections, license_numbers):
    detections = self.tracker.update_with_detections(detections)

    # remove old detections
    max_misses = self.fps
    expired_ids = []
    for id in self.tracked_objects:
      self.tracked_objects[id][1] += 1
      if self.tracked_objects[id][1] > max_misses:
        expired_ids.append(id)
    for id in expired_ids:
      self.tracked_objects.pop(id)

    # add or modify based on current detections
    for id, license_number in zip(detections.tracker_id, license_numbers):
      if id in self.tracked_objects:
        self.tracked_objects[id][0] += 1
        self.tracked_objects[id][1] = 0
        if license_number != "":
          self.tracked_objects[id][2] = license_number
      else:
        self.tracked_objects[id] = [1, 0, license_number]

    return detections

  def __annotate_frame(self, image, vehicle_detections, plate_detections, license_numbers, lane_detections):
    annotated_image = image.copy()

    if len(vehicle_detections):
      labels = []
      for id in list(vehicle_detections.tracker_id):
        time_tracked = self.tracked_objects[id][0]/self.fps
        labels.append(f"id:{id} vn:{self.tracked_objects[id][2]} time:{time_tracked:.2f}s")

      annotated_image = self.vehicle_model.annotate_image(annotated_image, vehicle_detections, labels)

    if len(plate_detections):
      annotated_image = self.plate_model.annotate_image(annotated_image, plate_detections, license_numbers)

    if len(lane_detections):
      annotated_image = self.lane_model.annotate_image(annotated_image, lane_detections)

    return annotated_image
