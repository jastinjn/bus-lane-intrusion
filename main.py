from inference.core.interfaces.camera.entities import VideoFrame
from inference import InferencePipeline
from typing import Union, List, Optional, Any
from detector import VehicleIntrusionDetector
import cv2
from dotenv import load_dotenv
import sys
import os
load_dotenv()

out_fps = 5
detector = VehicleIntrusionDetector(max_fps=out_fps)

def on_video_frame(video_frames: List[VideoFrame]) -> List[Any]:
  images = [v.image for v in video_frames]

  result = detector.infer(images)

  return result

def on_prediction(predictions: Union[dict, List[Optional[dict]]], video_frame: Union[VideoFrame, List[Optional[VideoFrame]]]) -> None:
  if not issubclass(type(predictions), list):
      # this is required to support both sequential and batch processing with single code
      # if you use only one mode - you may create function that handles with only one type
      # of input
      predictions = [predictions]
      video_frame = [video_frame]

  for result, frame in zip(predictions, video_frame):
    annotated_image, tracked_objects = detector.process_inference(frame.image, result[0], result[1])
    print(tracked_objects)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output",annotated_image)
    cv2.waitKey(1)

input_file_path = sys.argv[1]
    

if input_file_path.lower().endswith(('.jpg')) or input_file_path.lower().endswith(('.png')):
    image = cv2.imread(input_file_path)
    result = detector.infer(image)[0]
    annotated_image, tracked_objects = detector.process_inference(image, result[0], result[1])
    print(tracked_objects)
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.imshow("output", annotated_image)
    cv2.waitKey(0)
    file_name = os.path.splitext(os.path.basename(input_file_path))[0]
    cv2.imwrite(f'output/{file_name}.out.jpg', annotated_image)

elif input_file_path.lower().endswith(('.mp4')):

    pipeline = InferencePipeline.init_with_custom_logic(
    max_fps=out_fps,
    video_reference=input_file_path,
    on_video_frame=on_video_frame,
    on_prediction=on_prediction,
    )

    # start the pipeline
    pipeline.start()
    # wait for the pipeline to finish
    pipeline.join()

