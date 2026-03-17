import cv2
from ultralytics import YOLO

def process_video(input_video_path, output_video_path=None):
    model = YOLO(r"") # # Load the trained YOLOv8 model (pretrained model included in runs/detect/train/weights/best.pt)

    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {input_video_path}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for result in results:
            annotated_frame = result.plot()

        cv2.imshow('Trash Detection', annotated_frame)

        if output_video_path:
            out.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if output_video_path:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    input_video_path = r"" # Path to the input video file
    output_video_path = 'output_video.mp4'  # Path to save the output video file. one file existing to show the output in root directory

    process_video(input_video_path, output_video_path)
