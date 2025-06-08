# Face Mesh with OpenCV and Mediapipe

This project demonstrates real-time face mesh detection using [OpenCV](https://opencv.org/) and [Mediapipe](https://mediapipe.dev/) in Python. It detects facial landmarks from your webcam feed or video files and can be extended for applications like facial analysis, AR filters, or emotion recognition.

## Features

- Real-time face mesh (468 landmarks) detection and drawing
- Works with both webcam and video file input
- FPS display on video feed
- Modular, class-based code structure
- Easily extract and use specific face landmarks

## Requirements

- Python 3.7+
- opencv-python
- mediapipe

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Usage

Run the main script:

```sh
python FaceMeshModule.py
```

- Press `q` to quit the video window.
- The script prints the position of selected face mesh landmarks (e.g., landmark 0) in the console.

## Customization

- To use a different camera, change `camera_index` in `main()` or the class constructor.
- To use a video file, pass `video_path="your_video.mp4"` to the class.

## File Structure

- `FaceMesh.py` – Simple script for face mesh detection.
- `FaceMeshModule.py` – Modular, class-based version with more features.
- `requirements.txt` – Python dependencies.
- `FaceVideos/` – Folder with test videos.
- `Screenshot.png` – Example output screenshot.

## Screenshot

![Face Mesh Screenshot](https://github.com/PrabeshPathak2002/FaceMesh-using-opencv-and-mediapipe/blob/main/Screenshot.png "Screenshot")

## Landmarks

![Face Mesh Landmarks](https://github.com/PrabeshPathak2002/FaceMesh-using-opencv-and-mediapipe/blob/main/Landmarks.png "Landmarks")

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
