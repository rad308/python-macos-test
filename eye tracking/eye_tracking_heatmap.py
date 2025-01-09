import cv2
import mediapipe as mp
import numpy as np
from screeninfo import get_monitors
import time
from scipy.ndimage import gaussian_filter
from collections import deque
import threading
from filterpy.kalman import KalmanFilter
import pygetwindow as gw
from PIL import ImageGrab
import datetime


class EyeTrackingHeatmap:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5



        )



        # Video recording properties
        self.recording = None
        self.recording_thread = None
        self.is_recording = False
        self.output_filename = None
        self.frame_buffer = []
        self.fps = 60  # Set exact fps
        self.frame_time = 1/self.fps  # Calculate frame time
        self.last_frame_time = 0  # Track last frame time


        # Initialize webcam with higher resolution
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        
        # Monitor setup
        monitor = get_monitors()[0]
        self.screen_width = monitor.width
        self.screen_height = monitor.height

        # Initialize Kalman filter for each eye
        self.init_kalman_filters()

        # Eye landmarks indices
        self.LEFT_IRIS = [474, 475, 476, 477]
        self.RIGHT_IRIS = [469, 470, 471, 472]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

        # Calibration parameters
        self.calibration_points = []
        self.calibration_data = []
        self.calibration_matrix = None
        self.is_calibrated = False

        # Gaze smoothing
        self.gaze_history = deque(maxlen=10)
        self.smoothing_factor = 0.7

        # Initialize heatmap
        self.scale_factor = 0.25
        self.heatmap_width = int(self.screen_width * self.scale_factor)
        self.heatmap_height = int(self.screen_height * self.scale_factor)
        self.heatmap = np.zeros((self.heatmap_height, self.heatmap_width))
        self.decay_factor = 0.95
        self.gaussian_sigma = 20

        # Create single window
        #cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty('Display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)




    def start_recording(self):
        """Initialize video recording"""
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_filename = f"eye_tracking_session_{current_time}.mp4"
        
        # Get screen dimensions
        screen = ImageGrab.grab()
        width, height = screen.size
        
        # Initialize video writer with H264 codec
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'mp4v' if avc1 isn't available
        self.recording = cv2.VideoWriter(
            self.output_filename,
            fourcc,
            10,  # Reduced FPS to 10 for more stable recording
            (width, height)
        )
        self.is_recording = True
        self.frame_buffer = []
        self.last_frame_time = time.time()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_screen)
        self.recording_thread.start()



    def record_screen(self):
        """Record screen content with heatmap overlay"""
        frame_interval = 1.0 / 60 # 10 FPS
        next_frame_time = time.time()
        
        while self.is_recording:
            current_time = time.time()
            
            if current_time >= next_frame_time:
                try:
                    # Capture screen
                    screen = np.array(ImageGrab.grab())
                    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2BGR)
                    
                    # Get heatmap display
                    heatmap_display = self.display_heatmap()
                    
                    # Create mask for significant heatmap values
                    heatmap_gray = cv2.cvtColor(heatmap_display, cv2.COLOR_BGR2GRAY)
                    _, mask = cv2.threshold(heatmap_gray, 10, 255, cv2.THRESH_BINARY)
                    mask = cv2.merge([mask, mask, mask])
                    
                    # Apply overlay only where heatmap is significant
                    overlay = np.where(mask > 0, 
                                    cv2.addWeighted(screen, 0.5, heatmap_display, 0.5, 0),
                                    screen)
                    
                    # Write frame
                    self.recording.write(overlay)
                    
                    # Update next frame time
                    next_frame_time += frame_interval
                    
                    # If we're falling behind, reset the next frame time
                    if next_frame_time < time.time():
                        next_frame_time = time.time() + frame_interval
                    
                except Exception as e:
                    print(f"Error during recording: {e}")
                    continue
                
            # Small sleep to prevent CPU overuse
            time.sleep(max(0, min(next_frame_time - time.time(), frame_interval)))



    def stop_recording(self):
        """Stop recording and save video"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        if self.recording:
            self.recording.release()
        print(f"Recording saved as {self.output_filename}")




    def init_kalman_filters(self):
        # Initialize Kalman filters for x and y coordinates
        self.kf_x = KalmanFilter(dim_x=2, dim_z=1)
        self.kf_y = KalmanFilter(dim_x=2, dim_z=1)

        # Set initial state
        self.kf_x.x = np.array([0., 0.])
        self.kf_y.x = np.array([0., 0.])

        # State transition matrix
        self.kf_x.F = np.array([[1., 1.],
                               [0., 1.]])
        self.kf_y.F = np.array([[1., 1.],
                               [0., 1.]])

        # Measurement matrix
        self.kf_x.H = np.array([[1., 0.]])
        self.kf_y.H = np.array([[1., 0.]])

        # Measurement noise
        self.kf_x.R *= 0.1
        self.kf_y.R *= 0.1

        # Process noise
        self.kf_x.Q *= 0.1
        self.kf_y.Q *= 0.1

    def get_iris_position(self, frame, landmarks, eye_indices, iris_indices):
        """Get more precise iris position using iris landmarks"""
        iris_points = np.array([(landmarks.landmark[idx].x * frame.shape[1],
                               landmarks.landmark[idx].y * frame.shape[0])
                              for idx in iris_indices])
        
        iris_center = np.mean(iris_points, axis=0)
        eye_points = np.array([(landmarks.landmark[idx].x * frame.shape[1],
                              landmarks.landmark[idx].y * frame.shape[0])
                             for idx in eye_indices])
        
        eye_center = np.mean(eye_points, axis=0)
        
        # Calculate relative iris position
        eye_width = np.linalg.norm(eye_points[0] - eye_points[8])
        iris_pos_normalized = (iris_center - eye_center) / eye_width
        
        return iris_pos_normalized, iris_center, eye_center

    def process_eye_tracking(self, frame):
        """Process eye tracking with improved accuracy"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        landmarks = results.multi_face_landmarks[0]

        # Get iris positions for both eyes
        left_iris_pos, left_iris_center, left_eye_center = self.get_iris_position(
            frame, landmarks, self.LEFT_EYE, self.LEFT_IRIS)
        right_iris_pos, right_iris_center, right_eye_center = self.get_iris_position(
            frame, landmarks, self.RIGHT_EYE, self.RIGHT_IRIS)

        # Combine both eyes' data
        combined_iris_pos = (left_iris_pos + right_iris_pos) / 2

        # Apply Kalman filtering
        self.kf_x.predict()
        self.kf_y.predict()
        self.kf_x.update(combined_iris_pos[0])
        self.kf_y.update(combined_iris_pos[1])

        # Get filtered positions
        filtered_x = self.kf_x.x[0]
        filtered_y = self.kf_y.x[0]

        return np.array([filtered_x, filtered_y])

    def update_heatmap(self, x, y):
        """Update heatmap with new gaze point"""
        # Scale coordinates to heatmap size
        hx = int(x * self.scale_factor)
        hy = int(y * self.scale_factor)
        
        if 0 <= hx < self.heatmap_width and 0 <= hy < self.heatmap_height:
            # Apply decay to existing heatmap
            self.heatmap *= self.decay_factor
            
            # Add new point
            self.heatmap[hy, hx] += 1.0
            
            # Apply Gaussian blur
            self.heatmap = gaussian_filter(self.heatmap, sigma=self.gaussian_sigma)



    def display_heatmap(self):
        """Display the heatmap with reduced flickering"""
        # Normalize heatmap
        normalized_heatmap = (self.heatmap - self.heatmap.min()) / (self.heatmap.max() - self.heatmap.min() + 1e-8)
        
        # Apply smoothing to reduce flickering
        normalized_heatmap = cv2.GaussianBlur(normalized_heatmap, (15, 15), 0)
        
        # Apply power function to enhance visibility
        enhanced_heatmap = np.power(normalized_heatmap, 0.7)
        
        # Convert to color map
        heatmap_color = cv2.applyColorMap((enhanced_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Resize to screen size
        heatmap_display = cv2.resize(heatmap_color, (self.screen_width, self.screen_height))
        
        return heatmap_display


    def cleanup(self):
        """Ensure proper cleanup of resources"""
        self.is_recording = False
        if self.recording_thread:
            self.recording_thread.join()
        if self.recording:
            self.recording.release()
        self.cap.release()
        cv2.destroyAllWindows()


    def calibrate(self):
        """Improved calibration process"""
        print("Starting calibration...")
        calibration_points = [
            (0.1, 0.1), (0.5, 0.1), (0.9, 0.1),
            (0.1, 0.5), (0.5, 0.5), (0.9, 0.5),
            (0.1, 0.9), (0.5, 0.9), (0.9, 0.9)
        ]

        # Create fullscreen window for calibration
        cv2.namedWindow('Calibration', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('Calibration', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        self.calibration_points = []
        self.calibration_data = []

        for point_idx, (px, py) in enumerate(calibration_points):
            screen_x = int(px * self.screen_width)
            screen_y = int(py * self.screen_height)

            # Display calibration point on full screen
            calibration_screen = np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
            cv2.circle(calibration_screen, (screen_x, screen_y), 20, (0, 255, 0), -1)
            
            # Add text showing progress
            text = f"Look at the green dot ({point_idx + 1}/9)"
            cv2.putText(calibration_screen, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Calibration', calibration_screen)
            cv2.waitKey(1)

            # Wait for a moment before collecting data
            time.sleep(1)

            # Collect multiple samples for each point
            point_data = []
            for _ in range(60):  # Collect 30 samples per point
                ret, frame = self.cap.read()
                if ret:
                    gaze_pos = self.process_eye_tracking(frame)
                    if gaze_pos is not None:
                        point_data.append(gaze_pos)
                #time.sleep(0.1)  # ~30fps

            if point_data:
                avg_gaze = np.mean(point_data, axis=0)
                self.calibration_points.append([px, py])
                self.calibration_data.append(avg_gaze)

        # Calculate calibration matrix
        if len(self.calibration_points) >= 6:
            self.calibration_matrix = cv2.findHomography(
                np.array(self.calibration_data),
                np.array(self.calibration_points)
            )[0]
            self.is_calibrated = True
            print("Calibration completed successfully!")
        else:
            print("Calibration failed! Not enough points collected.")

        # Close calibration window
        cv2.destroyWindow('Calibration')


    def get_gaze_point(self, gaze_pos):
        """Convert gaze position to screen coordinates"""
        if not self.is_calibrated or gaze_pos is None:
            return None

        # Apply calibration transformation
        gaze_point = cv2.perspectiveTransform(
            gaze_pos.reshape(-1, 1, 2),
            self.calibration_matrix
        ).reshape(-1)

        # Apply smoothing
        self.gaze_history.append(gaze_point)
        smoothed_gaze = np.mean(self.gaze_history, axis=0)

        # Convert to screen coordinates
        screen_x = int(smoothed_gaze[0] * self.screen_width)
        screen_y = int(smoothed_gaze[1] * self.screen_height)

        return screen_x, screen_y
    


    def run(self):
        """Main loop with background tracking and recording"""
        try:
            # Perform calibration first
            self.calibrate()

            print("Calibration complete. Eye tracking is running in the background.")
            print("Press Ctrl+C in the terminal to stop recording and quit.")

            # Start recording
            self.start_recording()

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to get frame from camera")
                    break

                gaze_pos = self.process_eye_tracking(frame)
                if gaze_pos is not None and self.is_calibrated:
                    screen_point = self.get_gaze_point(gaze_pos)
                    if screen_point:
                        self.update_heatmap(screen_point[0], screen_point[1])

                time.sleep(0.01)  # Small sleep to reduce CPU usage

        except KeyboardInterrupt:
            print("\nPreparing to stop recording...")
            time.sleep(1)
            print("Stopping recording and cleaning up...")
        finally:
            self.cleanup()




if __name__ == "__main__":
    tracker = EyeTrackingHeatmap()
    tracker.run()
