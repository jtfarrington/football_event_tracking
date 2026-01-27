import cv2


def read_video(video_path):
    """
    Read a video file and extract all frames.
    
    Loads the entire video into memory as a list of frames.
    Useful for processing videos frame-by-frame.
    
    Args:
        video_path: Path to the video file (e.g., 'input_videos/match.mp4')
    
    Returns:
        List of frames (each frame is a NumPy array in BGR format)
    
    Example:
        frames = read_video('input_videos/match.mp4')
        print(f"Loaded {len(frames)} frames")
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    # Read frames until end of video
    while True:
        # ret = success boolean, frame = image data
        ret, frame = cap.read()
        
        # Break when no more frames (end of video)
        if not ret:
            break
            
        frames.append(frame)
    
    # Clean up
    cap.release()
    
    return frames


def save_video(output_video_frames, output_video_path):
    """
    Save a list of frames as a video file.
    
    Creates an AVI video file from a sequence of frames.
    Frame rate is set to 24 fps.
    
    Args:
        output_video_frames: List of frames (NumPy arrays in BGR format)
        output_video_path: Where to save the video (e.g., 'output_videos/result.avi')
    
    Example:
        save_video(annotated_frames, 'output_videos/match_analyzed.avi')
    """
    # Define video codec (XVID for AVI format)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # Get frame dimensions from first frame
    frame_height = output_video_frames[0].shape[0]
    frame_width = output_video_frames[0].shape[1]
    
    # Create video writer
    # Parameters: output path, codec, fps, (width, height)
    out = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        24,  # 24 frames per second
        (frame_width, frame_height)
    )
    
    # Write each frame to the video file
    for frame in output_video_frames:
        out.write(frame)
    
    # Finalize and save the video file
    out.release()
