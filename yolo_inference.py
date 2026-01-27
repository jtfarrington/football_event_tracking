"""
YOLO Model Inference Test Script

Quick test to verify the YOLO model is working correctly.
Runs detection on a single video and displays results.

Use this to:
- Test if your model loads properly
- Check what objects are being detected
- Verify confidence scores
- Debug detection issues
"""

from ultralytics import YOLO 


def main():
    """Run YOLO detection on a test video and display results."""
    
    print("Loading YOLO model...")
    model = YOLO('models/best.pt')
    
    print("Running inference on video...")
    print("This may take a moment...\n")
    
    # Run prediction on video
    # save=True: Saves annotated video to runs/detect/predict
    results = model.predict('input_videos/08fd33_4.mp4', save=True)
    
    # Display overall results for first frame
    print("="*60)
    print("FIRST FRAME DETECTION SUMMARY")
    print("="*60)
    print(results[0])
    print()
    
    # Display detailed information for each detected object
    print("="*60)
    print("DETAILED BOUNDING BOX INFORMATION")
    print("="*60)
    
    for i, box in enumerate(results[0].boxes, 1):
        print(f"\nObject {i}:")
        print(f"  {box}")
    
    print("\n" + "="*60)
    print(f"Total objects detected: {len(results[0].boxes)}")
    print("="*60)
    print("\n✓ Inference complete!")
    print("Check 'runs/detect/predict' for annotated output video")


if __name__ == '__main__':
    main()