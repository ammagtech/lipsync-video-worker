import os
import sys
import json
import base64
import tempfile
from rp_handler import handler

def test_musetalk():
    """Test the MuseTalk model with sample inputs"""
    print("Testing MuseTalk model...")
    
    # Test with official MuseTalk sample inputs
    test_input = {
        "input": {
            "image": "https://github.com/TMElyralab/MuseTalk/raw/main/assets/demo/Xinying_Sun.jpg",
            "audio": "https://github.com/TMElyralab/MuseTalk/raw/main/assets/demo/Xinying_Sun.wav",
            "bbox_shift": -7  # The recommended value for this sample from MuseTalk docs
        }
    }
    
    print(f"Using sample image and audio from official MuseTalk repository")
    print(f"Image: {test_input['input']['image']}")
    print(f"Audio: {test_input['input']['audio']}")
    print(f"bbox_shift: {test_input['input']['bbox_shift']}")
    
    
    # Call the handler
    try:
        result = handler(test_input)
        
        # Check if the result is successful
        if result.get("status") == "success":
            print("Test passed! MuseTalk model is working correctly.")
            
            # Save the output video to a file for inspection
            video_base64 = result.get("video_base64", "")
            if video_base64:
                with open("test_output.mp4", "wb") as f:
                    f.write(base64.b64decode(video_base64))
                print(f"Output video saved to test_output.mp4")
            
            return True
        else:
            print(f"Test failed! Error: {result.get('message', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"Test failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_musetalk()
