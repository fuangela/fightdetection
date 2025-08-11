import cv2
import glob
import requests
import base64
import json
import os
import time 

# --- Configuration for LLaVA API ---
# IMPORTANT: Replace with the actual address of your LLaVA Model Worker
# This is NOT the controller port (10000).
# Check your model_worker.py output for the correct port.
LLAVA_MODEL_WORKER_URL = "http://localhost:29999" # Common default for model workers
LLAVA_API_ENDPOINT = f"{LLAVA_MODEL_WORKER_URL}/worker_generate" # Standard worker inference endpoint
LLAVA_MODEL_NAME = "llava-v1.5-7b" # Or "llava-v1.6-34b", etc. - match your loaded model

def extract_frames(video_path, output_dir, frame_rate=1):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0

    count, frame_idx = 0, 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Ensure we don't divide by zero if fps is 0
        if fps > 0 and count % int(fps / frame_rate) == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_idx:04d}.jpg", frame)
            frame_idx += 1
        count += 1

    cap.release()
    print(f"Extracted {frame_idx} frames from {os.path.basename(video_path)}")
    return frame_idx # Return number of frames for better logging/debugging


def analyze_frame_with_llava(image_path):
    with open(image_path, "rb") as f:
        img_data = f.read()
    b64_img = base64.b64encode(img_data).decode("utf-8") # Ensure UTF-8 encoding

    # Constructing the payload as expected by the LLaVA model worker
    payload = {
        "model": LLAVA_MODEL_NAME,
        "prompt": (
            "Look at this image and respond only in JSON:\n"
            "{ \"is_fighting\": true/false, \"description\": \"<short description>\" }"
        ),
        "temperature": 0.2, # Adjust as needed for determinism vs creativity
        "top_p": 0.7,       # Adjust as needed
        "max_new_tokens": 512, # Adjust as needed
        "images": [b64_img]  # Images should be a list of base64 strings
    }
    
    # Simple retry mechanism
    for attempt in range(3):
        try:
            # Use the correct model worker endpoint
            resp = requests.post(LLAVA_API_ENDPOINT, json=payload, timeout=60) # Increased timeout
            resp.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            
            response_json = resp.json()
            # The actual text reply is often nested under 'text' in the response from the worker
            text_reply = response_json.get("text", "") 
            
            try:
                # Attempt to parse as JSON
                r = json.loads(text_reply)
                return r.get("is_fighting", False), r.get("description", text_reply)
            except json.JSONDecodeError:
                # Fallback if LLaVA doesn't return perfect JSON
                print(f"Warning: LLaVA did not return valid JSON for {os.path.basename(image_path)}. Reply: {text_reply[:200]}...")
                for kw in ["fight", "punch", "kick", "hit", "brawl", "violence", "attack", "fighting", "aggressive"]:
                    if kw in text_reply.lower():
                        return True, text_reply
                return False, text_reply
        
        except requests.exceptions.RequestException as e:
            print(f"Request failed (attempt {attempt + 1}/3) for {os.path.basename(image_path)}: {e}")
            if attempt < 2:
                time.sleep(5) # Wait before retrying
            else:
                print(f"Failed to analyze {os.path.basename(image_path)} after multiple attempts.")
                return False, f"Error: Request failed - {e}"
    
    return False, "Error: Unknown analysis failure"

# 4. Run end-to-end on the dataset
def process_dataset(root_dir="fight-detection-surv-dataset"):
    categories = ["fight", "noFight"]
    results = []

    for cat in categories:
        video_paths = glob.glob(os.path.join(root_dir, cat, "*.mp4"))
        if not video_paths:
            print(f"No .mp4 videos found in {os.path.join(root_dir, cat)}")
            continue

        print(f"\nProcessing category: {cat}")
        for vp in video_paths:
            vid = os.path.splitext(os.path.basename(vp))[0]
            frame_dir = os.path.join("frames", cat, vid)
            
            # Ensure frame extraction happens and check if any frames were extracted
            num_frames = extract_frames(vp, frame_dir, frame_rate=1)
            if num_frames == 0:
                print(f"Skipping {vid} due to no frames extracted.")
                continue

            vid_has_fight = False
            # Ensure the glob pattern matches the extracted frame names
            image_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            
            if not image_files:
                print(f"No image files found in {frame_dir} after extraction for {vid}. Skipping.")
                continue

            for img in image_files:
                is_fight, desc = analyze_frame_with_llava(img)
                results.append({"video": vid, "frame": os.path.basename(img),
                                "is_fighting": is_fight, "description": desc,
                                "category": cat # Add original category for analysis
                                })
                if is_fight:
                    vid_has_fight = True
            
            status = "⚠️ FIGHT" if vid_has_fight else "✅ NO FIGHT"
            print(f"{vid}: {status}")

    import pandas as pd
    if results:
        df = pd.DataFrame(results)
        df.to_csv("llava_results.csv", index=False)
        print("\nSaved results to llava_results.csv")
    else:
        print("\nNo results to save. Ensure videos are present and frames extracted.")

if __name__ == "__main__":
    process_dataset()