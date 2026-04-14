import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import os
import subprocess
from collections import defaultdict, deque
import time

# Load model
print("Loading model...")
model = YOLO('models/v1_best.pt')
print("✅ Model loaded!")

SAMPLE_IMAGES = {
    "🔵 Crack":      ["samples/images/Crack_1.jpg",     "samples/images/Crack_2.jpeg",
                      "samples/images/Crack_3.jpg",     "samples/images/Crack_4.jpg",
                      "samples/images/Crack_5.jpeg", "samples/images/Crack_6.jpeg"],
    "💠 Pothole":    ["samples/images/Pothole_1.jpg",   "samples/images/Pothole_2.jpeg",
                      "samples/images/Pothole_3.jpg",   "samples/images/Pothole_4.jpg"],
    "⚪ Speed Bump": ["samples/images/Unmarked_speed_bump_1.jpg", "samples/images/Unmarked_speed_bump_2.jpg",
                      "samples/images/Unmarked_speed_bump_3.jpg", "samples/images/Unmarked_speed_bump_4.jpg"],
}

SAMPLE_VIDEOS = {
    "🔵 Crack":      ["samples/videos/Crack_video_1.mp4"],
    "💠 Pothole":    ["samples/videos/Pothole_video_1.mp4",              "samples/videos/Pothole_video_2.mp4"],
    "⚪ Speed Bump": ["samples/videos/unmarked_speed_bump_video_1.mp4",    "samples/videos/unmarked_speed_bump_video_2.mp4", "samples/videos/unmarked_speed_bump_video_3.mp4"],
}

# ─────────────────────────────────────────────
# IMAGE DETECTION
# ─────────────────────────────────────────────
def detect_image(image, conf_val):
    if image is None:
        return None, "⚠️ No image uploaded"

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if (len(image.shape) == 3 and image.shape[2] == 3) else image
    results   = model.predict(image_bgr, conf=conf_val, imgsz=640, verbose=False)
    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

    boxes = results[0].boxes
    if len(boxes) == 0:
        return annotated, "⚠️ **No anomalies detected**\n\nTry:\n- Lowering confidence slider\n- Using clearer road images"

    stats       = f"✅ **Found {len(boxes)} anomaly/anomalies:**\n\n"
    class_names = results[0].names
    classes     = boxes.cls.cpu().numpy()

    for cls_id in np.unique(classes):
        count      = int(np.sum(classes == cls_id))
        class_name = class_names[int(cls_id)].lower()
        icon = "🔵" if "crack" in class_name else "💠" if "pothole" in class_name else "⚪"
        stats += f"{icon} **{class_names[int(cls_id)]}**: {count}\n"

    stats += f"\n📊 Average Confidence: {float(boxes.conf.mean()):.2%}"
    return annotated, stats


# Audio for detected video
def mux_audio(original_path, processed_path, output_path):
    try:
        # First check if the original video has an audio stream
        probe = subprocess.run([
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            original_path
        ], capture_output=True, text=True)

        has_audio = probe.stdout.strip() == "audio"

        if has_audio:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", processed_path,
                "-i", original_path,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                output_path
            ], check=True, capture_output=True)
        else:
            # No audio — just copy the video as-is
            subprocess.run([
                "ffmpeg", "-y",
                "-i", processed_path,
                "-c:v", "copy",
                output_path
            ], check=True, capture_output=True)

        return output_path

    except subprocess.CalledProcessError:
        return processed_path


# ─────────────────────────────────────────────
# VIDEO DETECTION (with progress bar + temporal filtering)
# ─────────────────────────────────────────────
def detect_video(video_path, conf_val, progress=gr.Progress()):
    if video_path is None:
        return None, None, "⚠️ No video uploaded"

    cap          = cv2.VideoCapture(video_path)
    fps          = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ✅ Write to AVI first — most reliable on Windows CPU
    avi_path = "output_video.avi"
    fourcc   = cv2.VideoWriter_fourcc(*"XVID")
    out      = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out    = cv2.VideoWriter(avi_path, fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        return None, None, "❌ Could not initialize video writer"

    # Temporal filter — rolling window of 2 frames per class
    class_history = defaultdict(lambda: deque(maxlen=2))

    frame_count       = 0
    confirmed_classes = defaultdict(int)
    frame_times       = []  # FPS tracking

    progress(0, desc="Starting...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        t0      = time.time()
        results = model.predict(frame, conf=conf_val, imgsz=640, verbose=False)
        t1      = time.time()
        frame_times.append(t1 - t0)

        boxes        = results[0].boxes
        class_names  = results[0].names
        detected_now = set()

        if len(boxes) > 0:
            for cls_id in boxes.cls.cpu().numpy():
                cls_id = int(cls_id)
                detected_now.add(cls_id)
                class_history[cls_id].append(1)
                if sum(class_history[cls_id]) >= 2:
                    confirmed_classes[cls_id] += 1

        # Zero out classes not detected this frame
        all_cls = set(class_history.keys())
        for cls_id in all_cls - detected_now:
            class_history[cls_id].append(0)

        # Draw live FPS on each frame
        current_fps = 1.0 / frame_times[-1] if frame_times[-1] > 0 else 0
        annotated   = results[0].plot()
        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        out.write(annotated)
        frame_count += 1

        if total_frames > 0:
            progress(frame_count / total_frames, desc=f"Processing frame {frame_count}/{total_frames}")

    cap.release()
    out.release()

    # ✅ Convert AVI → MP4 using ffmpeg
    mp4_path = "output_video.mp4"
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", avi_path,
            "-c:v", "libx264", "-preset", "fast",
            mp4_path
        ], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        mp4_path = avi_path  # fallback to AVI if conversion fails

    # Calculate avg FPS for stats
    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0
    min_fps = 1.0 / max(frame_times) if frame_times else 0
    max_fps = 1.0 / min(frame_times) if frame_times else 0

    final_path = "output_video_audio.mp4"
    final_path = mux_audio(video_path, mp4_path, final_path)

    stats = (
        f"**✅ Video processed successfully!**\n\n"
        f"⚡ **Avg Inference FPS:** {avg_fps:.1f}\n"
        f"📈 **Peak FPS:** {max_fps:.1f}\n"
        f"📉 **Min FPS:** {min_fps:.1f}\n"
        f"🎞️ **Total Frames:** {frame_count}"
    )
    return final_path, final_path, stats


# ─────────────────────────────────────────────
# WEBCAM
# ─────────────────────────────────────────────
last_time = [time.time()]

def detect_webcam(image, conf_val):
    if image is None:
        return None

    # FPS calculation
    now = time.time()
    fps = 1.0 / (now - last_time[0])
    last_time[0] = now

    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if (len(image.shape) == 3 and image.shape[2] == 3) else image
    results   = model.predict(image_bgr, conf=conf_val, imgsz=640, verbose=False)
    annotated = results[0].plot()

    # Draw FPS on frame
    cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


# ─────────────────────────────────────────────
# SAMPLE HELPERS
# ─────────────────────────────────────────────
def load_and_detect_sample_image(path, conf_val):
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return detect_image(img, conf_val)

def load_and_detect_sample_video(path, conf_val):
    return detect_video(path, conf_val)

def preview_sample_video(path):
    return path


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
with gr.Blocks() as demo:

    gr.Markdown("""
    # 🛣️ Road Anomaly Detection System
    ## YOLOv11s-seg Instance Segmentation
    **Detects:** Cracks 🔵 | Potholes 💠 | Unmarked Speed Bumps ⚪
    """)

    with gr.Row():
        conf_slider = gr.Slider(
            minimum=0.05, maximum=0.9, value=0.25, step=0.05,
            label="🎯 Confidence Threshold (Lower = More Sensitive)",
            info="Decrease if missing detections, increase if too many false positives"
        )

    with gr.Tabs():

        # ── IMAGE UPLOAD TAB ──────────────────────────────
        with gr.Tab("📷 Image Upload"):
            with gr.Row():
                with gr.Column():
                    img_input = gr.Image(type="numpy", label="Upload Road Image")
                    img_btn   = gr.Button("🔍 Detect Anomalies", variant="primary", size="lg")
                with gr.Column():
                    img_output = gr.Image(label="Detection Results")
                    img_stats  = gr.Markdown("Upload an image and click detect")

            img_btn.click(fn=detect_image, inputs=[img_input, conf_slider], outputs=[img_output, img_stats])

        # ── VIDEO UPLOAD TAB ──────────────────────────────
        with gr.Tab("🎥 Video Upload"):
            with gr.Row():
                with gr.Column():
                    vid_input = gr.Video(label="Upload Road Video")
                    vid_btn   = gr.Button("🔍 Process Video", variant="primary", size="lg")
                with gr.Column():
                    vid_output   = gr.Video(label="Processed Video")
                    vid_download = gr.File(label="⬇️ Download Processed Video")
                    vid_stats    = gr.Markdown("Upload a video and click process")

            vid_btn.click(
                fn=detect_video,
                inputs=[vid_input, conf_slider],
                outputs=[vid_output, vid_download, vid_stats]
            )

        # ── WEBCAM TAB ────────────────────────────────────
        with gr.Tab("📹 Webcam Live"):
            gr.Markdown("""
            ### Real-time Detection (Allow camera access)
            > 💡 Webcam streams frame-by-frame — no file is saved, so there is no download.
            > This is normal and works the same on CPU or GPU.
            """)
            with gr.Row():
                webcam_input  = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam Feed")
                webcam_output = gr.Image(label="Live Detection", streaming=True)

            webcam_input.stream(
                fn=lambda img: detect_webcam(img, conf_slider.value),
                inputs=webcam_input,
                outputs=webcam_output
            )

        # ── SAMPLE IMAGES TAB ────────────────────────────
        with gr.Tab("🖼️ Sample Images"):
            gr.Markdown("### Click any sample image to run detection instantly")

            sample_img_display = gr.Image(label="Detection Result")
            sample_img_stats   = gr.Markdown("Select a sample below")

            for class_label, paths in SAMPLE_IMAGES.items():
                gr.Markdown(f"#### {class_label}")
                with gr.Row():
                    for p in paths:
                        with gr.Column():
                            thumb = gr.Image(value=p, label=os.path.basename(p), interactive=False, height=180)
                            btn   = gr.Button(f"▶️ Detect — {os.path.basename(p)}", size="sm")
                            btn.click(
                                fn=load_and_detect_sample_image,
                                inputs=[gr.State(p), conf_slider],
                                outputs=[sample_img_display, sample_img_stats]
                            )

        # ── SAMPLE VIDEOS TAB ────────────────────────────
        with gr.Tab("🎞️ Sample Videos"):
            gr.Markdown("### Preview a video, then run detection")

            with gr.Row():
                with gr.Column():
                    sample_vid_preview  = gr.Video(label="▶️ Video Preview", interactive=False)
                with gr.Column():
                    sample_vid_output   = gr.Video(label="Processed Video")
                    sample_vid_download = gr.File(label="⬇️ Download")
                    sample_vid_stats    = gr.Markdown("Select a sample below")

            sample_detect_btn = gr.Button("🔍 Run Detection on Previewed Video", variant="primary", size="lg")
            current_video_path = gr.State(None)

            for class_label, paths in SAMPLE_VIDEOS.items():
                gr.Markdown(f"#### {class_label}")
                with gr.Row():
                    for p in paths:
                        with gr.Column():
                            btn = gr.Button(f"▶️ {os.path.basename(p)}", variant="secondary")
                            btn.click(
                                fn=preview_sample_video,
                                inputs=[gr.State(p)],
                                outputs=[sample_vid_preview]
                            ).then(
                                fn=lambda path=p: path,
                                inputs=[],
                                outputs=[current_video_path]
                            )

            sample_detect_btn.click(
                fn=load_and_detect_sample_video,
                inputs=[current_video_path, conf_slider],
                outputs=[sample_vid_output, sample_vid_download, sample_vid_stats]
            )

    gr.Markdown("""
    ---
    ### 📊 Model Information
    - **Architecture:** YOLOv11s-seg (Instance Segmentation)
    - **Training mAP50:** 0.869 | **Image Size:** 960×960
    - **Classes:** Crack 🔵 · Pothole 💠 · Unmarked Speed Bump ⚪
    """)

print("🚀 Launching Gradio app...")
demo.launch(share=False, theme=gr.themes.Soft())