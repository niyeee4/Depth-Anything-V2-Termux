import os
import sys
import cv2
import shutil
import subprocess
import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
import onnxruntime as ort
import tty
import termios

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

MODELS = [
    ('vits', 'Vit-Small~95MB  fastest, lower quality'),
    ('vitb', 'ViT-Base~370MB  balanced speed & quality'),
    ('vitl', 'ViT-Large~1.25GB best quality, slower'),
]

MODEL_URLS = {
    'vits': 'https://huggingface.co/niye4/depthmap-anything-v2-onnx/resolve/main/depth_anything_v2_vits.onnx',
    'vitb': 'https://huggingface.co/niye4/depthmap-anything-v2-onnx/resolve/main/depth_anything_v2_vitb.onnx',
    'vitl': 'https://huggingface.co/niye4/depthmap-anything-v2-onnx/resolve/main/depth_anything_v2_vitl.onnx',
}

script_dir = os.path.dirname(os.path.abspath(__file__))
LAST_MODEL_FILE = os.path.join(script_dir, ".last_model")

def onnx_path_for(enc):
    return os.path.join(script_dir, f"depth_anything_v2_{enc}.onnx")

def read_key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            return ch + ch2 + ch3
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)

def draw_menu(sel):
    sys.stdout.write("\033[H\033[J")
    print(f"{CYAN}{BOLD}=== Depth Anything V2 ==={RESET}\n")
    for i, (enc, desc) in enumerate(MODELS):
        exists = os.path.exists(onnx_path_for(enc))
        tag = f"{GREEN}[ready]{RESET}" if exists else f"{YELLOW}[not downloaded]{RESET}"
        cursor = f"{BOLD}>{RESET}" if i == sel else " "
        print(f" {cursor} {i+1}. {desc}  {tag}")
    print(f"\n  {YELLOW}up/down or 1/3 enter to confirm{RESET}")

def select_model():
    default = 1
    if os.path.exists(LAST_MODEL_FILE):
        try:
            last = open(LAST_MODEL_FILE).read().strip()
            for i, (enc, _) in enumerate(MODELS):
                if enc == last:
                    default = i
                    break
        except:
            pass
    sel = default
    draw_menu(sel)
    while True:
        k = read_key()
        if k == '\x1b[A':
            sel = (sel - 1) % 3
            draw_menu(sel)
        elif k == '\x1b[B':
            sel = (sel + 1) % 3
            draw_menu(sel)
        elif k in ('1', '2', '3'):
            sel = int(k) - 1
            draw_menu(sel)
        elif k in ('\r', '\n'):
            open(LAST_MODEL_FILE, "w").write(MODELS[sel][0])
            return MODELS[sel][0]
        elif k in ('\x03', '\x04'):
            print("\nCancelled.")
            sys.exit(0)

encoder = select_model()
onnx_path = onnx_path_for(encoder)
print(f"\nModel: {encoder}")

if not os.path.exists(onnx_path):
    sys.stdout.write("Model not found. Download? (y/n): ")
    sys.stdout.flush()
    ans = read_key()
    print(ans)
    if ans.lower() != 'y':
        print("Cancelled.")
        sys.exit(0)
    url = MODEL_URLS[encoder]
    ret = subprocess.run(["wget", "-q", "--show-progress", "-O", onnx_path, url])
    if ret.returncode != 0 or not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 1_000_000:
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        ret2 = subprocess.run(["curl", "-L", "--progress-bar", "-o", onnx_path, url])
        if ret2.returncode != 0 or not os.path.exists(onnx_path) or os.path.getsize(onnx_path) < 1_000_000:
            if os.path.exists(onnx_path):
                os.remove(onnx_path)
            print("Download failed.")
            sys.exit(1)
    print(f"{GREEN}Downloaded.{RESET}")

TMPFILE = "__input.tmp"
if os.path.exists(TMPFILE):
    os.remove(TMPFILE)
subprocess.Popen(["termux-storage-get", TMPFILE])
print("Pick a video/image from storage...")
import time
while not os.path.exists(TMPFILE):
    time.sleep(1)
time.sleep(1)

name = input("Enter name (no extension): ").strip()
if not name:
    name = "output"

mime = subprocess.check_output(["file", "--mime-type", "-b", TMPFILE]).decode().strip()
mime_sub = mime.split("/")[1]
ext_map = {"mp4": "mp4", "quicktime": "mov", "x-matroska": "mkv", "jpeg": "jpg", "png": "png", "webp": "webp", "bmp": "bmp"}
ext = ext_map.get(mime_sub, "mp4")

input_file = f"{name}.{ext}"
if os.path.exists(input_file):
    os.remove(input_file)
os.rename(TMPFILE, input_file)
is_image = ("." + ext) in IMAGE_EXTS

print(f"Loading model: {encoder}")
available = ort.get_available_providers()
providers = []
if 'NNAPIExecutionProvider' in available:
    providers.append(('NNAPIExecutionProvider', {'NNAPI_FLAG_USE_FP16': '1', 'NNAPI_FLAG_CPU_DISABLED': '0'}))
    print("Backend: NNAPI")
elif 'CUDAExecutionProvider' in available:
    providers.append('CUDAExecutionProvider')
    print("Backend: CUDA")
else:
    print("Backend: CPU")
providers.append('CPUExecutionProvider')

sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_opts.intra_op_num_threads = 4

session    = ort.InferenceSession(onnx_path, sess_opts, providers=providers)
input_name = session.get_inputs()[0].name
inp_shape  = session.get_inputs()[0].shape
INPUT_H    = inp_shape[2]
INPUT_W    = inp_shape[3]

def infer_depth(bgr):
    orig_h, orig_w = bgr.shape[:2]
    img = cv2.resize(bgr, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = img.transpose(2, 0, 1)[np.newaxis, ...]
    depth = session.run(None, {input_name: img})[0][0]
    depth = cv2.resize(depth, (orig_w, orig_h))
    mn, mx = depth.min(), depth.max()
    depth = (depth - mn) / (mx - mn + 1e-8) * 255.0
    return depth.astype(np.uint8)

os.makedirs("/sdcard/depthmap", exist_ok=True)

if is_image:
    bgr = cv2.imread(input_file)
    if bgr is None:
        print(f"Cannot read: {input_file}")
        sys.exit(1)
    print("Processing image...")
    depth = infer_depth(bgr)
    out_path = f"/sdcard/depthmap/{name}_depthmap.png"
    Image.fromarray(depth).save(out_path)
    os.remove(input_file)
    subprocess.run(["termux-media-scan", out_path])
    print(f"{GREEN}Done: {out_path}{RESET}")
    sys.exit(0)

def probe(field):
    return subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", f"stream={field}",
        "-of", "default=nw=1:nk=1", input_file
    ]).decode().strip()

fps     = probe("r_frame_rate")
codec   = probe("codec_name")
pixfmt  = probe("pix_fmt")
profile = probe("profile")

if codec == "hevc":   vcodec = "libx265"
elif codec == "vp9":  vcodec = "libvpx-vp9"
elif codec == "av1":  vcodec = "libaom-av1"
else:                 vcodec = "libx264"

frames_dir = "frames"
depth_dir  = "depth_frames"
shutil.rmtree(frames_dir, ignore_errors=True)
shutil.rmtree(depth_dir,  ignore_errors=True)
os.makedirs(frames_dir)
os.makedirs(depth_dir)

print("Extracting frames...")
subprocess.run(["ffmpeg", "-i", input_file, "-vsync", "0", f"{frames_dir}/%08d.png"],
               check=True, stderr=subprocess.DEVNULL)

files = sorted(os.listdir(frames_dir))
total = len(files)
print(f"Processing {total} frames...")

for i, fname in enumerate(files, 1):
    bgr   = cv2.imread(os.path.join(frames_dir, fname))
    depth = infer_depth(bgr)
    Image.fromarray(depth).save(os.path.join(depth_dir, fname))
    print(f"[{i}/{total}]", end='\r')

print()

output_video = f"/sdcard/depthmap/{name}_depthmap.mp4"
cmd = [
    "ffmpeg", "-y",
    "-framerate", fps,
    "-i", f"{depth_dir}/%08d.png",
    "-i", input_file,
    "-map", "0:v:0", "-map", "1:a?",
    "-c:v", vcodec, "-crf", "5", "-preset", "medium",
]
if pixfmt:
    cmd += ["-pix_fmt", pixfmt]
if vcodec == "libx264" and profile == "High":
    cmd += ["-profile:v", "high"]
elif vcodec == "libx265" and profile == "Main 10":
    cmd += ["-profile:v", "main10"]
cmd += ["-r", fps, "-c:a", "copy", "-shortest", output_video]

print("Encoding...")
subprocess.run(cmd, check=True, stderr=subprocess.DEVNULL)

shutil.rmtree(frames_dir, ignore_errors=True)
shutil.rmtree(depth_dir,  ignore_errors=True)
os.remove(input_file)

subprocess.run(["termux-media-scan", output_video])
print(f"{GREEN}Done: {output_video}{RESET}")
