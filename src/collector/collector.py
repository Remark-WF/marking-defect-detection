#!/usr/bin/env python3

import os
import time
import subprocess
import cv2
import getpass

# ================= CONFIG =================

MOUNT_POINT = "/media/data"

VIDEO_FPS = 20
SEGMENT_SECONDS = 600

WIDTH = 1920
HEIGHT = 1080

BITRATE = "12M"
MAXRATE = "14M"
BUFSIZE = "24M"

SCAN_INTERVAL = 2
FRAME_INTERVAL = 1.0 / VIDEO_FPS

PREVIEW_INTERVAL = 2.0

# ==========================================


def ensure_mount():
    os.makedirs(MOUNT_POINT, exist_ok=True)


def writable(path):
    try:
        test = os.path.join(path, ".test")
        with open(test, "w") as f:
            f.write("ok")
        os.remove(test)
        return True
    except Exception:
        return False


def find_usb():
    for dev in os.listdir("/dev"):
        if dev.startswith("sd") and dev.endswith("1"):
            return "/dev/" + dev
    return None


def mount_drive():
    ensure_mount()

    device = find_usb()

    if device is None:
        return None, None

    if not os.path.ismount(MOUNT_POINT):

        try:
            subprocess.run(
                ["sudo", "mount", "-o", "uid=1000,gid=1000", device, MOUNT_POINT],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except subprocess.CalledProcessError:
            return None, None

    if writable(MOUNT_POINT):
        return MOUNT_POINT, device

    return None, None


def wait_drive():

    print("Waiting data drive...")

    while True:

        drive, dev = mount_drive()

        if drive:
            print("Drive ready:", drive)
            return drive, dev

        time.sleep(SCAN_INTERVAL)


def find_camera():

    for dev in sorted(os.listdir("/dev")):

        if not dev.startswith("video"):
            continue

        device = "/dev/" + dev

        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)

        if not cap.isOpened():
            continue

        ret, _ = cap.read()
        cap.release()

        if ret:
            return device

    return None


def wait_camera():

    print("Waiting camera...")

    while True:

        cam = find_camera()

        if cam:
            print("Camera:", cam)
            return cam

        time.sleep(SCAN_INTERVAL)


def create_session_dir(drive):

    ts = time.strftime("%Y%m%d_%H%M%S")

    path = os.path.join(
        drive,
        f"video_session_{ts}"
    )

    os.makedirs(path, exist_ok=True)

    print("Session dir:", path)

    return path


def start_ffmpeg(out_path):

    cmd = [

        "ffmpeg",
        "-loglevel", "error",
        "-y",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{WIDTH}x{HEIGHT}",
        "-r", str(VIDEO_FPS),
        "-i", "-",

        "-an",

        "-c:v", "libx264",
        "-preset", "veryfast",
        "-pix_fmt", "yuv420p",

        "-b:v", BITRATE,
        "-maxrate", MAXRATE,
        "-bufsize", BUFSIZE,

        out_path
    ]

    return subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


class SegmentRecorder:

    def __init__(self, out_dir):

        self.out_dir = out_dir

        self.segment_index = 0
        self.proc = None
        self.start_ts = 0

        self.preview_deadline = 0

    def segment_path(self):

        ts = time.strftime("%Y%m%d_%H%M%S")

        return os.path.join(
            self.out_dir,
            f"segment_{ts}_{self.segment_index:03d}.mp4"
        )

    def start(self):

        self.close()

        self.segment_index += 1

        path = self.segment_path()

        print("Segment start:", path)

        self.proc = start_ffmpeg(path)

        self.start_ts = time.time()
        self.preview_deadline = self.start_ts

    def rotate_if_needed(self):

        if self.proc is None:
            self.start()
            return

        if time.time() - self.start_ts >= SEGMENT_SECONDS:
            self.start()

    def write(self, frame):

        if self.proc is None:
            return

        try:
            self.proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            raise RuntimeError("ffmpeg pipe broken")

    def close(self):

        if self.proc is None:
            return

        try:
            self.proc.stdin.close()
        except Exception:
            pass

        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()

        self.proc = None


def run_recording(drive, device, camera):

    session_dir = create_session_dir(drive)

    cap = cv2.VideoCapture(camera, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    recorder = SegmentRecorder(session_dir)

    last_frame = time.perf_counter()

    print("Recording started")

    try:

        while True:

            if not os.path.exists(device):
                print("Drive removed")
                break

            if not os.path.exists(camera):
                print("Camera removed")
                break

            now = time.perf_counter()

            dt = now - last_frame

            if dt < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - dt)

            last_frame = time.perf_counter()

            ret, frame = cap.read()

            if not ret:
                time.sleep(0.05)
                continue

            recorder.rotate_if_needed()

            recorder.write(frame)

    finally:

        recorder.close()
        cap.release()

        print("Recording finished")


def main():

    print("Video collector started")

    while True:

        try:

            drive, dev = wait_drive()

            camera = wait_camera()

            run_recording(drive, dev, camera)

        except KeyboardInterrupt:
            print("Stopped")
            break

        except Exception as e:

            print("Collector error:", e)

            time.sleep(2)


if __name__ == "__main__":
    main()