import cv2
import os
import threading
import queue
import config


class AsyncVideoRecorder:
    def __init__(self, output_path, fps=30.0, frame_size=(1280, 720)):
        self.output_path = output_path
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
        self.queue = queue.Queue()
        self.recording = True

        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()

    def write(self, frame):
        if self.recording:
            self.queue.put(frame)

    def _write_loop(self):
        while self.recording or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.recording = False
        self.thread.join()
        self.writer.release()