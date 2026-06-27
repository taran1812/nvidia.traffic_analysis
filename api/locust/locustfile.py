import os
from locust import HttpUser, task, between

_BUS_JPG = os.path.join(os.path.dirname(__file__), "..", "..", "bus.jpg")


class DetectUser(HttpUser):
    wait_time = between(0.1, 0.5)

    def on_start(self):
        with open(_BUS_JPG, "rb") as f:
            self._image_bytes = f.read()

    @task
    def detect(self):
        self.client.post(
            "/detect",
            files={"file": ("bus.jpg", self._image_bytes, "image/jpeg")},
        )
