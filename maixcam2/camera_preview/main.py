"""Run separately on MaixCAM2 in MaixVision to check the camera image."""
from maix import app, camera, display

cam = camera.Camera(320, 240)
screen = display.Display()
while not app.need_exit():
    screen.show(cam.read())
