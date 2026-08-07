# Models

`briscola_cards.onnx` is the one-class Briscola card detector trained from
`yolo26n-obb.pt`. Its fixed input is 1024 by 1024 pixels. OpenCV performs
rotated NMS after inference.

The repository ignores model binaries, so the file must be supplied separately.
