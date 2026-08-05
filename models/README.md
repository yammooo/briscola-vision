# Models

Place the OpenCV-compatible `cardcaptor_v3_best.onnx` in this directory. It is
exported once from the original `.pt` weights with model NMS disabled; OpenCV
performs rotated NMS after inference.

Source: https://huggingface.co/AlecKarfonta/cardcaptor-v3

The repository ignores model binaries, so the file must be supplied separately.
