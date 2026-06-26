"""Command line programs for S-JEPA.

Each program reads a YAML config with the -c/--config flag and runs one job:

  * `train`: train the model (command: trainsjepa).
  * `buildds`: build a ready-to-train HDF5 dataset (command: buildsjepa).
  * `evaluate`: evaluate a trained model on the full test set (command: evalsjepa).
  * `exportmodel`: export the encoder to ONNX (command: exportsjepa).
  * `inference`: run a standalone ONNX inference (command: infersjepa).
"""
