"""Точка входа веб-приложения ONNX Quantization Lab."""

from quantlab.ui.workspace import create_app, launch

app = create_app()

if __name__ == '__main__':
    launch(app)
