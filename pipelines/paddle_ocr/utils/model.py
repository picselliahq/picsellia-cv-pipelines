from picsellia_cv_engine.core.models import Model, ModelCollection


class PaddleOCRModelCollection(ModelCollection):
    """
    A specialized collection for Paddle OCR models that contains two models:
    one for bounding box detection and one for text recognition.

    Attributes:
        bbox_model (Model): The model for the bounding box detection model.
        text_model (Model): The model for the text recognition model.
    """

    def __init__(self, bbox_model: Model, text_model: Model):
        """
        Initializes a new PaddleOCRModelCollection with specified contexts for bounding box and text recognition models.

        Args:
            bbox_model (Model): The model for the bounding box detection.
            text_model (Model): The model for the text recognition.
        """
        super().__init__([bbox_model, text_model])
        self.bbox_model = bbox_model
        self.text_model = text_model
