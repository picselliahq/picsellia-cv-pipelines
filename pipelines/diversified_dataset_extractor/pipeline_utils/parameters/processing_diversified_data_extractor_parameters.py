from src.picsellia_cv_engine.models.parameters.parameters import Parameters


class ProcessingDiversifiedDataExtractorParameters(Parameters):
    def __init__(self, log_data):
        super().__init__(log_data)

        self.distance_threshold = self.extract_parameter(
            keys=["distance", "dist", "dist_threshold", "distance_threshold"],
            expected_type=int,
            default=5,
            range_value=(
                0.0,
                float("inf"),
            ),
        )

        self.embedding_model = self.extract_parameter(
            keys=["embedding_model", "model"],
            expected_type=str,
            default="openclip",
        )

        self.model_architecture = self.extract_parameter(
            keys=["model_architecture", "architecture"],
            expected_type=str,
            default="ViT-B-16-plus-240",
        )

        self.pretrained_weights = self.extract_parameter(
            keys=["pretrained_weights", "weights"],
            expected_type=str,
            default="laion400m_e32",
        )
