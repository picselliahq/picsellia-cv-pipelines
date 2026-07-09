from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import HyperParameters


class TrainingHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)

        # Core training loop
        self.epochs = self.extract_parameter(["epochs"], expected_type=int, default=10)
        # SAM-3's forward activations (e.g. the relative-position-bias matrix)
        # scale linearly with batch size and dominate memory; at 1008x1008 a single
        # sample already fills a 15 GB T4. Default to 1 and use grad accumulation to
        # raise the effective batch.
        self.batch_size = self.extract_parameter(
            ["batch_size"], expected_type=int, default=1
        )
        self.learning_rate = self.extract_parameter(
            ["learning_rate", "lr"], expected_type=float, default=1e-4
        )
        self.weight_decay = self.extract_parameter(
            ["weight_decay"], expected_type=float, default=1e-4
        )
        self.grad_accumulation_steps = self.extract_parameter(
            ["grad_accumulation_steps"], expected_type=int, default=1
        )
        self.max_grad_norm = self.extract_parameter(
            ["max_grad_norm"], expected_type=float, default=1.0
        )
        self.warmup_ratio = self.extract_parameter(
            ["warmup_ratio"], expected_type=float, default=0.0, range_value=(0.0, 1.0)
        )

        # SAM-3 is a large model: by default only the DETR encoder/decoder, mask
        # decoder and scoring heads are fine-tuned. The frozen ViT vision encoder
        # and CLIP text encoder keep their pretrained zero-shot features and avoid
        # the memory cost of backpropagating through them at 1008x1008 resolution.
        self.freeze_vision_encoder = self.extract_parameter(
            ["freeze_vision_encoder"], expected_type=bool, default=True
        )
        self.freeze_text_encoder = self.extract_parameter(
            ["freeze_text_encoder"], expected_type=bool, default=True
        )

        # Set-prediction loss weights (DETR-style criterion)
        self.class_loss_weight = self.extract_parameter(
            ["class_loss_weight"], expected_type=float, default=2.0
        )
        self.presence_loss_weight = self.extract_parameter(
            ["presence_loss_weight"], expected_type=float, default=1.0
        )
        self.bbox_loss_weight = self.extract_parameter(
            ["bbox_loss_weight"], expected_type=float, default=5.0
        )
        self.giou_loss_weight = self.extract_parameter(
            ["giou_loss_weight"], expected_type=float, default=2.0
        )
        self.mask_loss_weight = self.extract_parameter(
            ["mask_loss_weight"], expected_type=float, default=5.0
        )
        self.dice_loss_weight = self.extract_parameter(
            ["dice_loss_weight"], expected_type=float, default=5.0
        )
        self.focal_alpha = self.extract_parameter(
            ["focal_alpha"], expected_type=float, default=0.25
        )
        self.focal_gamma = self.extract_parameter(
            ["focal_gamma"], expected_type=float, default=2.0
        )

        # Ground-truth mask resolution used for the mask loss (matches SAM-3's
        # native 288x288 mask grid; predictions are resized to this if needed).
        self.mask_resolution = self.extract_parameter(
            ["mask_resolution"], expected_type=int, default=288
        )

        # Build one (image, concept) sample for every label in the dataset on
        # every image, including images where the concept is absent. Teaches the
        # model to predict "nothing" but multiplies the sample count by the number
        # of labels, so it is off by default.
        self.include_negative_samples = self.extract_parameter(
            ["include_negative_samples"], expected_type=bool, default=False
        )

        # Evaluation thresholds (used by the post-training evaluation step)
        self.eval_score_threshold = self.extract_parameter(
            ["eval_score_threshold"], expected_type=float, default=0.3, range_value=(0.0, 1.0)
        )
        self.eval_mask_threshold = self.extract_parameter(
            ["eval_mask_threshold"], expected_type=float, default=0.5, range_value=(0.0, 1.0)
        )
        self.eval_min_polygon_area = self.extract_parameter(
            ["eval_min_polygon_area"], expected_type=float, default=20.0
        )
