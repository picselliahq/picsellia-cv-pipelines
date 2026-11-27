from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.core.parameters import ExportParameters, HyperParameters


class TrainingHyperParameters(HyperParameters):
    """Defaults follow the official Mask2Former recipe (AdamW, backbone lr x0.1,
    weight decay 0.05, gradient clipping at 0.01)."""

    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        # Data and model
        # Empty: use the `pretrained-weights` file of the model version (no Hugging Face
        # access needed), else the default hub checkpoint. A hub identifier or a local
        # directory overrides it.
        self.pretrained_model = self.extract_parameter(
            ["pretrained_model"], expected_type=str, default=""
        )
        self.epochs = self.extract_parameter(["epochs"], expected_type=int, default=30)
        self.batch_size = self.extract_parameter(
            ["batch_size"], expected_type=int, default=2
        )
        self.image_size = self.extract_parameter(
            ["image_size"], expected_type=int, default=512
        )
        # Optimisation
        self.learning_rate = self.extract_parameter(
            ["learning_rate", "lr"], expected_type=float, default=1e-4
        )
        self.backbone_lr_multiplier = self.extract_parameter(
            ["backbone_lr_multiplier"], expected_type=float, default=0.1
        )
        self.weight_decay = self.extract_parameter(
            ["weight_decay"], expected_type=float, default=0.05
        )
        self.lr_scheduler = self.extract_parameter(
            ["lr_scheduler"], expected_type=str, default="cosine"
        )
        self.lr_scheduler_step_size = self.extract_parameter(
            ["lr_scheduler_step_size", "step_size"], expected_type=int, default=10
        )
        self.lr_scheduler_gamma = self.extract_parameter(
            ["lr_scheduler_gamma", "gamma"], expected_type=float, default=0.1
        )
        self.warmup_iters = self.extract_parameter(
            ["warmup_iters"], expected_type=int, default=100
        )
        self.clip_grad_norm = self.extract_parameter(
            ["clip_grad_norm"], expected_type=float, default=0.01
        )
        self.amp = self.extract_parameter(["amp"], expected_type=bool, default=True)
        # Augmentation (train split only)
        self.hflip_prob = self.extract_parameter(
            ["hflip_prob"], expected_type=float, default=0.5
        )
        # Model selection
        self.keep_best = self.extract_parameter(
            ["keep_best"], expected_type=bool, default=True
        )
        self.patience = self.extract_parameter(
            ["patience"], expected_type=int, default=0
        )
        self.val_map_every = self.extract_parameter(
            ["val_map_every"], expected_type=int, default=1
        )
        # Evaluation. Metrics (mAP) need low-score detections to be meaningful, while
        # the evaluations shown in Picsellia are filtered with a higher threshold.
        self.metrics_score_threshold = self.extract_parameter(
            ["metrics_score_threshold"], expected_type=float, default=0.05
        )
        self.score_threshold = self.extract_parameter(
            ["score_threshold", "confidence_threshold"],
            expected_type=float,
            default=0.5,
        )
        self.mask_threshold = self.extract_parameter(
            ["mask_threshold"], expected_type=float, default=0.5
        )
        self.overlap_mask_area_threshold = self.extract_parameter(
            ["overlap_mask_area_threshold"], expected_type=float, default=0.8
        )
        self.eval_batch_size = self.extract_parameter(
            ["eval_batch_size"], expected_type=int, default=4
        )
        # Reporting
        self.overlay_images = self.extract_parameter(
            ["overlay_images"], expected_type=int, default=8
        )
        self.log_every = self.extract_parameter(
            ["log_every"], expected_type=int, default=0
        )


class Mask2FormerExportParameters(ExportParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.export_format = self.extract_parameter(
            keys=["export_format"], expected_type=str, default="pytorch"
        )
