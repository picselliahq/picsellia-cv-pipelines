from picsellia.types.schemas import LogDataType
from picsellia_cv_engine.models.parameters.hyper_parameters import (
    HyperParameters,
)


class UltralyticsHyperParameters(HyperParameters):
    def __init__(self, log_data: LogDataType):
        super().__init__(log_data=log_data)
        self.time = self.extract_parameter(
            keys=["time"], expected_type=float | None, default=None
        )
        self.patience = self.extract_parameter(
            keys=["patience"], expected_type=int, default=100
        )
        self.save_period = self.extract_parameter(
            keys=["save_period"],
            expected_type=int,
            default=-1,
        )
        self.cache = self.extract_parameter(
            keys=["cache", "use_cache"],
            expected_type=bool,
            default=False,
        )
        self.workers = self.extract_parameter(
            keys=["workers"], expected_type=int, default=8
        )
        self.optimizer = self.extract_parameter(
            keys=["optimizer"], expected_type=str, default="auto"
        )
        self.deterministic = self.extract_parameter(
            keys=["deterministic"], expected_type=bool, default=True
        )
        self.single_cls = self.extract_parameter(
            keys=["single_cls"], expected_type=bool, default=False
        )
        self.rect = self.extract_parameter(
            keys=["rect"], expected_type=bool, default=False
        )
        self.cos_lr = self.extract_parameter(
            keys=["cos_lr"], expected_type=bool, default=False
        )
        self.close_mosaic = self.extract_parameter(
            keys=["close_mosaic"], expected_type=int, default=10
        )
        self.amp = self.extract_parameter(
            keys=["amp"], expected_type=bool, default=True
        )
        self.fraction = self.extract_parameter(
            keys=["fraction"], expected_type=float, default=1.0
        )
        self.profile = self.extract_parameter(
            keys=["profile"], expected_type=bool, default=False
        )
        self.freeze = self.extract_parameter(
            keys=["freeze"], expected_type=int | None, default=None
        )
        self.lr0 = self.extract_parameter(
            keys=["lr0"], expected_type=float, default=0.01
        )
        self.lrf = self.extract_parameter(
            keys=["lrf"], expected_type=float, default=0.1
        )
        self.momentum = self.extract_parameter(
            keys=["momentum"], expected_type=float, default=0.937
        )
        self.weight_decay = self.extract_parameter(
            keys=["weight_decay"], expected_type=float, default=0.0005
        )
        self.warmup_epochs = self.extract_parameter(
            keys=["warmup_epochs"], expected_type=float, default=3.0
        )
        self.warmup_momentum = self.extract_parameter(
            keys=["warmup_momentum"], expected_type=float, default=0.8
        )
        self.warmup_bias_lr = self.extract_parameter(
            keys=["warmup_bias_lr"], expected_type=float, default=0.1
        )
        self.box = self.extract_parameter(
            keys=["box"], expected_type=float, default=7.5
        )
        self.cls = self.extract_parameter(
            keys=["cls"], expected_type=float, default=0.5
        )
        self.dfl = self.extract_parameter(
            keys=["dfl"], expected_type=float, default=1.5
        )
        self.pose = self.extract_parameter(
            keys=["pose"], expected_type=float, default=12.0
        )
        self.kobj = self.extract_parameter(
            keys=["kobj"], expected_type=float, default=2.0
        )
        self.label_smoothing = self.extract_parameter(
            keys=["label_smoothing"], expected_type=float, default=0.0
        )
        self.nbs = self.extract_parameter(keys=["nbs"], expected_type=int, default=64)
        self.overlap_mask = self.extract_parameter(
            keys=["overlap_mask"], expected_type=bool, default=True
        )
        self.mask_ratio = self.extract_parameter(
            keys=["mask_ratio"], expected_type=int, default=4
        )
        self.dropout = self.extract_parameter(
            keys=["dropout"], expected_type=float, default=0.0
        )
        self.plots = self.extract_parameter(
            keys=["plots"], expected_type=bool, default=False
        )
