import unittest
import uuid

from picsellia.sdk.label import Label
from torch import tensor
from utils.picsellia import (
    create_rectangle_list,
    get_filename_from_fullpath,
    reformat_box_to_coco,
)
from utils.vit import (
    format_coco_annot_to_jsonlines_format,
    format_evaluation_results,
    formatted_annotations,
    get_category_names,
    get_id2label_mapping,
    read_annotation_file,
)


class TestDetectionVit(unittest.TestCase):
    checkpoint = "facebook/detr-resnet-50"
    id2label = {0: "grape"}
    label2id = {"grape": 0}

    def setUp(self) -> None:
        self.annotations = read_annotation_file("test_files/annotations.json")

        # self.image_processor = AutoImageProcessor.from_pretrained(self.checkpoint)
        # self.model = AutoModelForObjectDetection.from_pretrained(
        #     self.checkpoint,
        #     id2label=self.id2label,
        #     label2id=self.label2id,
        #     ignore_mismatched_sizes=True,
        # )

    def test_get_filename_from_fullpath(self):
        filename = get_filename_from_fullpath("hajer/home/sample.jpg")
        self.assertEqual(filename, "sample.jpg")

    def test_create_rectangle_list(self):
        sample_results = {
            "scores": tensor(
                [
                    0.5396,
                    0.5871,
                ]
            ),
            "labels": tensor([0, 0, 0]),
            "boxes": tensor(
                [
                    [1176.8936, 218.8128, 1193.3655, 252.6124],
                    [1183.8063, 206.4708, 1208.8052, 248.1937],
                ]
            ),
        }
        label1 = Label(None, {"id": uuid.uuid4(), "name": "grape"})

        expected_rectangle_list = [
            (
                1176,
                218,
                17,
                34,
                label1,
                0.54,
            ),
            (1183, 206, 25, 42, label1, 0.587),
        ]
        rectangle_list = create_rectangle_list(
            results=sample_results,
            id2label=self.id2label,
            dataset_labels={"grape": label1},
        )
        self.assertEqual(rectangle_list, expected_rectangle_list)

    def test_reformat_box_to_coco(self):
        sample_box = tensor([1176.8936, 218.8128, 1193.3655, 252.6124])
        expected_formatted_box = [1176, 218, 17, 34]
        formatted_box = reformat_box_to_coco(box=sample_box)
        self.assertEqual(formatted_box, expected_formatted_box)

    def test_get_id2label_mapping(self):
        id2label = get_id2label_mapping(self.annotations)
        expected_id2label = {0: "grape"}
        self.assertEqual(id2label, expected_id2label)

    def test_get_category_names(self):
        category_names = get_category_names(self.annotations)
        expected_category_names = ["grape"]
        self.assertEqual(category_names, expected_category_names)

    def test_format_coco_annot_to_jsonlines_format(self):
        formatted_coco = format_coco_annot_to_jsonlines_format(
            annotations=self.annotations
        )
        formatted_coco_annotations = [
            {
                "file_name": "SYH_2017-04-27_1275.jpg",
                "image_id": 0,
                "width": 2048,
                "height": 1365,
                "objects": {
                    "id": [14, 15, 16, 17, 18, 19, 20, 21],
                    "bbox": [
                        [1679, 510, 286, 537],
                        [717, 744, 207, 407],
                        [95, 655, 184, 434],
                        [3, 701, 97, 339],
                        [463, 597, 197, 297],
                        [700, 567, 216, 150],
                        [681, 423, 188, 167],
                        [475, 253, 221, 369],
                    ],
                    "category": [0, 0, 0, 0, 0, 0, 0, 0],
                    "area": [
                        153582.0,
                        84249.0,
                        79856.0,
                        32883.0,
                        58509.0,
                        32400.0,
                        31396.0,
                        81549.0,
                    ],
                    "image_id": [0, 0, 0, 0, 0, 0, 0, 0],
                },
            },
            {
                "file_name": "CFR_1658.jpg",
                "image_id": 1,
                "width": 2048,
                "height": 1365,
                "objects": {
                    "id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                    "bbox": [
                        [305, 179, 84, 113],
                        [194, 353, 161, 195],
                        [287, 474, 169, 312],
                        [441, 486, 156, 317],
                        [953, 557, 180, 350],
                        [623, 402, 259, 441],
                        [1123, 572, 156, 176],
                        [1207, 170, 161, 443],
                        [1315, 373, 187, 376],
                        [1477, 527, 101, 300],
                        [1545, 449, 125, 238],
                        [1755, 483, 115, 219],
                        [1892, 486, 154, 333],
                        [1825, 610, 141, 267],
                    ],
                    "category": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    "area": [
                        9492.0,
                        31395.0,
                        52728.0,
                        49452.0,
                        63000.0,
                        114219.0,
                        27456.0,
                        71323.0,
                        70312.0,
                        30300.0,
                        29750.0,
                        25185.0,
                        51282.0,
                        37647.0,
                    ],
                    "image_id": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                },
            },
        ]

        self.assertEqual(formatted_coco, formatted_coco_annotations)

    def test_format_evaluation_results(self):
        initial_results = {
            "iou_bbox": {
                "AP-IoU=0.50:0.95-area=all-maxDets=100": 1,
                "AP-IoU=0.50-area=all-maxDets=100": 0.035726408052333576,
                "AP-IoU=0.75-area=all-maxDets=100": 0.007142415373835251,
                "AP-IoU=0.50:0.95-area=small-maxDets=100": 0.005269675319251388,
                "AP-IoU=0.50:0.95-area=medium-maxDets=100": 0.0207959731645528,
                "AP-IoU=0.50:0.95-area=large-maxDets=100": 0.07910861863813597,
                "AR-IoU=0.50:0.95-area=all-maxDets=1": 0.004692312821993718,
                "AR-IoU=0.50:0.95-area=all-maxDets=10": 0.01904160489808846,
                "AR-IoU=0.50:0.95-area=all-maxDets=100": 0.04086775125540654,
                "AR-IoU=0.50:0.95-area=small-maxDets=100": 0.01977514455081171,
                "AR-IoU=0.50:0.95-area=medium-maxDets=100": 0.051431341425814486,
                "AR-IoU=0.50:0.95-area=large-maxDets=100": 0.16052469135802466,
            }
        }
        results = format_evaluation_results(initial_results)
        expected_results = {
            "AP-IoU=0.50:0.95-area=all-maxDets=100": 1.0,
            "AP-IoU=0.50-area=all-maxDets=100": 0.035726408052333576,
            "AP-IoU=0.75-area=all-maxDets=100": 0.007142415373835251,
            "AP-IoU=0.50:0.95-area=small-maxDets=100": 0.005269675319251388,
            "AP-IoU=0.50:0.95-area=medium-maxDets=100": 0.0207959731645528,
            "AP-IoU=0.50:0.95-area=large-maxDets=100": 0.07910861863813597,
            "AR-IoU=0.50:0.95-area=all-maxDets=1": 0.004692312821993718,
            "AR-IoU=0.50:0.95-area=all-maxDets=10": 0.01904160489808846,
            "AR-IoU=0.50:0.95-area=all-maxDets=100": 0.04086775125540654,
            "AR-IoU=0.50:0.95-area=small-maxDets=100": 0.01977514455081171,
            "AR-IoU=0.50:0.95-area=medium-maxDets=100": 0.051431341425814486,
            "AR-IoU=0.50:0.95-area=large-maxDets=100": 0.16052469135802466,
        }
        self.assertEqual(results, expected_results)

    def test_formatted_annotations(self):
        id_, cat_, ar_, box_ = (
            68,
            [3],
            [71574.0],
            [
                (
                    746.8235294117648,
                    1.2549019607843137,
                    213.17647058823525,
                    297.4117647058824,
                )
            ],
        )
        expected_results = [
            {
                "image_id": 68,
                "category_id": 3,
                "isCrowd": 0,
                "area": 71574.0,
                "bbox": [
                    746.8235294117648,
                    1.2549019607843137,
                    213.17647058823525,
                    297.4117647058824,
                ],
            }
        ]

        formatted_annotation_list = formatted_annotations(id_, cat_, ar_, box_)
        self.assertEqual(formatted_annotation_list, expected_results)


if __name__ == "__main__":
    unittest.main()
