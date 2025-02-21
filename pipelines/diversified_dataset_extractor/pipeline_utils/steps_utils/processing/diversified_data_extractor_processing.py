import logging
import math
from typing import List, Optional

import PIL
import numpy as np
import picsellia
import requests
from PIL import Image, ImageOps
from picsellia import Client, Data, Datalake, DatasetVersion
from picsellia.sdk.asset import MultiAsset
from scipy.spatial import KDTree
from tqdm import tqdm

from src.picsellia_cv_engine import Colors
from src.picsellia_cv_engine.models.dataset.base_dataset_context import (
    TBaseDatasetContext,
)
from src.picsellia_cv_engine.models.steps.processing.dataset_version_creation_processing import (
    DatasetVersionCreationProcessing,
)
from pipelines.diversified_dataset_extractor.pipeline_utils.steps.model_loading.processing_diversified_data_extractor_model_loader import (
    EmbeddingModel,
)

logger = logging.getLogger("picsellia-engine")


class DiversifiedDataExtractorProcessing(DatasetVersionCreationProcessing):
    """
    This class is used to extract bounding boxes from images in a dataset version for a specific label.
    """

    def __init__(
        self,
        client: Client,
        datalake: Datalake,
        input_dataset_context: TBaseDatasetContext,
        output_dataset_version: DatasetVersion,
        embedding_model: EmbeddingModel,
        distance_threshold: float,
    ):
        super().__init__(
            client=client,
            datalake=datalake,
            output_dataset_version=output_dataset_version,
        )
        self.input_dataset_context = input_dataset_context
        self.embedding_model = embedding_model
        self.distance_threshold = distance_threshold

        self.skipped_similar_asset_number = 0
        self.skipped_error_asset_number = 0
        self.uploaded_asset_number = 0

    @property
    def input_dataset_version_size(self) -> int:
        return self.input_dataset_context.dataset_version.sync()["size"]

    @property
    def output_dataset_description(self) -> str:
        """
        Returns the description of the output dataset version.

        Returns:
            str: The description of the output dataset version.
        """
        return (
            f"Diversified dataset created from dataset version "
            f"'{self.input_dataset_context.dataset_version.version}' "
            f"(id: {self.input_dataset_context.dataset_version.id}). "
            f"The distance threshold used is {self.distance_threshold}"
        )

    @property
    def skipped_asset_number(self) -> int:
        return self.skipped_similar_asset_number + self.skipped_error_asset_number

    def compute_image_tensor(self, image: Image) -> np.ndarray:
        """
        Computes the tensor representation of an image using the embedding model.

        Args:
            image: The image to compute the tensor representation of.

        Returns:
            The tensor representation of the image.
        """
        preprocessed_image = self.embedding_model.apply_preprocessing(image)
        return self.embedding_model.encode_image(image=preprocessed_image)

    def fetch_and_prepare_image(self, image_url: str) -> Optional[PIL.Image.Image]:
        """
        Fetches an image from a URL and prepares it for processing.

        Args:
            image_url: The URL of the image to fetch.

        Returns:
            The fetched and prepared image.
        """
        try:
            with requests.get(image_url, stream=True) as response:
                response.raise_for_status()
                image = Image.open(response.raw)

                if image.getexif().get(0x0112, 1) != 1:
                    image = ImageOps.exif_transpose(image)

                return image.convert("RGB") if image.mode != "RGB" else image

        except requests.RequestException as e:
            logger.error(f"Failed to fetch or process image from {image_url}: {str(e)}")
            return None

    def get_tqdm_description_string(
        self, current_offset: int, batch_size: int, input_dataset_version_size: int
    ) -> str:
        """
        Returns a string to display in the tqdm progress bar.

        Args:
            current_offset: The current offset in the dataset version.
            batch_size: The maximum number of assets to process in a batch.
            input_dataset_version_size: The size of the dataset version being processed.

        Returns:
            The string to display in the tqdm progress bar.
        """
        current_batch_size = min(
            batch_size, input_dataset_version_size - current_offset
        )
        return (
            f"Batch {math.ceil(current_offset / batch_size + 1)}/"
            f"{math.ceil(input_dataset_version_size / batch_size)} "
            f"(size: {current_batch_size})"
        )

    def get_tqdm_postfix_string(self, is_batch_uploading: bool = False) -> str:
        """
        Returns a string to display in the tqdm progress bar.
        This string includes the number of added, skipped (errors) and skipped (too similar) assets.

        Args:
            is_batch_uploading: Whether the current batch is being added.
                This will add a suffix to the uploaded count.

        Returns:
            The string to display in the tqdm progress bar.
        """
        uploaded_suffix = "(↑)" if is_batch_uploading else ""
        return (
            f"Added {Colors.GREEN}{self.uploaded_asset_number}{uploaded_suffix}{Colors.ENDC}, "
            f"Skipped (errors) {Colors.RED}{self.skipped_error_asset_number}{Colors.ENDC}, "
            f"Skipped (too similar) {Colors.WARNING}{self.skipped_similar_asset_number}{Colors.ENDC}"
        )

    def process(self) -> None:
        self.update_output_dataset_version_description(
            description=self.output_dataset_description
        )
        self.update_output_dataset_version_inference_type(
            inference_type=self.input_dataset_context.dataset_version.type
        )
        self._process_dataset_version()

    def _get_assets_batch(
        self, limit: int, offset: int, retry_number: int
    ) -> MultiAsset:
        """
        Fetches a batch of assets from the dataset version.

        Args:
            limit: The maximum number of assets to retrieve.
            offset: The offset from which to start retrieving assets.
            retry_number: The number of retries to attempt if the batch fetch fails.

        Returns:
            The fetched batch of assets.
        """
        result_batch = None
        for i in range(0, retry_number):
            try:
                result_batch = self.input_dataset_context.get_assets_batch(
                    limit=limit, offset=offset
                )

            except Exception as e:
                logger.warning(f"Failed to fetch batch: {e}. Retrying ({i + 1}/3)...")
                continue

            else:
                break

        return result_batch

    def _is_tensor_unique(
        self, tensor: np.ndarray, kd_tree: KDTree, distance_threshold: float
    ) -> bool:
        """
        Checks if a tensor is unique in a KDTree.

        Args:
            tensor: The tensor to check.
            kd_tree: The KDTree to check against.
            distance_threshold: The distance threshold to consider a tensor unique.

        Returns:
            If the tensor is unique.
        """
        if kd_tree is None:
            return True

        nearest_neighbour_distance, _ = kd_tree.query(tensor)
        return nearest_neighbour_distance > distance_threshold

    def _process_batch(
        self, assets_batch: MultiAsset, kd_tree: Optional[KDTree], pbar: tqdm
    ) -> None:
        """
        Processes a batch of assets to filter out similar images and uploads them to the output dataset version.

        Args:
            assets_batch: The batch of assets to process.
        """
        batch_to_upload: List[Data] = []

        for asset in assets_batch:
            success = False
            attempts = 0
            max_retries = 2

            while not success and attempts < max_retries:
                try:
                    image = self.fetch_and_prepare_image(asset.url)

                    if not image:
                        self.skipped_error_asset_number += 1
                        break

                    tensor = self.compute_image_tensor(image=image)

                    if kd_tree is None:
                        kd_tree = KDTree(tensor)
                        batch_to_upload.append(asset.get_data())
                        success = True

                    else:
                        if self._is_tensor_unique(
                            tensor=tensor,
                            kd_tree=kd_tree,
                            distance_threshold=self.distance_threshold,
                        ):
                            kd_tree = KDTree(np.vstack([kd_tree.data, tensor]))
                            batch_to_upload.append(asset.get_data())
                            success = True

                        else:
                            self.skipped_similar_asset_number += 1
                            break

                except Exception as e:
                    self.skipped_error_asset_number += 1
                    attempts += 1

                    if attempts < 2:
                        logger.warning(f"Retrying asset due to exception: {e}")
                    else:
                        logger.warning(
                            f"Skipped asset due to exception after retry: {e}"
                        )

                finally:
                    if success or attempts >= 2:
                        pbar.update(1)

        pbar.set_postfix_str(s=self.get_tqdm_postfix_string(is_batch_uploading=True))

        if len(batch_to_upload) > 0:
            self._add_data_to_dataset_version(data=batch_to_upload)
            self.uploaded_asset_number += len(batch_to_upload)

        pbar.set_postfix_str(s=self.get_tqdm_postfix_string())

    def _process_dataset_version(self) -> None:
        """
        Processes the images in the dataset version to filter out similar images and uploads them to
        the output dataset version.
        """
        kd_tree: Optional[KDTree] = None

        input_dataset_version_size = self.input_dataset_version_size

        batch_size = 10
        current_offset = 0
        batch_fetching_retry_number = 3

        picsellia_logger_original_level = picsellia.logger.level
        picsellia.logger.setLevel(logging.WARNING)

        with tqdm(
            total=input_dataset_version_size, unit="asset", colour="WHITE"
        ) as pbar:
            pbar.set_postfix_str(s=self.get_tqdm_postfix_string())

            while current_offset < input_dataset_version_size:
                pbar.set_description(
                    self.get_tqdm_description_string(
                        current_offset=current_offset,
                        batch_size=batch_size,
                        input_dataset_version_size=input_dataset_version_size,
                    )
                )

                assets_batch = self._get_assets_batch(
                    limit=batch_size,
                    offset=current_offset,
                    retry_number=batch_fetching_retry_number,
                )

                if assets_batch is None:
                    logger.error(
                        f"Failed to fetch batch after {batch_fetching_retry_number} retries. "
                        f"Skipping to the next batch."
                    )
                    self.skipped_error_asset_number += batch_size

                else:
                    self._process_batch(
                        assets_batch=assets_batch, kd_tree=kd_tree, pbar=pbar
                    )

                current_offset += batch_size

            pbar.set_description(
                f"Finished uploading {math.ceil(current_offset / batch_size)} batches"
            )

        # Set picsellia logger's level back
        picsellia.logger.setLevel(picsellia_logger_original_level)
