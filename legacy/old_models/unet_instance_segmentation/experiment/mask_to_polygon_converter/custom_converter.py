import glob
import os
from typing import List

from .abstract_converter import AbstractConverter


class CustomConverter(AbstractConverter):
    def _get_masks_filepaths_of_image(
        self, img_filename: str, label_dir: str
    ) -> List[str]:
        """Find all the paths of the masks linked to an image.
        Here we are looking for the mask that has the same
        filename with an extension that may be different.

        :param img_filename: the filename of the image whose masks are being searched for
        :param label_dir: the folder in which to look for masks,
        this folder being a sub folder of "self.masks_dir", i.e. a label name
        :return: the list of all the full paths of the masks linked to the image
        with the name img_filename
        """
        dir_path = os.path.join(self._masks_dir, label_dir)
        img_filename_without_extension = img_filename.split(".")[0]
        character_to_match = img_filename_without_extension + ".*"
        return glob.glob(os.path.join(glob.escape(dir_path), character_to_match))
