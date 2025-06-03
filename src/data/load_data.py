import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.utils.logger import setup_logger


class OmniglotLoader:
    def __init__(
        self, background_path: str = "data/images_background", evaluation_path: str = "data/images_evaluation"
    ):
        """
        Initialize the Omniglot data loader.

        :param background_path: Path to background (training) dataset
        :param evaluation_path: Path to evaluation (test) dataset
        """
        self.background_path = background_path
        self.evaluation_path = evaluation_path
        self.label_encoder = LabelEncoder()

        self.logger = setup_logger("OmniglotLoader")

        self.trainx, self.trainy = None, None
        self.testx, self.testy = None, None

    def _read_alphabet(self, alphabet_directory_path: str) -> tuple[np.array, np.array]:
        """
        Reads all characters from a given alphabet directory.

        :param alphabet_directory_path: Path to alphabet directory
        :return: (image_paths, labels)
        """
        datax = []  # all file names of images
        datay = []  # all class names

        for character_dir in os.listdir(alphabet_directory_path):
            character_path = os.path.join(alphabet_directory_path, character_dir)
            if os.path.isdir(character_path):
                # Get all image files for this character
                image_files = [
                    os.path.join(character_path, f) for f in os.listdir(character_path) if f.endswith(".png")
                ]

                datax.extend(image_files)
                datay.extend([character_dir] * len(image_files))

        return np.array(datax), np.array(datay)

    def _read_images(self, base_directory: str, num_workers: int = 4) -> tuple[np.array, np.array]:
        """
        Reads all alphabets from base_directory using multithreading.

        :param base_directory: Root directory containing alphabets
        :param num_workers: Number of threads to use
        :return: (image_paths, labels)
        """
        alphabet_dirs = [
            os.path.join(base_directory, d)
            for d in os.listdir(base_directory)
            if os.path.isdir(os.path.join(base_directory, d))
        ]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self._read_alphabet, alphabet_dirs))

        datax = np.concatenate([r[0] for r in results])
        datay = np.concatenate([r[1] for r in results])

        return datax, datay

    def load_data(self, augment_with_rotations: bool = False) -> tuple[np.array, np.array, np.array, np.array]:
        """
        Load and prepare the Omniglot dataset.

        :param augment_with_rotations: Whether to augment data with 90° rotations
        :returns: (trainx, trainy, testx, testy)
        """
        self.logger.info("Loading training data...")
        self.trainx, self.trainy = self._read_images(self.background_path)
        self.logger.info("Loading test data...")
        self.testx, self.testy = self._read_images(self.evaluation_path)

        # Fit encoder on ALL possible labels (train + test)
        all_labels = np.concatenate([self.trainy, self.testy])
        self.label_encoder.fit(all_labels)

        # Transform both sets
        self.trainy = self.label_encoder.transform(self.trainy)
        self.testy = self.label_encoder.transform(self.testy)

        if augment_with_rotations:
            full_background_path = Path(__file__).parent.parent.parent / f"{self.background_path}_augmented"
            full_evaluation_path = Path(__file__).parent.parent.parent / f"{self.evaluation_path}_augmented"
            if not full_background_path.exists() or not full_evaluation_path.exists():
                full_background_path.mkdir(parents=True, exist_ok=True)
                full_evaluation_path.mkdir(parents=True, exist_ok=True)
                angles = [90, 180, 270]
                self.logger.info(f"Augmenting data with rotations {angles}...")
                self._augment_with_rotations(angles)

        return self.trainx, self.trainy, self.testx, self.testy

    def _augment_with_rotations(self, angles: list):
        """
        Augment the dataset by adding rotations of each image.
        :param angles: Angles to rotate on.
        """
        # For training data
        rotated_paths = []
        rotated_labels = []

        data_root = Path(__file__).parent.parent.parent / "data"

        for path, label in tqdm(zip(self.trainx, self.trainy), desc="Rotating Images", total=len(self.trainx)):
            for angle in angles:
                # Create new path for rotated image
                dirname, filename = os.path.split(path)

                parts = list(Path(dirname).parts)
                grandparent_dirname = parts[-3]
                parent_dirname = parts[-2]
                character_dirname = parts[-1]
                dirname = data_root / f"{str(grandparent_dirname)}_augmented" / parent_dirname / character_dirname
                if not dirname.exists():
                    dirname.mkdir(parents=True, exist_ok=True)

                basename, ext = os.path.splitext(filename)
                new_filename = f"{basename}_rot{angle}{ext}"
                new_path = os.path.join(dirname, new_filename)

                # Rotate and save the image
                img = Image.open(path)
                rotated_img = img.rotate(angle)
                rotated_img.save(new_path)

                rotated_paths.append(new_path)
                rotated_labels.append(label)

        # Add rotated data to original data
        self.trainx = np.concatenate([self.trainx, np.array(rotated_paths)])
        self.trainy = np.concatenate([self.trainy, np.array(rotated_labels)])
