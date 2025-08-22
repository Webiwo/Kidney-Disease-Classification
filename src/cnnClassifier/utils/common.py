import os
import yaml
import json
import joblib
import base64
from box import ConfigBox
from box.exceptions import BoxValueError
from beartype import beartype
from pathlib import Path
from beartype.typing import Any, Union, Annotated, Sequence
from cnnClassifier import logger


NonEmptyStr = Annotated[str, lambda s: s.strip() != ""]
NonEmptyPath = Annotated[Path, lambda p: str(p).strip() != ""]


@beartype
def read_yaml(path_to_yaml: NonEmptyPath) -> ConfigBox:
    """Read YAML file and return as ConfigBox.

    Args:
        path_to_yaml (Path): Path to YAML file.

    Raises:
        ValueError: If YAML file is empty or invalid.

    Returns:
        ConfigBox: ConfigBox object representing YAML content.
    """

    try:
        with open(path_to_yaml) as f:
            content = yaml.safe_load(f)
            if content is None:
                raise ValueError("yaml file is empty")
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("Invalid structure in yaml file")


@beartype
def create_directories(
    path_to_directories: Union[NonEmptyStr, Sequence[NonEmptyStr]], verbose: bool = True
) -> None:
    """Create a list of directories.

    Args:
        path_to_directories (Sequence[str]): List or tuple of directory paths.
        verbose (bool, optional): If True, log created directories. Defaults to True.
    """

    if isinstance(path_to_directories, str):
        path_to_directories = [path_to_directories]

    for path in path_to_directories:
        if not path.strip():
            raise ValueError("Empty path provided")
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")


@beartype
def save_json(path: NonEmptyPath, data: dict) -> None:
    """Save data to JSON file.

    Args:
        path (Path): Path to JSON file.
        data (dict): Data to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


@beartype
def load_json(path: NonEmptyPath) -> ConfigBox:
    """Load data from JSON file as ConfigBox.

    Args:
        path (Path): Path to JSON file.

    Returns:
        ConfigBox: JSON data as ConfigBox.
    """

    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)


@beartype
def save_bin(data: Any, path: NonEmptyPath) -> None:
    """Save data as a binary file.

    Args:
        data (Any): Object to save.
        path (Path): Path to binary file.
    """

    joblib.dump(data, path)
    logger.info(f"Binary file saved at: {path}")


@beartype
def load_bin(path: NonEmptyPath) -> Any:
    """Load binary data from file.

    Args:
        path (Path): Path to binary file.

    Returns:
        Any: Object stored in the file.
    """

    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data


@beartype
def get_size(path: NonEmptyPath) -> str:
    """Get size of file in KB.

    Args:
        path (Path): Path of the file.

    Returns:
        str: File size in KB.
    """

    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"


@beartype
def decode_image(imgstring: NonEmptyStr, file_name: NonEmptyStr) -> None:
    """Decode base64 string into image file.

    Args:
        imgstring (str): Base64-encoded string.
        file_name (str): File path to save decoded image.
    """

    imgdata = base64.b64decode(imgstring)
    with open(file_name, "wb") as f:
        f.write(imgdata)


@beartype
def encode_image_into_base64(cropped_image_path: NonEmptyStr) -> bytes:
    """Encode image file to base64.

    Args:
        cropped_image_path (str): Path to image file.

    Returns:
        bytes: Base64-encoded image.
    """

    with open(cropped_image_path, "rb") as f:
        return base64.b64encode(f.read())
