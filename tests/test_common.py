import os
import json
import base64
import joblib
import pytest
from pathlib import Path
from box import ConfigBox
from beartype.roar import BeartypeCallHintParamViolation
from cnnClassifier.utils.common import (
    read_yaml,
    create_directories,
    save_json,
    load_json,
    save_bin,
    load_bin,
    get_size,
    decode_image,
    encode_image_into_base64,
)


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_data():
    return {"key": "value", "num": 42}


# ==========================
# create_directories tests
# ==========================
def test_create_directories_valid(tmp_dir):
    dirs = [tmp_dir / "dir1", tmp_dir / "dir2"]
    create_directories([str(d) for d in dirs])
    for d in dirs:
        assert d.exists() and d.is_dir()


def test_create_directories_single_string(tmp_dir):
    dir_path = tmp_dir / "single_dir"
    create_directories(str(dir_path))
    assert dir_path.exists() and dir_path.is_dir()


def test_create_directories_empty_string(tmp_dir):
    with pytest.raises(ValueError):
        create_directories("   ")


# ==========================
# read_yaml tests
# ==========================
def test_read_yaml_valid(tmp_dir, sample_data):
    yaml_path = tmp_dir / "config.yaml"
    import yaml

    yaml.dump(sample_data, open(yaml_path, "w"))
    loaded = read_yaml(yaml_path)
    assert loaded.key == sample_data["key"]
    assert loaded.num == sample_data["num"]


def test_read_yaml_invalid(tmp_dir):
    yaml_path = tmp_dir / "empty.yaml"
    import yaml

    yaml.dump("", open(yaml_path, "w"))
    with pytest.raises(ValueError, match="Invalid structure in yaml file"):
        read_yaml(yaml_path)


def test_read_yaml_empty(tmp_dir):
    yaml_path = tmp_dir / "empty.yaml"
    yaml_path.write_text("")
    with pytest.raises(ValueError, match="yaml file is empty"):
        read_yaml(yaml_path)


# ==================================
# JSON tests
# ==================================
def test_save_and_load_json(tmp_dir, sample_data):
    json_path = tmp_dir / "data.json"
    save_json(json_path, sample_data)
    loaded = load_json(json_path)
    assert isinstance(loaded, ConfigBox)
    assert loaded.key == sample_data["key"]
    assert loaded.num == sample_data["num"]


def test_save_json_invalid_type(tmp_dir):
    json_path = tmp_dir / "data.json"
    with pytest.raises(BeartypeCallHintParamViolation):
        save_json(json_path, data="not a dict")


def test_load_json_nonexistent(tmp_dir):
    fake_path = tmp_dir / "no.json"
    with pytest.raises(FileNotFoundError):
        load_json(fake_path)


# ==================================
# Binary (joblib) tests
# ==================================
def test_save_and_load_bin(tmp_dir, sample_data):
    bin_path = tmp_dir / "data.pkl"
    save_bin(sample_data, bin_path)
    loaded = load_bin(bin_path)
    assert loaded == sample_data


def test_load_bin_nonexistent(tmp_dir):
    fake_path = tmp_dir / "no.pkl"
    with pytest.raises(FileNotFoundError):
        load_bin(fake_path)


# ==========================
# get_size test
# ==========================
def test_get_size(tmp_dir):
    file_path = tmp_dir / "test.txt"
    file_path.write_text("12345")
    size_str = get_size(file_path)
    assert "~" in size_str


# ==========================
# Base64 image tests
# ==========================
def test_encode_and_decode_image(tmp_dir):
    img_path = tmp_dir / "img.txt"
    content = b"test image content"
    img_path.write_bytes(content)

    encoded = encode_image_into_base64(str(img_path))
    decoded_path = tmp_dir / "decoded_img.txt"
    decode_image(encoded, str(decoded_path))

    assert decoded_path.exists()
    assert decoded_path.read_bytes() == content
