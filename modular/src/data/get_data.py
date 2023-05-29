import os
import requests
import shutil
from pathlib import Path


def get_data(
    request_url: str,
    data_path: str,
    unzip: bool = True,
    clean_zip: bool = True,
):
    """_summary_
    Download, saves and unzips data to DATA_PATH directory for a given request_url
    Assumes request url ends with zip file name

    Args:
        request_url (str): Url to download zipped data from
        data_path: (Path): Root path for download_path
        unzip (bool, optional): Unzips downloaded zip by default. Defaults to True.
        clean_zip (bool, optional): Deletes downloaded zip after unzip. Defaults to True.

    Returns:
        str: Data downlaoded directory
    """

    data_path = Path(data_path)
    file_name = request_url.split("/")[-1]
    folder_name = file_name.split(".")[0]

    download_dir = Path(f"{data_path}/{folder_name}")
    file_path = Path(f"{data_path}/{file_name}")

    print(f"Image download directory: {download_dir}")
    print(f"Zip path: {file_path}")

    if data_path.is_dir():
        print(f"{data_path} directory exists")
    else:
        print(f"Did not find {data_path}, creating one...")

    if download_dir.is_dir():
        print(f"{download_dir} directory exists")
    else:
        print(f"Did not find {download_dir}, creating one...")
        download_dir.mkdir(parents=True, exist_ok=True)

    # Download zip
    with open(file_path, "wb") as f:
        request = requests.get(url=request_url)
        print(f"Downloading data from{request_url}...")
        f.write(request.content)

    # Unzip pizz, steak, sushi data
    if unzip:
        print(f"Unzip data")
        shutil.unpack_archive(file_path, download_dir)

    if clean_zip:
        os.remove(file_path)
        print(f"{file_path} cleand after unzip")

    return folder_name


get_data(
    request_url="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
    data_path="/Users/jayaprakashsivagami/Documents/Tech/ML/pytorch/computer_vision/data",
)
