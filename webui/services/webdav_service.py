# -*- coding: utf-8 -*-
import os
import shutil
import time
from xml.etree import ElementTree as ET

import requests
from requests.auth import HTTPBasicAuth

from config_manager import load_config, save_config


class WebDAVClient:
    def __init__(self, base_url: str, username: str, password: str):
        if not base_url:
            raise ValueError("WebDAV URL 不能为空。")
        self.base_url = base_url.rstrip("/") + "/"
        self.auth = HTTPBasicAuth(username or "", password or "")
        self.headers = {
            "User-Agent": "AI Novel Generator WebDAV Client",
            "Accept": "*/*",
        }
        self.ns = {"d": "DAV:"}

    def _get_url(self, path: str) -> str:
        return self.base_url + path.lstrip("/")

    def list_directory(self, path: str = "") -> bool:
        headers = self.headers.copy()
        headers["Depth"] = "1"
        response = requests.request("PROPFIND", self._get_url(path), headers=headers, auth=self.auth)
        response.raise_for_status()
        return True

    def directory_exists(self, path: str) -> bool:
        headers = self.headers.copy()
        headers["Depth"] = "0"
        response = requests.request("PROPFIND", self._get_url(path), headers=headers, auth=self.auth)
        if response.status_code != 207:
            return False
        root = ET.fromstring(response.content)
        resource_type = root.find(".//d:resourcetype", namespaces=self.ns)
        return resource_type is not None and resource_type.find("d:collection", namespaces=self.ns) is not None

    def create_directory(self, path: str) -> bool:
        response = requests.request("MKCOL", self._get_url(path), auth=self.auth, headers=self.headers)
        response.raise_for_status()
        return True

    def ensure_directory_exists(self, path: str) -> bool:
        path = path.rstrip("/")
        if not path or self.directory_exists(path):
            return True
        parent_dir = os.path.dirname(path)
        if parent_dir:
            self.ensure_directory_exists(parent_dir)
        return self.create_directory(path)

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        if not os.path.isfile(local_path):
            raise FileNotFoundError(local_path)
        with open(local_path, "rb") as handle:
            response = requests.put(self._get_url(remote_path), data=handle, auth=self.auth, headers=self.headers)
        response.raise_for_status()
        return True

    def download_file(self, remote_path: str, local_path: str) -> bool:
        response = requests.get(self._get_url(remote_path), auth=self.auth, headers=self.headers, stream=True)
        response.raise_for_status()
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        _backup_existing(local_path)
        with open(local_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        return True


def save_webdav_config(config_file: str, url: str, username: str, password: str) -> dict:
    config = load_config(config_file)
    config["webdav_config"] = {
        "webdav_url": (url or "").strip(),
        "webdav_username": (username or "").strip(),
        "webdav_password": password or "",
    }
    save_config(config, config_file)
    return config


def test_connection(config_file: str, url: str, username: str, password: str) -> str:
    WebDAVClient(url, username, password).list_directory()
    save_webdav_config(config_file, url, username, password)
    return "WebDAV 连接成功。"


def backup_config(config_file: str, url: str, username: str, password: str) -> str:
    save_webdav_config(config_file, url, username, password)
    client = WebDAVClient(url, username, password)
    target_dir = "AI_Novel_Generator"
    client.ensure_directory_exists(target_dir)
    client.upload_file(config_file, f"{target_dir}/config.json")
    return "配置备份成功。"


def restore_config(config_file: str, url: str, username: str, password: str) -> str:
    save_webdav_config(config_file, url, username, password)
    client = WebDAVClient(url, username, password)
    target_dir = "AI_Novel_Generator"
    client.download_file(f"{target_dir}/config.json", config_file)
    return "配置恢复成功。"


def _backup_existing(local_path: str):
    if not os.path.exists(local_path):
        return
    directory = os.path.dirname(os.path.abspath(local_path))
    backup_dir = os.path.join(directory, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    base_name, extension = os.path.splitext(os.path.basename(local_path))
    timestamp = time.strftime("%Y%m%d%H%M%S")
    shutil.copy2(local_path, os.path.join(backup_dir, f"{base_name}_{timestamp}_bak{extension}"))

