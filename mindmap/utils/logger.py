import logging

try:
    logging.basicConfig(
        filename="/var/log/mindmap_creator.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    suppress_packages = [
       "httpx", "openai", "urllib3", "httpcore"
    ]
except PermissionError:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    suppress_packages = [
        "docling", "httpx", "openai", "urllib3", "docling_ibm_models", "httpcore"
    ]

logger = logging.getLogger()


for package in suppress_packages:
    logging.getLogger(package).setLevel(logging.CRITICAL)