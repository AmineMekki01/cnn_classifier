import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "cnn_classifier"
AUTHOR_USER_NAME = "AmineMekki01"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "amine.mekki@mines-ales.org"

setuptools.setup(
    name = SRC_REPO,
    version = __version__, 
    author = AUTHOR_USER_NAME,
    author_email = AUTHOR_EMAIL,
    description = "A small package for chicken disease classification.",
    long_description = "Long description of the project", 
    long_description_content_type = "text/markdown",
    url=f"https://github.vom/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls = { 
        "Bug Tracker" : f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",

    },
    package_dir = {"" : "src"},
    packages= setuptools.find_packages(where = "src"),
    )