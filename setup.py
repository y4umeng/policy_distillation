import setuptools

DISTNAME = "src"  # Consider using a better package name than "src"
DESCRIPTION = "A Deep Learning Toolkit for Policy Distillation."
AUTHOR = "y4umeng"

setuptools.setup(
    name=DISTNAME,
    packages=setuptools.find_packages(where="src"),  # Look for packages in src/
    package_dir={"": "src"},  # Tell setuptools that packages are under src/
    version="0.1",
    description=DESCRIPTION,
    author=AUTHOR,
)