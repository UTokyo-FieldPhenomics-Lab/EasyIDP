import pathlib
from setuptools import setup, find_packages

here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

with open('easyidp/__init__.py', encoding='utf-8') as fid:
    for line in fid:
        if line.startswith('__version__'):
            VERSION = line.strip().split()[-1][1:-1]
            break


def parse_requirements_file(filename):
    with open(filename, encoding='utf-8') as fid:
        requires = [line.strip() for line in fid.readlines() if line]

    return requires


INSTALL_REQUIRES = parse_requirements_file('requirements/default.txt')
# The `requirements/extras.txt` file is explicitely omitted because
# it contains requirements that do not have wheels uploaded to pip
# for the platforms we wish to support.
extras_require = {
    dep: parse_requirements_file('requirements/' + dep + '.txt')
    for dep in ['docs', 'test', 'build']
}

setup(
    name="easyidp", 
    version=VERSION,
    author="Haozhou Wang", 
    author_email="howcanoewang@gmail.com",
    description="A handy tool for dealing with region of interest (ROI) on the image reconstruction (Metashape & Pix4D) outputs, mainly in agriculture applications",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    extras_require=extras_require,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires='>=3.6, <4',
    project_urls={
        'Documentation': 'https://easyidp.readthedocs.io/en/latest/',
        'Source': 'https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP',
        'Tracker': 'https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/issues',
        'forum': 'https://github.com/UTokyo-FieldPhenomics-Lab/EasyIDP/discussions'
    },
)