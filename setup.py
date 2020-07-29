import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='pyrcn-amt',
    version='0.0.1',
    author='Peter Steiner',
    author_email='peter.steiner@tu-dresden.de',
    description='Automatic Music Transcription using PyRCN',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/TUD-STKS/Automatic-Music-Transcription',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
    ],
    keywords='PyRCN, Automatic Music Transcription',
    project_urls={
        'Documentation': 'https://github.com/TUD-STKS/Automatic-Music-Transcription',
        'Funding': 'https://github.com/TUD-STKS/Automatic-Music-Transcription',
        'Source': 'https://github.com/TUD-STKS/Automatic-Music-Transcription',
        'Tracker': 'https://github.com/TUD-STKS/Automatic-Music-Transcription',
    },
    install_requires=[
        'scikit-learn>=0.22.1',
        'numpy>=1.18.1',
        'scipy>=1.2.0',
        'joblib>=0.13.2',
    ],
    python_requires='>=3.6',
)
