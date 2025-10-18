"""
Setup script for package installation
"""

from setuptools import setup, find_packages

setup(
    name='task-aware-medical-denoising',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@university.edu',
    description='Task-Aware Medical Image Denoising with Uncertainty Quantification',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/task-aware-medical-denoising',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Medical Science Apps.',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8',
    install_requires=[
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'opencv-python>=4.8.0',
        'Pillow>=10.0.0',
        'scikit-image>=0.21.0',
        'albumentations>=1.3.0',
        'numpy>=1.24.0',
        'scipy>=1.11.0',
        'pandas>=2.0.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'scikit-learn>=1.3.0',
        'pyyaml>=6.0',
        'tqdm>=4.65.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.3.0',
            'flake8>=6.0.0',
        ],
    },
)