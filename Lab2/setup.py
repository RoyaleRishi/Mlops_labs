from setuptools import setup, find_packages

setup(
    name='trainer',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scikit-learn==0.23.2',
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'google-cloud-storage>=1.28.0',
    ],
    description='Heart Disease Prediction Model Training Package',
    author='Your Name',
    python_requires='>=3.7',
)