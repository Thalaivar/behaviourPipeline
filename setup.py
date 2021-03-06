from setuptools import setup

setup(
    name='behaviourPipeline',
    version='0.3.9',
    author='Dhruv Laad',
    author_email='dhruvlaad@gmail.com',
    packages=['behaviourPipeline'],
    license='LICENSE.txt',
    install_requires=[
       "ray",
       "umap-learn",
       "hdbscan",
       "catboost",
       "pyyaml",
       "joblib",
       "scikit_learn",
       "opencv_python", 
       "ffmpeg_python",
       "tqdm",
       "seaborn",
       "h5py"
    ],
)