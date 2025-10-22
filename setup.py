from setuptools import setup, find_packages

setup(
    name='chemrxnsage',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'rdkit-pypi',
        'umap-learn',
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn',
        'torch',
        'SciencePlots',
        'seaborn',
        'plotly',
        'sentencepiece',
        'ipykernel',
        'vendi-score',
        'statsmodels',
        'notebook',
    ],
    entry_points={
        'console_scripts': [
            'chemrxnsage=chemrxnsage.main:main',
        ],
    },
)
