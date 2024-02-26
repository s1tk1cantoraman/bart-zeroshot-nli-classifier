from setuptools import setup, find_packages

setup(
    name='bart_zeroshot_demo',
    version='0.1.1',
    packages=find_packages(include=['bart_zeroshot', 'bart_zeroshot.*']),
    package_dir={'': 'src'},
    package_data={'bart_zeroshot': ['data/dialogue.json'],
    },
    url='',
    license='',
    author='Sıtkı Can Toraman',
    author_email='sitkicantoraman@gmail.com',
    description='BART Zero Shot Text Classifier Demo'
)
