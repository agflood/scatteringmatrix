from setuptools import setup, find_packages
from scatteringmatrix import __version__

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='scatteringmatrix',
      version=__version__,
      description='Optical scattering matrix library',
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Andrew G. Flood',
      author_email='andrew.flood@mail.utoronto.ca',
      url='https://github.com/agflood/scatteringmatrix',
      license='MIT',
      classifiers=[
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering',
              'License :: OSI Approved :: MIT License',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.2',
              'Programming Language :: Python :: 3.3',
              'Programming Language :: Python :: 3.4',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              ],
      keywords='optics scattering matrix photonics',
      packages=find_packages(exclude=('tests', 'docs', 'sphinx')),
      py_modules=["scatteringmatrix"],
      install_requires=['numpy','scipy'],
      python_requires='>=3',
      zip_safe=False)
