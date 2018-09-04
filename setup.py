from setuptools import setup, find_packages
from scatteringmatrix import __version__

setup(name='scatteringmatrix',
      version=__version__,
      description='Optical scattering matrix library',
      author='Andrew G. Flood',
      author_email='andrew.gary.flood@gmail.com',
      url='https://github.com/agflood/scatteringmatrix',
      license='MIT',
      classifiers=[
              'Development Status :: 2 - Pre-Alpha',
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