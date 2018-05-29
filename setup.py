from setuptools import setup

setup(name='quinton',
      version='1.0.0',
      description='Neural networks for tomography in Python.',
      url='https://github.com/glassnotes/NN_QTomo',
      author='Olivia Di Matteo',
      author_email='odimatte@uwaterloo.ca',
      license='MIT',
      package_dir={'': 'src'},
      packages=['quinton'],
      zip_safe=False)
