from setuptools import setup

setup(name='snn',
      version='0.1',
      description='',
      url='',
      author='',
      author_email='',
      license='MIT',
      packages=['snn'],
      install_requires=[
            'tensorflow',
            'numpy',
            'keras'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      zip_safe=False)
