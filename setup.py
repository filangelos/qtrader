from setuptools import setup, find_packages

setup(name='qtrader',
      version='0.0.1',
      description='Reinforcement Learning for Portfolio Management',
      url='https://github.com/filangel/qtrader',
      author='filangel',
      author_email='filos.angel@gmail.com',
      license='Apache-2.0',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'scipy',
          'pandas',
          'matplotlib',
          'seaborn',
          'scikit-learn',
          'quandl',
          'gym',
          'tensorflow',
          'torch',
          'statsmodels',
          'pyyaml',
          'bs4'
      ],
      test_suite='tests'
      )
