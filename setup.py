from setuptools import setup, find_packages

setup(name='spikelearn',
      version='0.1',
      description='The funniest joke in the world',
      url='https://github.com/EstevaoUyra/spikelearn',
      author='Estevao Uyra',
      author_email='estevao.uyra.pv@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
                'pandas',
                'seaborn',
                'scikit-learn',
                'scipy',
                ],
      zip_safe=False)
