from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='reactea',
    version='0.0.0',
    python_requires='>=3.6',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    include_package_data=True,
    zip_safe=False,
    author='BiSBII CEB University of Minho',
    author_email='jcorreia95@gmail.com',
    description='...',
    license='MIT',
    keywords='...',
    url='...',
    long_description=readme(),
    long_description_content_type='text/markdown',
)
