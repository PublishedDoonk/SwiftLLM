from setuptools import setup, find_packages

setup(
    name='SwiftLLM',
    version='0.0.2',
    author='Zachary Ivie',
    author_email='zachary.ivie@gmail.com',
    description='A python package to access most common foundation models easily.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/PublishedDoonk/SwiftLLM',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent'
    ],
    python_requires='>=3.6',
    install_requires=[
        'openai',
        'requests'
    ],
)