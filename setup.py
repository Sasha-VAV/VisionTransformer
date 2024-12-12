from setuptools import setup, find_packages

setup(
    name='VisionTransformer',
    py_modules=['VisionTransformer'],
    version='0.1.0',
    description='Visual transformer module with training, testing, and executing as CV model',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    readme="README.md",
    python_requires='>=3.10',
    author='Sashavav',
    url='https://github.com/Sasha-VAV/VisualTransformer',
    packages=find_packages(exclude=['tests*']),
    extras_require={"dev": ["black"]}
)
