from setuptools import setup, find_packages

setup(
    name="VarTracer",
    version="0.1.8",
    description="A Python-based tool for dynamic code execution tracing and dependency analysis.",
    long_description=open("readme.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Mengqi Zhang",
    author_email="jiujiuchangshou@gmail.com",
    url="https://github.com/jiujiucs17/VarTracer",
    packages=find_packages(include=["VarTracer", "VarTracer.*"]),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)