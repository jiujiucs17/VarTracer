from setuptools import setup, find_packages

setup(
    name='VarTracer',          # 包名，需唯一，PyPI 上不能重复
    version='0.1.0',                   # 语义化版本号
    author='Mengqi Zhang',
    author_email='jiujiuchangshou@gmail.com',
    description='A hybrid analysis tool for tracing variable dependencies across python codes',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',  # 告诉 PyPI README 用的是 markdown
    url='https://github.com/jiujiucs17/VarTracer',  # 项目主页，如 GitHub
    packages=find_packages(exclude=["tests*"]),     # 自动发现子包
    classifiers=[                                    # PyPI 分类标准
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',   # 或 BSD, GPL, Apache 等
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)