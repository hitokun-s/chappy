#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
# from distutils.core import setup

version = '0.1dev'

# pypandocを使ってREADME.mdをrstに変換する。最初からrstで書いた場合は不要。
try:
    import pypandoc

    read_md = lambda f: pypandoc.convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(
        name='chappy',
        version=version,
        packages=find_packages(exclude=['tests']),
        description="Helper Tool for Machine Learning",
        license='MIT',
        author="Hitoshi Wada",
        author_email="jgpuauno@gmail.com",
        url='https://github.com/hitokun-s/chappy',
        long_description=read_md('README.md'),
        install_requires=[],
        tests_require=["nose"],
        classifiers=["Development Status :: 2 - Pre-Alpha"]
)
