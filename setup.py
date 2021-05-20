from setuptools import find_packages, setup

install_requires = ["calamus>=0.3.8", "gorilla", "numpy"]
packages = find_packages()
version_file = open("VERSION")

setup(
    name="mlschema-converters",
    description="MLSchema Converter",
    keywords="MLSchema",
    license="BSD 3.0",
    author="Viktor Gal",
    author_email="viktor.gal@maeth.com",
    install_requires=install_requires,
    packages=packages,
    tests_require=["pytest>=4.0.0"],
    zip_safe=False,
    include_package_data=True,
    platforms="any",
    version=version_file.read().strip(),
    classifiers=[
        "Environment :: Web Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Development Status :: 4 - Beta",
    ],
)
