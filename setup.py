import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
     name='PyTE',  
     version='0.2',
     author="Kelechi Ikegwu, Jacob Trauger, Robert Brunner",
     author_email="ikegwu2@illinois.edu, jtt2@illinois.edu, bigdog@illinois.edu",
     description="An open source implementation to compute bi-variate Transfer Entropy.",
     long_description=long_description,
   long_description_content_type="text/markdown",
     url="https://github.com/lcdm-uiuc/PyTE",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: University of Illinois/NCSA Open Source License",
         "Operating System :: OS Independent",
     ],
 )
