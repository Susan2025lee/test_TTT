import setuptools

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
# Filter out development dependencies
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()
install_requires = [req for req in requirements if not req.startswith(('pytest', 'black', 'flake8'))]


setuptools.setup(
    name="tic_tac_toe_agent", # Replace with your desired package name
    version="0.1.0",         # Initial version
    author="Your Name",       # Replace with your name
    author_email="your.email@example.com", # Replace with your email
    description="A Tic Tac Toe game environment and AI agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/yourrepository", # Replace with your project URL
    package_dir={"": "tic_tac_toe/src"}, # Tell setuptools packages are under src
    packages=setuptools.find_packages(where="tic_tac_toe/src"), # Find packages in src
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7", # Specify minimum Python version
    install_requires=install_requires, # List runtime dependencies
) 