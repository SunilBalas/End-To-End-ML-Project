from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str]:
    '''
    Retrieves a list of package requirements from a specified requirements file.

    Parameters:
        file_path (str): The path to the requirements file to be parsed.

    Returns:
        List[str]: A list of package names and versions extracted from the requirements file.

    '''
    requirements = []
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            
        return requirements

# Project Setup
setup(
    name='mlproject',
    version='0.0.1',
    author='Sunil',
    author_email='balassunil606@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)