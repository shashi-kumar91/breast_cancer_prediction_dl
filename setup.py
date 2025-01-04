from setuptools import find_packages,setup
from typing import List
HYPEN_E_DOT='-e'
def get_requirements(file_path:str)->List[str]:
    requirements=[]
    with open(file_path) as file_obj:
        requiremts=file_obj.readlines()
        [req.replace("\n","") for req in requirements]
        if HYPEN_E_DOT in requiremts:
            requiremts.remove(HYPEN_E_DOT)
        return requirements

setup(
    name='breast cancer prediction',
    version='0.01',
    author='shashi kumar',
    author_email='shashikumar9182805541@gmail.com',
    install_requires=get_requirements('requirements.txt'),
    packages=find_packages()
)