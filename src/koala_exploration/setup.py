from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'koala_exploration'

setup(
    name=package_name,  # Changed from package_name to use hyphen
    version='0.0.0',
    packages=find_packages(exclude=['test']) + [f'{package_name}.utils']+[f'{package_name}.nodes'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*')),
        (os.path.join('share', package_name, 'maps'), glob('maps/*')),
        (os.path.join('share', package_name, 'behavior_trees'), glob('behavior_trees/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jakub',
    maintainer_email='124900594+PanNiewidomy@users.noreply.github.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'explorer_node = koala_exploration.explorer_node:main',
            'WFD_node = koala_exploration.nodes.WFD_node:main',
            'FFD_node = koala_exploration.nodes.FFD_node:main'
        ],
    },
)