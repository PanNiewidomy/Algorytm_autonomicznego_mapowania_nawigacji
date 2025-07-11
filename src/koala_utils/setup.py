from setuptools import find_packages, setup

package_name = 'koala_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            'battery_status = koala_utils.battery_status:main',
            'init_pose = koala_utils.final_inicjalizer:main',
            'spin_robot = koala_utils.spin_robot:main',
            'gui_debugger = koala_utils.debug_gui:main',
        ],
    },
)
