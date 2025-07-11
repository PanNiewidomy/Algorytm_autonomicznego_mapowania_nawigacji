from setuptools import find_packages, setup

package_name = 'koala_teleop'

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
            'vel_mux = koala_teleop.vel_mux:main',
            'distance = koala_teleop.distance:main',
            'camera_preview = koala_teleop.cameraPreview:main',
            'pose = koala_teleop.Pose:main',  #
            'siatka = koala_teleop.SiatkaMoje:main',
        ],
    },
)
