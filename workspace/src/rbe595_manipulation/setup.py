from glob import glob

from setuptools import find_packages, setup

package_name = "rbe595_manipulation"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Philip Smith",
    maintainer_email="philip@randomsmiths.com",
    description="Dead-reckoning pick/place/shake action server for the spice bottle demo",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "pick_place_spice_action_server = "
            "rbe595_manipulation.pick_place_spice_action_server:main",
            "test_shake = rbe595_manipulation.test_shake:main",
        ],
    },
)
