cmake_minimum_required(VERSION 3.8)
project(rbe595_manipulation_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PickPlaceSpice.action"
)

ament_package()
