cmake_minimum_required(VERSION 3.8)
project(rbe595_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch models tags worlds
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
