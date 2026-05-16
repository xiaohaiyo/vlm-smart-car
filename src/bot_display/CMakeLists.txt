cmake_minimum_required(VERSION 3.8)
project(bot_display)

set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(bot_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(display_node src/display_node.cpp)
target_include_directories(display_node PRIVATE include ${OpenCV_INCLUDE_DIRS})
ament_target_dependencies(display_node rclcpp sensor_msgs std_msgs bot_interfaces cv_bridge image_transport)
target_link_libraries(display_node ${OpenCV_LIBS})

install(TARGETS display_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS bot_display/gui_node.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch DESTINATION share/${PROJECT_NAME})

ament_package()
