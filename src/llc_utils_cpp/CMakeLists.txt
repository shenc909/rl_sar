cmake_minimum_required(VERSION 3.8)
project(llc_utils_cpp)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rclcpp REQUIRED)

# LibTorch (downloaded to library/inference_runtime/libtorch by build.sh's
# download_inference_runtime.sh on first build). Used to parallelise the
# per-point downsampling math in pointcloud_decimator.
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(LIBTORCH_DIR "${PROJECT_ROOT_DIR}/library/inference_runtime/libtorch")
set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR};${CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Resolve the libtorch shared objects at runtime via install RPATH.
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "${LIBTORCH_DIR}/lib")
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
endif()

add_executable(hesai_relay src/hesai_relay.cpp)
ament_target_dependencies(hesai_relay
        rclcpp
        sensor_msgs
    )
target_include_directories(hesai_relay PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(hesai_relay PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS hesai_relay
  DESTINATION lib/${PROJECT_NAME})

add_executable(pointcloud_decimator src/pointcloud_decimator.cpp)
ament_target_dependencies(pointcloud_decimator
        rclcpp
        sensor_msgs
    )
target_include_directories(pointcloud_decimator PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${TORCH_INCLUDE_DIRS})
target_link_libraries(pointcloud_decimator ${TORCH_LIBRARIES})
target_compile_features(pointcloud_decimator PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS pointcloud_decimator
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
