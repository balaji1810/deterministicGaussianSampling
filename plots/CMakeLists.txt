# Documentation sample-plot generator.

add_executable(generate_samples generate_samples.cpp)

target_link_libraries(generate_samples PRIVATE approxLCD_static)
