# Documentation sample-plot generator.

add_executable(generate_samples generate_samples.cpp)

target_link_libraries(generate_samples PRIVATE approxLCD_static)

# Experiment driver used for quality/convergence studies (not shipped).
add_executable(lcd_experiment lcd_experiment.cpp)

target_link_libraries(lcd_experiment PRIVATE approxLCD_static)
