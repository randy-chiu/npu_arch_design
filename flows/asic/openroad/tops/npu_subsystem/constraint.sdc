create_clock -name subsystem_clk -period 10.000 [get_ports clk]
set_clock_uncertainty 0.100 [get_clocks subsystem_clk]
set_input_delay 0.200 -clock subsystem_clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0.200 -clock subsystem_clk [all_outputs]
