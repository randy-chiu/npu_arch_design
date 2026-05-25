create_clock -name core_clk -period 10.000 [get_ports clk]
set_clock_uncertainty 0.100 [get_clocks core_clk]
set_input_delay 0.200 -clock core_clk [remove_from_collection [all_inputs] [get_ports clk]]
set_output_delay 0.200 -clock core_clk [all_outputs]
