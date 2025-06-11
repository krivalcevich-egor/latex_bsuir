`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Create Date: 30.12.2024 10:17:16
// Design Name: 
// Module Name: tb_tanh_func
// Project Name: 
//////////////////////////////////////////////////////////////////////////////////


module tb_tanh_func();

localparam INT_SIZE = 5;
localparam FRC_SIZE = 7;

logic [(INT_SIZE+FRC_SIZE - 1):0] x;
logic [(INT_SIZE+FRC_SIZE - 1):0] y;

tanh_function #(.INT_SIZE(INT_SIZE), .FRC_SIZE(FRC_SIZE)) tanh_inst(.X(x), .Y(y));

initial begin
//for (int i = -1024; i < 0; i++) begin
//    x = i;
//    #10;
//end
			
//for (int i = 0; i < 1023; i++) begin
//    x = i;
//    #10;
//end
x = 12'h07e;
#30;
x = 12'hfd2;
#30;
x = 12'hf00;
#30;
end 

endmodule
