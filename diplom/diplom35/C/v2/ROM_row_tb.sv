`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 16.01.2025 16:20:50
// Design Name: 
// Module Name: ROM_row_tb
// Project Name: LST-1 model
//////////////////////////////////////////////////////////////////////////////////

module ROM_row_tb();

localparam INT_BITS = 5;
localparam FRC_BITS = 7;
parameter PERIOD = 10ns;


logic [(INT_BITS+FRC_BITS - 1):0] Dout_r0, Dout_r1, Dout_c27, Dout_w_out;
logic [4:0] addr_row_col;
logic [9:0] addr_w_out;
logic CLK;

ROM_row #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(0)) rom_r0(.address(addr_row_col), .dout(Dout_r0), .clk(CLK));
ROM_row #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(1)) rom_r1(.address(addr_row_col), .dout(Dout_r1), .clk(CLK));

ROM_col #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(27)) rom_c27(.address(addr_row_col), .dout(Dout_c27), .clk(CLK));

ROM_w_out #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(0)) rom_w(.address(addr_w_out), .dout(Dout_w_out), .clk(CLK));

always begin
  CLK = 1'b0;
  #(PERIOD/2) CLK = 1'b1;
  #(PERIOD/2);
end

initial begin
   integer i;

   for (i = 0; i < 28; i=i+1) begin
      addr_row_col = i;
      addr_w_out = i;
      #10;
   end

   for (i = 28; i < 784; i=i+1) begin
      addr_w_out = i;
      #10;
   end
end 

endmodule
