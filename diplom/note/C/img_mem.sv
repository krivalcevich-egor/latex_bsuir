`timescale 1ns / 1ps

module img_mem 
  import nn_param_pkg::*;
(
    input  logic                           clk, // clock
    input  logic                           we, // clock
    input  logic                           en, // clock
    input  logic [           ADDR_OUT-1:0] addr,
    input  logic [INT_BITS + FRC_BITS-1:0] din,
    output logic [INT_BITS + FRC_BITS-1:0] dout
);

    (* rom_style = "block" *) reg [INT_BITS + FRC_BITS-1:0] data [WIDTH-1:0];

     always @(posedge clk) begin
       if(we & en) begin
         data[addr] <= din;
       end
     end

    assign dout = (!we & en)? data[addr] : 0;
endmodule
