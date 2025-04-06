`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// WEIGHT MEMORY (ROM)
//////////////////////////////////////////////////////////////////////////////////
  
  (* dont_touch = "yes" *) (* keep_hierarchy = "yes" *) 

module ROM_row #(
    parameter int INT_BITS = 6,  // integer part
    parameter int FRC_BITS = 7, // fractional part
    parameter int N = 0 // block number
)(
    input logic clk, // clock
    input logic [4:0] address,
    output [INT_BITS + FRC_BITS-1:0] dout
);

   (* rom_style = "block" *) reg [INT_BITS + FRC_BITS-1:0] data;

generate
  if (N == 0) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0000;
         ...
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 1) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h000d;
         ...
            default: data <= 0;
          endcase
        end
      end
    endgenerate

...

generate
  if (N == 27) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0027;
         ...
            default: data <= 0;
          endcase
        end
      end
    endgenerate

    assign dout = data;
endmodule
