`timescale 1ns / 1ps

//////////////////////////////////////////////////////////////////////////////////
// WEIGHT MEMORY (ROM)
//////////////////////////////////////////////////////////////////////////////////

  (* dont_touch = "yes" *) (* keep_hierarchy = "yes" *) 

module ROM_w_out #(
    parameter int INT_BITS = 5,  // integer part
    parameter int FRC_BITS = 7, // fractional part
    parameter int N = 0 // block number
)(
    input logic clk, // clock
    input logic [9:0] address,
    output [INT_BITS + FRC_BITS-1:0] dout
);

   (* rom_style = "block" *) reg [INT_BITS + FRC_BITS-1:0] data;

generate
  if (N == 0) begin
     always @(posedge clk) begin
       case(address)
         10'b0000000000: data <= 13'h1ff2;
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
         10'b0000000000: data <= 13'h0006;
         ...
            default: data <= 0;
          endcase
        end
      end
    endgenerate

...

generate
  if (N == 9) begin
     always @(posedge clk) begin
       case(address)
         10'b0000000000: data <= 13'h1ff2;
         ...
            default: data <= 0;
          endcase
        end
      end
    endgenerate

    assign dout = data;
endmodule
