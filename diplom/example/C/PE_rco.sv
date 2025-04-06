`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Module Name: PE_rco
// Project Name: LST-1 model
//////////////////////////////////////////////////////////////////////////////////

  (* dont_touch = "yes" *) (* keep_hierarchy = "yes" *) 

module PE_rco#(
    parameter int INT_BITS = 5,  // integer part
    parameter int FRC_BITS = 7, // fractional part
    parameter int N = 0 // block number
)(
    input logic clk,    // clock
    input logic init,   // write initial value
    input logic en,     // execute MAC-operation
    input logic [4:0] address_row,
    input logic [4:0] address_col,
    input logic [9:0] address_o,
    input logic [1:0] rco_sel,      // selects row, column or w_out memory block
    input [INT_BITS + FRC_BITS-1:0] din,
    output [INT_BITS + FRC_BITS-1:0] dout
);

// Internal signals
logic [INT_BITS + FRC_BITS-1:0] rom_row_out, rom_col_out, rom_w_out, mux_rom;
logic [4:0] addr_row, addr_col;
assign addr_row = (init)? 0 : address_row+1;
assign addr_col = (init)? 0 : address_col+1;

ROM_row #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(N)) rom_row(.address(addr_row), .dout(rom_row_out), .clk(clk));
ROM_col #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(N)) rom_col(.address(addr_col), .dout(rom_col_out), .clk(clk));
ROM_w_out #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(N)) rom_w(.address(address_o), .dout(rom_w_out), .clk(clk));

always_comb begin : MUX_ROM
 unique case (rco_sel)
    2'b00: mux_rom = rom_row_out;
    2'b01: mux_rom = rom_col_out;
    2'b10: mux_rom = rom_w_out;
endcase
end 

mac_core #(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS))
          mac_inst(.din(din), .mem_in(mux_rom), .mac_out(dout), .clk(clk), .init(init), .en(en));

endmodule
