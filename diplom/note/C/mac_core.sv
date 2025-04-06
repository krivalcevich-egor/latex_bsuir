`timescale 1ns / 1ps

  (* use_dsp = "yes" *) (* dont_touch = "yes" *) (* keep_hierarchy = "yes" *) 

module mac_core #( 
    parameter int INT_BITS = 5, // integer part 
    parameter int FRC_BITS = 7 // fractional part
)(
    input logic clk,    // clock
    input logic init,   // write initial value
    input logic en,     // execute MAC-operation
    input  logic [INT_BITS + FRC_BITS-1:0] din,     // input data
    input  logic [INT_BITS + FRC_BITS-1:0] mem_in,  // weight of NN
    output logic [INT_BITS + FRC_BITS-1:0] mac_out  // accumulator output
);
logic [INT_BITS + FRC_BITS-1:0] acc;        // register-accumulator
logic [INT_BITS + FRC_BITS-1:0] m_o;    // multiplier output
logic [2*(INT_BITS + FRC_BITS)-1:0] mul_out;
logic [2*(INT_BITS + FRC_BITS)-1:0] mul_out_1;
logic [2*(INT_BITS + FRC_BITS)-1:0] correct = 24'b000000000000_000001000000;
//assign mul_out = signed'(din) * signed'(mem_in);
MUL_N #(.N(INT_BITS + FRC_BITS), .qA(FRC_BITS)) mul_inst(.A(din), .B(mem_in), .P(mul_out));

always_ff @(posedge clk) begin
    if (init) begin
        acc <= mem_in;
    end else begin
        if (en) begin
            acc <= m_o + acc; // MAC: A = A + B * C  
        end    
    end
end

assign mul_out_1 = mul_out;// + correct;
assign m_o = mul_out_1[INT_BITS + 2*FRC_BITS-1 : FRC_BITS];
assign mac_out = acc;
endmodule
