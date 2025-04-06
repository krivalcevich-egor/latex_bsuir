`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Create Date: 29.12.2024 18:44:00
// Design Name: 
// Module Name: tanh_function
// Project Name: L2DST-one-block-NN
//////////////////////////////////////////////////////////////////////////////////


module tanh_function #(
    parameter int INT_SIZE = 3,  // integer part 
    parameter int FRC_SIZE = 8   // fractional part
) (
    input [(INT_SIZE+FRC_SIZE - 1):0] X,
    output logic [(INT_SIZE+FRC_SIZE - 1):0] Y
);
  localparam logic [FRC_SIZE-1:0] zero_string = '0;
  localparam logic [FRC_SIZE-1:0] ones_string = '1;

  localparam const_plus_two = {3'b010, zero_string};
  localparam const_minus_two = {3'b110, zero_string};
  localparam const_q_plus_one = {1'b0, ones_string};
  localparam const_q_minus_one = {1'b1, zero_string};

  logic [FRC_SIZE:0] add_sub_out;
  logic [INT_SIZE+FRC_SIZE-1:0] add_sub_out_ext;
//  logic [2*(INT_SIZE+FRC_SIZE)-1:0] mul_out;
  logic [(INT_SIZE+FRC_SIZE-1):0] x_quater;
  logic x_sign;


logic [INT_SIZE + FRC_SIZE-1:0] m_o;    // multiplier output
logic [2*(INT_SIZE + FRC_SIZE)-1:0] mul_out;
logic [2*(INT_SIZE + FRC_SIZE)-1:0] mul_out_1;
logic [2*(INT_SIZE + FRC_SIZE)-1:0] correct = 26'b000000_0000_0000_0000_0100_0000;

  assign x_sign = X[INT_SIZE+FRC_SIZE-1];

  always_comb begin : Tanh_approximation
//    if (x_sign == 0) begin  // (x>0) branch
      if ($signed(X) >= $signed(const_plus_two)) Y = 13'b00000_1000_0000;
      else if ($signed(X) <= $signed(const_minus_two)) Y = 13'b11111_1000_0000;
      else Y = m_o;
//    end
  end

  assign x_quater = {x_sign, x_sign, X[INT_SIZE+FRC_SIZE-1:2]};

  always_comb begin : Add_Sub_block
    if (x_sign == 0) add_sub_out = const_q_plus_one - x_quater[FRC_SIZE:0];
    else add_sub_out = const_q_plus_one + x_quater[FRC_SIZE:0];
  end

  assign add_sub_out_ext = {add_sub_out[FRC_SIZE], add_sub_out[FRC_SIZE], add_sub_out[FRC_SIZE], add_sub_out[FRC_SIZE], add_sub_out};

  MUL_N #(
      .N (INT_SIZE + FRC_SIZE),
      .qA(FRC_SIZE)
  ) mult_inst (
      .A(add_sub_out_ext),
      .B(X),
      .P(mul_out)
  );

assign mul_out_1 = mul_out; // + correct;
assign m_o = mul_out_1[INT_SIZE + 2*FRC_SIZE-1 : FRC_SIZE];

endmodule
