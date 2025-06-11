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

//   (* dont_touch = "yes" *) (* rom_style = "block" *) reg [INT_BITS + FRC_BITS-1:0] data;
   (* rom_style = "block" *) reg [INT_BITS + FRC_BITS-1:0] data;

generate
  if (N == 0) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0000;
         5'b00001: data <= 13'h0017;
         5'b00010: data <= 13'h001b;
         5'b00011: data <= 13'h1ff8;
         5'b00100: data <= 13'h0009;
         5'b00101: data <= 13'h0011;
         5'b00110: data <= 13'h1ff9;
         5'b00111: data <= 13'h001f;
         5'b01000: data <= 13'h001b;
         5'b01001: data <= 13'h0014;
         5'b01010: data <= 13'h001b;
         5'b01011: data <= 13'h0019;
         5'b01100: data <= 13'h0032;
         5'b01101: data <= 13'h0018;
         5'b01110: data <= 13'h0028;
         5'b01111: data <= 13'h003f;
         5'b10000: data <= 13'h001f;
         5'b10001: data <= 13'h1fd6;
         5'b10010: data <= 13'h1fb5;
         5'b10011: data <= 13'h1fd9;
         5'b10100: data <= 13'h1fc8;
         5'b10101: data <= 13'h1fd5;
         5'b10110: data <= 13'h1fec;
         5'b10111: data <= 13'h1fc9;
         5'b11000: data <= 13'h1fd7;
         5'b11001: data <= 13'h1fdf;
         5'b11010: data <= 13'h001b;
         5'b11011: data <= 13'h0012;
         5'b11100: data <= 13'h000d;
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
         5'b00001: data <= 13'h1ffb;
         5'b00010: data <= 13'h000a;
         5'b00011: data <= 13'h1fea;
         5'b00100: data <= 13'h1fe8;
         5'b00101: data <= 13'h1fdf;
         5'b00110: data <= 13'h0004;
         5'b00111: data <= 13'h0022;
         5'b01000: data <= 13'h001a;
         5'b01001: data <= 13'h002b;
         5'b01010: data <= 13'h001f;
         5'b01011: data <= 13'h001b;
         5'b01100: data <= 13'h0008;
         5'b01101: data <= 13'h002f;
         5'b01110: data <= 13'h000c;
         5'b01111: data <= 13'h1fdb;
         5'b10000: data <= 13'h1fd8;
         5'b10001: data <= 13'h1fe8;
         5'b10010: data <= 13'h0000;
         5'b10011: data <= 13'h000d;
         5'b10100: data <= 13'h000e;
         5'b10101: data <= 13'h0000;
         5'b10110: data <= 13'h1ff6;
         5'b10111: data <= 13'h1fda;
         5'b11000: data <= 13'h1fe4;
         5'b11001: data <= 13'h0011;
         5'b11010: data <= 13'h0004;
         5'b11011: data <= 13'h0017;
         5'b11100: data <= 13'h0007;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 2) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0009;
         5'b00001: data <= 13'h1ff2;
         5'b00010: data <= 13'h1ff4;
         5'b00011: data <= 13'h1ff0;
         5'b00100: data <= 13'h0016;
         5'b00101: data <= 13'h000d;
         5'b00110: data <= 13'h0027;
         5'b00111: data <= 13'h002f;
         5'b01000: data <= 13'h001c;
         5'b01001: data <= 13'h001b;
         5'b01010: data <= 13'h0008;
         5'b01011: data <= 13'h000d;
         5'b01100: data <= 13'h001b;
         5'b01101: data <= 13'h1fff;
         5'b01110: data <= 13'h1fcd;
         5'b01111: data <= 13'h1fb4;
         5'b10000: data <= 13'h1ffd;
         5'b10001: data <= 13'h002c;
         5'b10010: data <= 13'h000d;
         5'b10011: data <= 13'h1ff9;
         5'b10100: data <= 13'h1ff5;
         5'b10101: data <= 13'h0009;
         5'b10110: data <= 13'h1fff;
         5'b10111: data <= 13'h1ff6;
         5'b11000: data <= 13'h0006;
         5'b11001: data <= 13'h0000;
         5'b11010: data <= 13'h0009;
         5'b11011: data <= 13'h1fec;
         5'b11100: data <= 13'h1feb;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 3) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ff5;
         5'b00001: data <= 13'h0021;
         5'b00010: data <= 13'h000c;
         5'b00011: data <= 13'h001a;
         5'b00100: data <= 13'h0005;
         5'b00101: data <= 13'h1fec;
         5'b00110: data <= 13'h1fd6;
         5'b00111: data <= 13'h1ff3;
         5'b01000: data <= 13'h1fe6;
         5'b01001: data <= 13'h1fe8;
         5'b01010: data <= 13'h1fe7;
         5'b01011: data <= 13'h1fd2;
         5'b01100: data <= 13'h1ffa;
         5'b01101: data <= 13'h1fe3;
         5'b01110: data <= 13'h1fe9;
         5'b01111: data <= 13'h1fe1;
         5'b10000: data <= 13'h1fd0;
         5'b10001: data <= 13'h1fce;
         5'b10010: data <= 13'h1fd6;
         5'b10011: data <= 13'h1fdc;
         5'b10100: data <= 13'h1ff1;
         5'b10101: data <= 13'h0005;
         5'b10110: data <= 13'h1ffa;
         5'b10111: data <= 13'h1fec;
         5'b11000: data <= 13'h1fdb;
         5'b11001: data <= 13'h000b;
         5'b11010: data <= 13'h0029;
         5'b11011: data <= 13'h0010;
         5'b11100: data <= 13'h0013;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 4) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h002b;
         5'b00001: data <= 13'h1fd6;
         5'b00010: data <= 13'h1fd4;
         5'b00011: data <= 13'h1fcd;
         5'b00100: data <= 13'h1fe1;
         5'b00101: data <= 13'h0024;
         5'b00110: data <= 13'h003b;
         5'b00111: data <= 13'h0043;
         5'b01000: data <= 13'h0049;
         5'b01001: data <= 13'h0036;
         5'b01010: data <= 13'h0041;
         5'b01011: data <= 13'h004f;
         5'b01100: data <= 13'h0033;
         5'b01101: data <= 13'h0023;
         5'b01110: data <= 13'h1ffa;
         5'b01111: data <= 13'h1ffd;
         5'b10000: data <= 13'h1ffe;
         5'b10001: data <= 13'h0007;
         5'b10010: data <= 13'h0003;
         5'b10011: data <= 13'h0007;
         5'b10100: data <= 13'h000b;
         5'b10101: data <= 13'h0000;
         5'b10110: data <= 13'h1ffd;
         5'b10111: data <= 13'h1ffd;
         5'b11000: data <= 13'h1ff6;
         5'b11001: data <= 13'h1fe4;
         5'b11010: data <= 13'h1fd4;
         5'b11011: data <= 13'h1fcf;
         5'b11100: data <= 13'h1fbf;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 5) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ff3;
         5'b00001: data <= 13'h0026;
         5'b00010: data <= 13'h0028;
         5'b00011: data <= 13'h0025;
         5'b00100: data <= 13'h1ff9;
         5'b00101: data <= 13'h1ffb;
         5'b00110: data <= 13'h1fef;
         5'b00111: data <= 13'h1fec;
         5'b01000: data <= 13'h1ffb;
         5'b01001: data <= 13'h0004;
         5'b01010: data <= 13'h1ff6;
         5'b01011: data <= 13'h1ff9;
         5'b01100: data <= 13'h1ffe;
         5'b01101: data <= 13'h1ff0;
         5'b01110: data <= 13'h0000;
         5'b01111: data <= 13'h0012;
         5'b10000: data <= 13'h0043;
         5'b10001: data <= 13'h002e;
         5'b10010: data <= 13'h002b;
         5'b10011: data <= 13'h1fef;
         5'b10100: data <= 13'h1fdb;
         5'b10101: data <= 13'h1feb;
         5'b10110: data <= 13'h1fd6;
         5'b10111: data <= 13'h1fdc;
         5'b11000: data <= 13'h1ff4;
         5'b11001: data <= 13'h1fff;
         5'b11010: data <= 13'h0009;
         5'b11011: data <= 13'h0023;
         5'b11100: data <= 13'h0019;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 6) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ffd;
         5'b00001: data <= 13'h0006;
         5'b00010: data <= 13'h1ffd;
         5'b00011: data <= 13'h1ffe;
         5'b00100: data <= 13'h0010;
         5'b00101: data <= 13'h1ff9;
         5'b00110: data <= 13'h1ff9;
         5'b00111: data <= 13'h1ffc;
         5'b01000: data <= 13'h1ff9;
         5'b01001: data <= 13'h0018;
         5'b01010: data <= 13'h001e;
         5'b01011: data <= 13'h1ffe;
         5'b01100: data <= 13'h1fef;
         5'b01101: data <= 13'h1fd1;
         5'b01110: data <= 13'h1fae;
         5'b01111: data <= 13'h1fdb;
         5'b10000: data <= 13'h000f;
         5'b10001: data <= 13'h0048;
         5'b10010: data <= 13'h0036;
         5'b10011: data <= 13'h0020;
         5'b10100: data <= 13'h001e;
         5'b10101: data <= 13'h001f;
         5'b10110: data <= 13'h0014;
         5'b10111: data <= 13'h1ffc;
         5'b11000: data <= 13'h1fec;
         5'b11001: data <= 13'h1fe6;
         5'b11010: data <= 13'h000b;
         5'b11011: data <= 13'h0001;
         5'b11100: data <= 13'h001c;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 7) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ffc;
         5'b00001: data <= 13'h0006;
         5'b00010: data <= 13'h1ff7;
         5'b00011: data <= 13'h1ff4;
         5'b00100: data <= 13'h0001;
         5'b00101: data <= 13'h0006;
         5'b00110: data <= 13'h0006;
         5'b00111: data <= 13'h001a;
         5'b01000: data <= 13'h0006;
         5'b01001: data <= 13'h0015;
         5'b01010: data <= 13'h1fff;
         5'b01011: data <= 13'h1fef;
         5'b01100: data <= 13'h1fc1;
         5'b01101: data <= 13'h1fd0;
         5'b01110: data <= 13'h1fe9;
         5'b01111: data <= 13'h1fe8;
         5'b10000: data <= 13'h1ff9;
         5'b10001: data <= 13'h1fe1;
         5'b10010: data <= 13'h1fde;
         5'b10011: data <= 13'h1fd8;
         5'b10100: data <= 13'h0024;
         5'b10101: data <= 13'h0041;
         5'b10110: data <= 13'h002a;
         5'b10111: data <= 13'h0030;
         5'b11000: data <= 13'h0031;
         5'b11001: data <= 13'h0018;
         5'b11010: data <= 13'h0002;
         5'b11011: data <= 13'h0001;
         5'b11100: data <= 13'h0003;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 8) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ff4;
         5'b00001: data <= 13'h1ffe;
         5'b00010: data <= 13'h0005;
         5'b00011: data <= 13'h0019;
         5'b00100: data <= 13'h0005;
         5'b00101: data <= 13'h1ffb;
         5'b00110: data <= 13'h0013;
         5'b00111: data <= 13'h000d;
         5'b01000: data <= 13'h1feb;
         5'b01001: data <= 13'h1ff6;
         5'b01010: data <= 13'h1feb;
         5'b01011: data <= 13'h1fe4;
         5'b01100: data <= 13'h1fdb;
         5'b01101: data <= 13'h1fd0;
         5'b01110: data <= 13'h1fae;
         5'b01111: data <= 13'h1fbd;
         5'b10000: data <= 13'h1fea;
         5'b10001: data <= 13'h1fcd;
         5'b10010: data <= 13'h1fd7;
         5'b10011: data <= 13'h1fdc;
         5'b10100: data <= 13'h000b;
         5'b10101: data <= 13'h000e;
         5'b10110: data <= 13'h1fea;
         5'b10111: data <= 13'h0005;
         5'b11000: data <= 13'h0003;
         5'b11001: data <= 13'h0006;
         5'b11010: data <= 13'h0019;
         5'b11011: data <= 13'h0026;
         5'b11100: data <= 13'h000c;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 9) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0005;
         5'b00001: data <= 13'h0012;
         5'b00010: data <= 13'h1ffa;
         5'b00011: data <= 13'h1ffe;
         5'b00100: data <= 13'h001c;
         5'b00101: data <= 13'h0030;
         5'b00110: data <= 13'h0011;
         5'b00111: data <= 13'h0015;
         5'b01000: data <= 13'h0002;
         5'b01001: data <= 13'h0012;
         5'b01010: data <= 13'h000f;
         5'b01011: data <= 13'h1ff8;
         5'b01100: data <= 13'h002b;
         5'b01101: data <= 13'h0034;
         5'b01110: data <= 13'h0039;
         5'b01111: data <= 13'h0044;
         5'b10000: data <= 13'h0037;
         5'b10001: data <= 13'h0031;
         5'b10010: data <= 13'h0008;
         5'b10011: data <= 13'h0018;
         5'b10100: data <= 13'h0014;
         5'b10101: data <= 13'h000e;
         5'b10110: data <= 13'h0010;
         5'b10111: data <= 13'h0012;
         5'b11000: data <= 13'h000f;
         5'b11001: data <= 13'h0019;
         5'b11010: data <= 13'h0014;
         5'b11011: data <= 13'h1ff5;
         5'b11100: data <= 13'h0007;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 10) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0013;
         5'b00001: data <= 13'h1fed;
         5'b00010: data <= 13'h1fca;
         5'b00011: data <= 13'h1fd9;
         5'b00100: data <= 13'h1ff5;
         5'b00101: data <= 13'h1ffc;
         5'b00110: data <= 13'h000a;
         5'b00111: data <= 13'h0028;
         5'b01000: data <= 13'h0029;
         5'b01001: data <= 13'h000f;
         5'b01010: data <= 13'h0014;
         5'b01011: data <= 13'h0014;
         5'b01100: data <= 13'h0023;
         5'b01101: data <= 13'h0028;
         5'b01110: data <= 13'h0018;
         5'b01111: data <= 13'h0028;
         5'b10000: data <= 13'h0025;
         5'b10001: data <= 13'h0033;
         5'b10010: data <= 13'h0023;
         5'b10011: data <= 13'h0014;
         5'b10100: data <= 13'h001c;
         5'b10101: data <= 13'h001b;
         5'b10110: data <= 13'h0010;
         5'b10111: data <= 13'h000f;
         5'b11000: data <= 13'h0004;
         5'b11001: data <= 13'h1fe0;
         5'b11010: data <= 13'h1ff0;
         5'b11011: data <= 13'h1fd6;
         5'b11100: data <= 13'h1fd1;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 11) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0011;
         5'b00001: data <= 13'h0013;
         5'b00010: data <= 13'h0002;
         5'b00011: data <= 13'h1ffa;
         5'b00100: data <= 13'h1fe1;
         5'b00101: data <= 13'h1fee;
         5'b00110: data <= 13'h0001;
         5'b00111: data <= 13'h1ff1;
         5'b01000: data <= 13'h0009;
         5'b01001: data <= 13'h000c;
         5'b01010: data <= 13'h1ffe;
         5'b01011: data <= 13'h0008;
         5'b01100: data <= 13'h1fec;
         5'b01101: data <= 13'h1fec;
         5'b01110: data <= 13'h1fff;
         5'b01111: data <= 13'h0000;
         5'b10000: data <= 13'h0055;
         5'b10001: data <= 13'h0065;
         5'b10010: data <= 13'h0029;
         5'b10011: data <= 13'h1fca;
         5'b10100: data <= 13'h1fdd;
         5'b10101: data <= 13'h1fd8;
         5'b10110: data <= 13'h1fda;
         5'b10111: data <= 13'h1fc7;
         5'b11000: data <= 13'h1fdf;
         5'b11001: data <= 13'h0012;
         5'b11010: data <= 13'h0004;
         5'b11011: data <= 13'h1ffb;
         5'b11100: data <= 13'h1ffd;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 12) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0017;
         5'b00001: data <= 13'h0008;
         5'b00010: data <= 13'h000a;
         5'b00011: data <= 13'h000a;
         5'b00100: data <= 13'h0008;
         5'b00101: data <= 13'h1fee;
         5'b00110: data <= 13'h1fd9;
         5'b00111: data <= 13'h1fe2;
         5'b01000: data <= 13'h1fd4;
         5'b01001: data <= 13'h1fd5;
         5'b01010: data <= 13'h1fdd;
         5'b01011: data <= 13'h000f;
         5'b01100: data <= 13'h0026;
         5'b01101: data <= 13'h0034;
         5'b01110: data <= 13'h0011;
         5'b01111: data <= 13'h0006;
         5'b10000: data <= 13'h0026;
         5'b10001: data <= 13'h0002;
         5'b10010: data <= 13'h0016;
         5'b10011: data <= 13'h0006;
         5'b10100: data <= 13'h000e;
         5'b10101: data <= 13'h000a;
         5'b10110: data <= 13'h000f;
         5'b10111: data <= 13'h001e;
         5'b11000: data <= 13'h001b;
         5'b11001: data <= 13'h0021;
         5'b11010: data <= 13'h1ffe;
         5'b11011: data <= 13'h1fef;
         5'b11100: data <= 13'h1fef;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 13) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1fd5;
         5'b00001: data <= 13'h002a;
         5'b00010: data <= 13'h0021;
         5'b00011: data <= 13'h001f;
         5'b00100: data <= 13'h000f;
         5'b00101: data <= 13'h1fe3;
         5'b00110: data <= 13'h1fd4;
         5'b00111: data <= 13'h1fcc;
         5'b01000: data <= 13'h1fdc;
         5'b01001: data <= 13'h1fd2;
         5'b01010: data <= 13'h1fd6;
         5'b01011: data <= 13'h1fdf;
         5'b01100: data <= 13'h1fdc;
         5'b01101: data <= 13'h1fd3;
         5'b01110: data <= 13'h1fd4;
         5'b01111: data <= 13'h1fdc;
         5'b10000: data <= 13'h1fd6;
         5'b10001: data <= 13'h1ff9;
         5'b10010: data <= 13'h0019;
         5'b10011: data <= 13'h000a;
         5'b10100: data <= 13'h0012;
         5'b10101: data <= 13'h0005;
         5'b10110: data <= 13'h0005;
         5'b10111: data <= 13'h001b;
         5'b11000: data <= 13'h0018;
         5'b11001: data <= 13'h0032;
         5'b11010: data <= 13'h002c;
         5'b11011: data <= 13'h0027;
         5'b11100: data <= 13'h001a;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 14) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1fec;
         5'b00001: data <= 13'h0027;
         5'b00010: data <= 13'h0015;
         5'b00011: data <= 13'h0026;
         5'b00100: data <= 13'h0012;
         5'b00101: data <= 13'h0007;
         5'b00110: data <= 13'h000c;
         5'b00111: data <= 13'h000a;
         5'b01000: data <= 13'h001e;
         5'b01001: data <= 13'h0029;
         5'b01010: data <= 13'h002b;
         5'b01011: data <= 13'h0030;
         5'b01100: data <= 13'h0025;
         5'b01101: data <= 13'h1fcc;
         5'b01110: data <= 13'h1fa3;
         5'b01111: data <= 13'h1fd4;
         5'b10000: data <= 13'h1fda;
         5'b10001: data <= 13'h1fd9;
         5'b10010: data <= 13'h1fc9;
         5'b10011: data <= 13'h1fd7;
         5'b10100: data <= 13'h1fd5;
         5'b10101: data <= 13'h1fe8;
         5'b10110: data <= 13'h1feb;
         5'b10111: data <= 13'h1fe3;
         5'b11000: data <= 13'h1fdb;
         5'b11001: data <= 13'h001c;
         5'b11010: data <= 13'h0004;
         5'b11011: data <= 13'h0025;
         5'b11100: data <= 13'h0017;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 15) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0008;
         5'b00001: data <= 13'h1ff6;
         5'b00010: data <= 13'h1fda;
         5'b00011: data <= 13'h1fe1;
         5'b00100: data <= 13'h1fdf;
         5'b00101: data <= 13'h1fcf;
         5'b00110: data <= 13'h1fe5;
         5'b00111: data <= 13'h1ff0;
         5'b01000: data <= 13'h0004;
         5'b01001: data <= 13'h000e;
         5'b01010: data <= 13'h002e;
         5'b01011: data <= 13'h0047;
         5'b01100: data <= 13'h0028;
         5'b01101: data <= 13'h0015;
         5'b01110: data <= 13'h0027;
         5'b01111: data <= 13'h000a;
         5'b10000: data <= 13'h0024;
         5'b10001: data <= 13'h001d;
         5'b10010: data <= 13'h0017;
         5'b10011: data <= 13'h002a;
         5'b10100: data <= 13'h0025;
         5'b10101: data <= 13'h000c;
         5'b10110: data <= 13'h0002;
         5'b10111: data <= 13'h1ffd;
         5'b11000: data <= 13'h1fef;
         5'b11001: data <= 13'h1fe3;
         5'b11010: data <= 13'h1fe5;
         5'b11011: data <= 13'h1ff1;
         5'b11100: data <= 13'h1fe7;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 16) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h000e;
         5'b00001: data <= 13'h000c;
         5'b00010: data <= 13'h1ff4;
         5'b00011: data <= 13'h1fea;
         5'b00100: data <= 13'h000e;
         5'b00101: data <= 13'h000b;
         5'b00110: data <= 13'h0010;
         5'b00111: data <= 13'h0016;
         5'b01000: data <= 13'h1fff;
         5'b01001: data <= 13'h1ffa;
         5'b01010: data <= 13'h1ff2;
         5'b01011: data <= 13'h0010;
         5'b01100: data <= 13'h1ffe;
         5'b01101: data <= 13'h000d;
         5'b01110: data <= 13'h1fd8;
         5'b01111: data <= 13'h1fd8;
         5'b10000: data <= 13'h1fbe;
         5'b10001: data <= 13'h1fd3;
         5'b10010: data <= 13'h1fd5;
         5'b10011: data <= 13'h1fc0;
         5'b10100: data <= 13'h1fc1;
         5'b10101: data <= 13'h1fc6;
         5'b10110: data <= 13'h1fe6;
         5'b10111: data <= 13'h1fe1;
         5'b11000: data <= 13'h0011;
         5'b11001: data <= 13'h1ff9;
         5'b11010: data <= 13'h1ffa;
         5'b11011: data <= 13'h1ffb;
         5'b11100: data <= 13'h1fff;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 17) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h000c;
         5'b00001: data <= 13'h1ff1;
         5'b00010: data <= 13'h1fe6;
         5'b00011: data <= 13'h1fe8;
         5'b00100: data <= 13'h0010;
         5'b00101: data <= 13'h000e;
         5'b00110: data <= 13'h0026;
         5'b00111: data <= 13'h1fe7;
         5'b01000: data <= 13'h000c;
         5'b01001: data <= 13'h0010;
         5'b01010: data <= 13'h000f;
         5'b01011: data <= 13'h0002;
         5'b01100: data <= 13'h0018;
         5'b01101: data <= 13'h1fee;
         5'b01110: data <= 13'h1ff0;
         5'b01111: data <= 13'h0016;
         5'b10000: data <= 13'h001c;
         5'b10001: data <= 13'h1ffd;
         5'b10010: data <= 13'h0015;
         5'b10011: data <= 13'h0010;
         5'b10100: data <= 13'h001d;
         5'b10101: data <= 13'h001f;
         5'b10110: data <= 13'h0038;
         5'b10111: data <= 13'h001d;
         5'b11000: data <= 13'h003c;
         5'b11001: data <= 13'h0033;
         5'b11010: data <= 13'h0005;
         5'b11011: data <= 13'h1ff4;
         5'b11100: data <= 13'h1fe2;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 18) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h002f;
         5'b00001: data <= 13'h1fe7;
         5'b00010: data <= 13'h1fea;
         5'b00011: data <= 13'h1fef;
         5'b00100: data <= 13'h1ff6;
         5'b00101: data <= 13'h1fe1;
         5'b00110: data <= 13'h1ffa;
         5'b00111: data <= 13'h1ffd;
         5'b01000: data <= 13'h0005;
         5'b01001: data <= 13'h0016;
         5'b01010: data <= 13'h1ffc;
         5'b01011: data <= 13'h0009;
         5'b01100: data <= 13'h1ff1;
         5'b01101: data <= 13'h1ff2;
         5'b01110: data <= 13'h1fe8;
         5'b01111: data <= 13'h1feb;
         5'b10000: data <= 13'h1ffb;
         5'b10001: data <= 13'h1fff;
         5'b10010: data <= 13'h0014;
         5'b10011: data <= 13'h0018;
         5'b10100: data <= 13'h002e;
         5'b10101: data <= 13'h0025;
         5'b10110: data <= 13'h002b;
         5'b10111: data <= 13'h002f;
         5'b11000: data <= 13'h002a;
         5'b11001: data <= 13'h0000;
         5'b11010: data <= 13'h0008;
         5'b11011: data <= 13'h1fdd;
         5'b11100: data <= 13'h1fce;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 19) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1fee;
         5'b00001: data <= 13'h000e;
         5'b00010: data <= 13'h0000;
         5'b00011: data <= 13'h000f;
         5'b00100: data <= 13'h0006;
         5'b00101: data <= 13'h1ff6;
         5'b00110: data <= 13'h0010;
         5'b00111: data <= 13'h1ff5;
         5'b01000: data <= 13'h1ff5;
         5'b01001: data <= 13'h1fea;
         5'b01010: data <= 13'h1ff0;
         5'b01011: data <= 13'h0003;
         5'b01100: data <= 13'h1ff2;
         5'b01101: data <= 13'h1fc6;
         5'b01110: data <= 13'h1fe6;
         5'b01111: data <= 13'h001c;
         5'b10000: data <= 13'h002c;
         5'b10001: data <= 13'h003c;
         5'b10010: data <= 13'h0037;
         5'b10011: data <= 13'h0031;
         5'b10100: data <= 13'h0029;
         5'b10101: data <= 13'h0034;
         5'b10110: data <= 13'h0007;
         5'b10111: data <= 13'h1fd7;
         5'b11000: data <= 13'h1fc6;
         5'b11001: data <= 13'h1fcf;
         5'b11010: data <= 13'h1fdf;
         5'b11011: data <= 13'h1feb;
         5'b11100: data <= 13'h0006;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 20) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0004;
         5'b00001: data <= 13'h0002;
         5'b00010: data <= 13'h0008;
         5'b00011: data <= 13'h1feb;
         5'b00100: data <= 13'h1fe4;
         5'b00101: data <= 13'h1ff8;
         5'b00110: data <= 13'h1ffb;
         5'b00111: data <= 13'h0019;
         5'b01000: data <= 13'h000d;
         5'b01001: data <= 13'h1ff9;
         5'b01010: data <= 13'h1fde;
         5'b01011: data <= 13'h1fe2;
         5'b01100: data <= 13'h1fef;
         5'b01101: data <= 13'h0007;
         5'b01110: data <= 13'h0040;
         5'b01111: data <= 13'h002d;
         5'b10000: data <= 13'h1ff9;
         5'b10001: data <= 13'h1fe8;
         5'b10010: data <= 13'h1ffe;
         5'b10011: data <= 13'h1ff6;
         5'b10100: data <= 13'h1fff;
         5'b10101: data <= 13'h1fef;
         5'b10110: data <= 13'h0001;
         5'b10111: data <= 13'h0024;
         5'b11000: data <= 13'h003b;
         5'b11001: data <= 13'h001e;
         5'b11010: data <= 13'h0024;
         5'b11011: data <= 13'h1ff1;
         5'b11100: data <= 13'h000a;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 21) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ffe;
         5'b00001: data <= 13'h1ffe;
         5'b00010: data <= 13'h0012;
         5'b00011: data <= 13'h1ff4;
         5'b00100: data <= 13'h1ff3;
         5'b00101: data <= 13'h0006;
         5'b00110: data <= 13'h001a;
         5'b00111: data <= 13'h001e;
         5'b01000: data <= 13'h000d;
         5'b01001: data <= 13'h1ffb;
         5'b01010: data <= 13'h1fc5;
         5'b01011: data <= 13'h1fdc;
         5'b01100: data <= 13'h1fc0;
         5'b01101: data <= 13'h1fc5;
         5'b01110: data <= 13'h0040;
         5'b01111: data <= 13'h003c;
         5'b10000: data <= 13'h0021;
         5'b10001: data <= 13'h0011;
         5'b10010: data <= 13'h0016;
         5'b10011: data <= 13'h0011;
         5'b10100: data <= 13'h001e;
         5'b10101: data <= 13'h001b;
         5'b10110: data <= 13'h0014;
         5'b10111: data <= 13'h1ffb;
         5'b11000: data <= 13'h1fee;
         5'b11001: data <= 13'h1ff9;
         5'b11010: data <= 13'h1fdd;
         5'b11011: data <= 13'h1ff7;
         5'b11100: data <= 13'h1ff7;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 22) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ff9;
         5'b00001: data <= 13'h1fff;
         5'b00010: data <= 13'h0007;
         5'b00011: data <= 13'h1ffe;
         5'b00100: data <= 13'h1ff4;
         5'b00101: data <= 13'h0011;
         5'b00110: data <= 13'h000e;
         5'b00111: data <= 13'h002f;
         5'b01000: data <= 13'h002b;
         5'b01001: data <= 13'h001d;
         5'b01010: data <= 13'h002c;
         5'b01011: data <= 13'h002b;
         5'b01100: data <= 13'h002f;
         5'b01101: data <= 13'h003a;
         5'b01110: data <= 13'h003d;
         5'b01111: data <= 13'h0021;
         5'b10000: data <= 13'h0004;
         5'b10001: data <= 13'h0001;
         5'b10010: data <= 13'h000c;
         5'b10011: data <= 13'h0005;
         5'b10100: data <= 13'h0003;
         5'b10101: data <= 13'h0008;
         5'b10110: data <= 13'h1fde;
         5'b10111: data <= 13'h000f;
         5'b11000: data <= 13'h1fee;
         5'b11001: data <= 13'h1fed;
         5'b11010: data <= 13'h1feb;
         5'b11011: data <= 13'h0003;
         5'b11100: data <= 13'h0003;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 23) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1fd5;
         5'b00001: data <= 13'h000d;
         5'b00010: data <= 13'h0019;
         5'b00011: data <= 13'h0014;
         5'b00100: data <= 13'h0017;
         5'b00101: data <= 13'h1fef;
         5'b00110: data <= 13'h1ffa;
         5'b00111: data <= 13'h1ff6;
         5'b01000: data <= 13'h1ffa;
         5'b01001: data <= 13'h000a;
         5'b01010: data <= 13'h1ff7;
         5'b01011: data <= 13'h0011;
         5'b01100: data <= 13'h1fef;
         5'b01101: data <= 13'h1ff3;
         5'b01110: data <= 13'h1fd7;
         5'b01111: data <= 13'h1fb3;
         5'b10000: data <= 13'h1fbd;
         5'b10001: data <= 13'h1fe6;
         5'b10010: data <= 13'h0005;
         5'b10011: data <= 13'h1ffb;
         5'b10100: data <= 13'h1ffd;
         5'b10101: data <= 13'h0008;
         5'b10110: data <= 13'h1ff8;
         5'b10111: data <= 13'h000c;
         5'b11000: data <= 13'h1ff8;
         5'b11001: data <= 13'h0011;
         5'b11010: data <= 13'h1ff7;
         5'b11011: data <= 13'h0012;
         5'b11100: data <= 13'h0008;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 24) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1fd9;
         5'b00001: data <= 13'h002a;
         5'b00010: data <= 13'h0010;
         5'b00011: data <= 13'h000f;
         5'b00100: data <= 13'h0023;
         5'b00101: data <= 13'h0019;
         5'b00110: data <= 13'h1ff7;
         5'b00111: data <= 13'h0003;
         5'b01000: data <= 13'h0005;
         5'b01001: data <= 13'h1ff7;
         5'b01010: data <= 13'h1fe1;
         5'b01011: data <= 13'h1fed;
         5'b01100: data <= 13'h1fe4;
         5'b01101: data <= 13'h1fe8;
         5'b01110: data <= 13'h1fd5;
         5'b01111: data <= 13'h1fe6;
         5'b10000: data <= 13'h1fde;
         5'b10001: data <= 13'h1fde;
         5'b10010: data <= 13'h1feb;
         5'b10011: data <= 13'h1fd3;
         5'b10100: data <= 13'h1fd5;
         5'b10101: data <= 13'h1fd8;
         5'b10110: data <= 13'h1fe5;
         5'b10111: data <= 13'h1ff2;
         5'b11000: data <= 13'h1fe3;
         5'b11001: data <= 13'h1fe6;
         5'b11010: data <= 13'h1ffe;
         5'b11011: data <= 13'h0025;
         5'b11100: data <= 13'h0031;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 25) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0002;
         5'b00001: data <= 13'h1ffc;
         5'b00010: data <= 13'h1ffb;
         5'b00011: data <= 13'h1ff4;
         5'b00100: data <= 13'h0016;
         5'b00101: data <= 13'h0020;
         5'b00110: data <= 13'h0019;
         5'b00111: data <= 13'h0003;
         5'b01000: data <= 13'h001c;
         5'b01001: data <= 13'h1ffd;
         5'b01010: data <= 13'h1ffd;
         5'b01011: data <= 13'h1fe0;
         5'b01100: data <= 13'h1fe1;
         5'b01101: data <= 13'h1fd0;
         5'b01110: data <= 13'h1fa4;
         5'b01111: data <= 13'h1fb9;
         5'b10000: data <= 13'h0015;
         5'b10001: data <= 13'h000c;
         5'b10010: data <= 13'h0023;
         5'b10011: data <= 13'h000e;
         5'b10100: data <= 13'h1ffb;
         5'b10101: data <= 13'h000f;
         5'b10110: data <= 13'h0007;
         5'b10111: data <= 13'h0017;
         5'b11000: data <= 13'h0007;
         5'b11001: data <= 13'h1ffd;
         5'b11010: data <= 13'h1ff0;
         5'b11011: data <= 13'h0001;
         5'b11100: data <= 13'h000a;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 26) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h1ffa;
         5'b00001: data <= 13'h0008;
         5'b00010: data <= 13'h0011;
         5'b00011: data <= 13'h1ffd;
         5'b00100: data <= 13'h0013;
         5'b00101: data <= 13'h1ff3;
         5'b00110: data <= 13'h1fe9;
         5'b00111: data <= 13'h1fca;
         5'b01000: data <= 13'h1fd2;
         5'b01001: data <= 13'h1fec;
         5'b01010: data <= 13'h1fe8;
         5'b01011: data <= 13'h1ffb;
         5'b01100: data <= 13'h0003;
         5'b01101: data <= 13'h002a;
         5'b01110: data <= 13'h0020;
         5'b01111: data <= 13'h1fe7;
         5'b10000: data <= 13'h1fe3;
         5'b10001: data <= 13'h0002;
         5'b10010: data <= 13'h1ffa;
         5'b10011: data <= 13'h1fe9;
         5'b10100: data <= 13'h1ffd;
         5'b10101: data <= 13'h1fe9;
         5'b10110: data <= 13'h1ff4;
         5'b10111: data <= 13'h0019;
         5'b11000: data <= 13'h0033;
         5'b11001: data <= 13'h002c;
         5'b11010: data <= 13'h0016;
         5'b11011: data <= 13'h0001;
         5'b11100: data <= 13'h1fff;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

generate
  if (N == 27) begin
     always @(posedge clk) begin
       case(address)
         5'b00000: data <= 13'h0027;
         5'b00001: data <= 13'h1fd6;
         5'b00010: data <= 13'h1fd5;
         5'b00011: data <= 13'h1fcb;
         5'b00100: data <= 13'h0004;
         5'b00101: data <= 13'h000b;
         5'b00110: data <= 13'h0031;
         5'b00111: data <= 13'h002a;
         5'b01000: data <= 13'h003e;
         5'b01001: data <= 13'h0033;
         5'b01010: data <= 13'h002b;
         5'b01011: data <= 13'h0036;
         5'b01100: data <= 13'h0037;
         5'b01101: data <= 13'h004e;
         5'b01110: data <= 13'h003b;
         5'b01111: data <= 13'h1ff2;
         5'b10000: data <= 13'h1fce;
         5'b10001: data <= 13'h1fd9;
         5'b10010: data <= 13'h1feb;
         5'b10011: data <= 13'h1fea;
         5'b10100: data <= 13'h1fe8;
         5'b10101: data <= 13'h1fea;
         5'b10110: data <= 13'h1fec;
         5'b10111: data <= 13'h1fea;
         5'b11000: data <= 13'h1fd8;
         5'b11001: data <= 13'h1fe6;
         5'b11010: data <= 13'h1fcd;
         5'b11011: data <= 13'h1ff0;
         5'b11100: data <= 13'h1fe4;
            default: data <= 0;
          endcase
        end
      end
    endgenerate

    assign dout = data;
endmodule
