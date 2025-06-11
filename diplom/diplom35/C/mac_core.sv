(* use_dsp = "yes" *) (* dont_touch = "yes" *) (* keep_hierarchy = "yes" *) 

module mac_core #( 
    parameter int INT_BITS = 5, // integer part 
    parameter int FRC_BITS = 7 // fractional part
)(
    input  logic clk,    // clock
    input  logic init,   // write initial value
    input  logic en,     // execute MAC-operation
    input  logic [INT_BITS + FRC_BITS-1:0] din,     
    input  logic [INT_BITS + FRC_BITS-1:0] mem_in,  
    output logic [INT_BITS + FRC_BITS-1:0] mac_out 
);
logic [INT_BITS + FRC_BITS-1:0] acc;   
logic [INT_BITS + FRC_BITS-1:0] m_o;   
logic [2*(INT_BITS + FRC_BITS)-1:0] mul_out;
assign mul_out = signed'(din) * signed'(mem_in);

always_ff @(posedge clk) begin
    if (init) begin
        acc <= mem_in;
    end else begin
        if (en) begin
            acc <= m_o + acc; // MAC: A = A + B * C  
        end    
    end
end

assign m_o = mul_out[INT_BITS + 2*FRC_BITS-1 : FRC_BITS];
assign mac_out = acc;
endmodule