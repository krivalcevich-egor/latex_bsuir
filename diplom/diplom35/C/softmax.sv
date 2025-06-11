(* dont_touch = "yes" *) (* keep_hierarchy = "yes" *)

module softmax#( 
    parameter int BITS = 24, // bit depth
    parameter int HEIGHT = 10 // size of array_weight
)( 
    input  logic [BITS - 1 : 0] result_layer [HEIGHT-1:0], 
    output logic [BITS - 1 : 0] predict_num 
);   
                 
logic [BITS - 1 : 0] max;      
            
always_comb begin
    max = result_layer[0];
    predict_num = 0;
    for (int i = 1; i < HEIGHT; i++) begin
        if (signed'(result_layer[i]) > signed'(max)) begin
            max  = result_layer[i];
            predict_num = i;
        end
    end
end
                 
endmodule