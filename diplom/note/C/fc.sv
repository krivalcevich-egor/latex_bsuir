
(* dont_touch = "yes" *) (* keep_hierarchy = "yes" *) 

module fc 
    import nn_param_pkg::*;
  (
    input  logic                         clk,
    input  logic                         rst_n,
    input  logic                         start_i,
    input  logic [         ADDR_OUT-1:0] x_i,
    input  logic [INT_BITS+FRC_BITS-1:0] din_i,
    output logic [         INT_BITS-1:0] num_o,
    output logic                         rdy_o
  );

logic [INT_BITS + FRC_BITS-1:0] memory         [PICT_SIZE-1:0][PICT_SIZE-1:0];
logic [INT_BITS + FRC_BITS-1:0] internal_rc    [PICT_SIZE-1:0];
logic [INT_BITS + FRC_BITS-1:0] internal_tanh  [PICT_SIZE-1:0];
logic [INT_BITS + FRC_BITS-1:0] result         [ ADDR_OUT-1:0];

logic sel_data;
logic init;
logic en;
logic mem_en;
logic tg_en;

(* dont_touch = "yes" *) logic [ ADDR_RC-1:0] fc_addr;
(* dont_touch = "yes" *) logic [ ADDR_RC-1:0] str_cnt;
(* dont_touch = "yes" *) logic [RCO_TYPE-1:0] rco_sel;      // selects row, column or w_out memory bloc
logic [INT_BITS + FRC_BITS-1:0] tang;

logic [INT_BITS + FRC_BITS-1:0] din;
logic [INT_BITS + FRC_BITS-1:0] dout;
logic we;
logic m_en;
(* dont_touch = "yes" *) logic [ADDR_OUT-1:0] addr;

  img_mem img_mem_inst(
    .clk(clk),
    .we(we),
    .en(m_en),
    .addr(addr),
    .din(din),
    .dout(dout)
  );
  
assign we = (sel_data)? 1 :
          | (tg_en & rco_sel  == 0)? 1 :
          | (tg_en & rco_sel  == 1)? 1 : 0
          ;
assign addr = ( sel_data)? x_i : 
            | (!sel_data & tg_en  & rco_sel == 0)? PICT_SIZE*str_cnt + fc_addr :
            | (!sel_data & !tg_en & rco_sel == 0)? PICT_SIZE*str_cnt + fc_addr-1 :
            | (!sel_data & tg_en  & rco_sel == 1)? PICT_SIZE*fc_addr + str_cnt :
            | (!sel_data & !tg_en & rco_sel == 1)? PICT_SIZE*(fc_addr-1) + str_cnt : 
            | (!sel_data & !tg_en & rco_sel == 2)? PICT_SIZE*str_cnt + fc_addr : 0
            ;

assign din  = (sel_data)? din_i : 
            | (tg_en & rco_sel== 0)? tang : 
            | (tg_en & rco_sel== 1)? tang : 0
            ;

(* dont_touch = "yes" *) logic [ADDR_OUT-1:0]   out_addr;
logic [INT_BITS + FRC_BITS-1:0] data2PE;
assign data2PE = dout;

genvar i, j;
generate
        for (i = 0; i < 10; i = i + 1) begin : PE_rco_inst
           PE_rco#(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(i)) PE_rco_0(
            .clk(clk),
            .init(init),
            .en(en),
            .address_row(fc_addr[ADDR_RC-1:0]),
            .address_col(fc_addr[ADDR_RC-1:0]),
            .address_o(out_addr),
            .rco_sel(rco_sel),
            .din(data2PE),
            .dout(internal_rc[i]));
        end
endgenerate

generate
    for (j = 0; j < 18; j = j + 1) begin : PE_rc_inst
       PE_rc#(.INT_BITS(INT_BITS), .FRC_BITS(FRC_BITS), .N(j+10)) PE_rc_0(
        .clk(clk), 
        .init(init), 
        .en(en), 
        .address_row(fc_addr[ADDR_RC-1:0]),
        .address_col(fc_addr[ADDR_RC-1:0]),
        .din(data2PE),
        .dout(internal_rc[j+10]), 
        .rc_sel(rco_sel[0]));
    end 
endgenerate
    
logic [INT_BITS + FRC_BITS-1:0] data_tg;

always_comb begin
  data_tg = internal_tanh[fc_addr];
end

tanh_function #(.INT_SIZE(INT_BITS), .FRC_SIZE(FRC_BITS)) tanh_inst(.X(data_tg), .Y(tang));

always_comb begin
  if(mem_en) begin
    internal_tanh = internal_rc; 
  end
end

assign result = internal_rc[ADDR_OUT-1:0];

softmax #(.BITS(BITS), .HEIGHT(10)) softmax_inst(.result_layer(result), .predict_num(num_o)); 

// -------------------------------------------------------
// CNT
// -------------------------------------------------------
logic out_en;
logic [ADDR_OUT-1:0] out_addr_next;

always_ff @( posedge clk ) begin
  if(~rst_n) begin
      out_addr = ADDR_OUT'(0);
  end else if(out_en) begin
      out_addr <= out_addr_next;
    end else begin
      out_addr <= ADDR_OUT'(0);
    end
end

assign out_addr_next = out_addr + ADDR_OUT'(1);

// -------------------------------------------------------
// CNT
// -------------------------------------------------------
logic fc_en;
logic [ADDR_RC-1:0] fc_addr_next;

always_ff @( posedge clk ) begin 
  if(~rst_n) begin
    fc_addr = ADDR_RC'(0);
  end else if(fc_en) begin
      fc_addr <= fc_addr_next;
    end else begin
      fc_addr <= ADDR_RC'(0);
    end
end

assign fc_addr_next = fc_addr + ADDR_RC'(1);

// -------------------------------------------------------
// CNT
// -------------------------------------------------------
logic str_en;
logic rst_str_cnt;
logic [ADDR_RC-1:0] str_cnt_next;

always_ff @( posedge clk ) begin 
  if(~rst_n) begin
    str_cnt = ADDR_RC'(0);
  end else if(str_en) begin
      str_cnt <= str_cnt_next;
    end else begin
      if(~rst_str_cnt) begin
        str_cnt <= str_cnt;
      end else begin
        str_cnt <= ADDR_RC'(0);
      end
    end
end

assign str_cnt_next = str_cnt + ADDR_RC'(1);

// -------------------------------------------------------
// FSM
// -------------------------------------------------------

typedef enum logic [NN_FSM_BUS_WIDTH-1:0] {
  NN_INIT             = NN_FSM_BUS_WIDTH'('b00000),
  NN_LOAD_IMG         = NN_FSM_BUS_WIDTH'('b10000),
  NN_LD_MEM           = NN_FSM_BUS_WIDTH'('b10001),
  NN_INIT_CALC        = NN_FSM_BUS_WIDTH'('b10011),
  NN_STRING_CALC      = NN_FSM_BUS_WIDTH'('b10010),
  NN_STRING_RDY       = NN_FSM_BUS_WIDTH'('b10110),
  NN_TANG_STRING_CALC = NN_FSM_BUS_WIDTH'('b10111),
  NN_UPD_RCO_TYPE     = NN_FSM_BUS_WIDTH'('b10101),
  NN_INIT_OUT         = NN_FSM_BUS_WIDTH'('b11101),
  NN_OUT_CALC         = NN_FSM_BUS_WIDTH'('b11111),
  NN_OUT_RDY          = NN_FSM_BUS_WIDTH'('b11110)
} nn_fc_state_t;

nn_fc_state_t fsm_state_ff, fsm_state_next;

always_ff @(posedge clk) begin
  if(~rst_n) begin
    fsm_state_ff <= NN_INIT;
  end else begin
    fsm_state_ff <= fsm_state_next;
  end
end

logic fsm_init_tr;
logic fsm_load_img_tr;
logic fsm_ld_mem_tr;
logic fsm_init_calc_tr;
logic fsm_str_calc_tr;
logic fsm_str_rdy_tr;
logic fsm_tang_str_calc_tr;
logic fsm_upd_rco_type_tr;
logic fsm_init_out_tr;
logic fsm_out_calc_tr;
logic fsm_out_rdy_tr;

assign fsm_state_next  = (fsm_init_tr          ) ? NN_INIT             :
                         (fsm_load_img_tr      ) ? NN_LOAD_IMG         :
                         (fsm_ld_mem_tr        ) ? NN_LD_MEM           :
                         (fsm_init_calc_tr     ) ? NN_INIT_CALC        :
                         (fsm_str_calc_tr      ) ? NN_STRING_CALC      :
                         (fsm_str_rdy_tr       ) ? NN_STRING_RDY       :
                         (fsm_tang_str_calc_tr ) ? NN_TANG_STRING_CALC :
                         (fsm_upd_rco_type_tr  ) ? NN_UPD_RCO_TYPE     :
                         (fsm_init_out_tr      ) ? NN_INIT_OUT         :
                         (fsm_out_calc_tr      ) ? NN_OUT_CALC         :
                         (fsm_out_rdy_tr       ) ? NN_OUT_RDY          : fsm_state_ff;

assign rdy_o = (fsm_state_ff == NN_OUT_RDY          );

// -------------------------------------------------------
logic col_en;
logic rdy_en;
logic str_tg_en;   

logic rc_done;

assign fsm_init_tr          = (fsm_state_ff == NN_INIT             ) & ~start_i
                            | (fsm_state_ff == NN_OUT_RDY          );

assign fsm_load_img_tr      = (fsm_state_ff == NN_INIT             ) & start_i; 
assign fsm_ld_mem_tr        = (fsm_state_ff == NN_LOAD_IMG         ) & (x_i == WIDTH-1) 
                            | (fsm_state_ff == NN_TANG_STRING_CALC ) & (fc_addr == PICT_SIZE-1) 
                            | (fsm_state_ff == NN_UPD_RCO_TYPE     ) & (rco_sel != 2)
                            ;
assign fsm_init_calc_tr     = (fsm_state_ff == NN_LD_MEM           );
assign fsm_str_calc_tr      = (fsm_state_ff == NN_INIT_CALC        ) & (str_cnt != PICT_SIZE);
assign fsm_str_rdy_tr       = (fsm_state_ff == NN_STRING_CALC      ) & (fc_addr == PICT_SIZE);
assign fsm_tang_str_calc_tr = (fsm_state_ff == NN_STRING_RDY       ) ;
assign fsm_upd_rco_type_tr  = (fsm_state_ff == NN_INIT_CALC        ) & col_en;
assign fsm_init_out_tr      = (fsm_state_ff == NN_UPD_RCO_TYPE     ) & (rco_sel == 2);
assign fsm_out_calc_tr      = (fsm_state_ff == NN_INIT_OUT         );
assign fsm_out_rdy_tr       = (fsm_state_ff == NN_OUT_CALC         ) & rdy_en;

// -------------------------------------------------------

assign m_en = (fsm_state_ff == NN_LOAD_IMG         ) 
            | (fsm_state_ff == NN_STRING_CALC      ) 
            | (fsm_state_ff == NN_TANG_STRING_CALC )
            | (fsm_state_ff == NN_OUT_CALC   )
            | (fsm_state_ff == NN_INIT) & start_i;

// -------------------------------------------------------
assign rdy_en      = (out_addr == WIDTH+1);
assign rst_str_cnt = fsm_upd_rco_type_tr;
assign tg_en     = (fsm_state_ff == NN_TANG_STRING_CALC );
assign rc_done   = (fc_addr == PICT_SIZE);
assign mem_en    = (fsm_state_ff == NN_STRING_RDY       );
assign str_en    = (fsm_state_ff == NN_TANG_STRING_CALC ) & (fc_addr == PICT_SIZE-1)
                 | (fsm_state_ff == NN_OUT_CALC         ) & (fc_addr == PICT_SIZE-1);
assign fc_en     = (fsm_state_ff == NN_INIT_CALC        ) 
                 | (fsm_state_ff == NN_STRING_CALC      ) & (fc_addr != PICT_SIZE) 
                 | (fsm_state_ff == NN_TANG_STRING_CALC )
                 | (fsm_state_ff == NN_OUT_CALC         ) & (fc_addr != PICT_SIZE-1);

assign str_tg_en = (str_cnt == PICT_SIZE-1);

assign col_en    =  (fc_addr == 0)  & (str_cnt == PICT_SIZE) & (rco_sel != 1) | (fc_addr == PICT_SIZE)  & (str_cnt == PICT_SIZE) & (rco_sel == 1);

assign out_en    = (fsm_state_ff == NN_OUT_CALC    ) 
                 | (fsm_state_ff == NN_INIT_OUT    )
                 | (fsm_state_ff == NN_UPD_RCO_TYPE);

// -------------------------------------------------------
assign sel_data            = (fsm_state_ff == NN_LOAD_IMG) | (fsm_state_ff == NN_INIT) & start_i;

// --------------------------------------------------------
logic [RCO_TYPE-1:0] rco_sel_next;

always_ff @(posedge clk) begin
  if(~rst_n) begin
    rco_sel = RCO_TYPE'(0);
  end else if(start_i) begin
    rco_sel <= RCO_TYPE'(0);
  end else if(fsm_upd_rco_type_tr) begin
    rco_sel <= rco_sel_next;
  end else begin
    rco_sel <= rco_sel;
  end
end

assign rco_sel_next = rco_sel + RCO_TYPE'(1);

assign init = (fsm_state_ff == NN_LOAD_IMG         ) 
            | (fsm_state_ff == NN_LD_MEM           ) 
            | (fsm_state_ff == NN_STRING_RDY       ) 
            | (fsm_state_ff == NN_TANG_STRING_CALC ) 
            | (fsm_state_ff == NN_INIT_OUT         )
            | (fsm_state_ff == NN_UPD_RCO_TYPE     )
            ; 
assign en   = (fsm_state_ff == NN_STRING_CALC)
            | (fsm_state_ff == NN_OUT_CALC   )
            | (fsm_state_ff == NN_OUT_RDY    )
            ;
endmodule
