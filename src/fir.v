module fir 
#(  parameter pADDR_WIDTH = 12,
    parameter pDATA_WIDTH = 32,
    parameter Tape_Num    = 11
)
(
    //axi-lite write
    output wire                     awready,    // address write ready
    output wire                     wready,     // data    write ready
    input  wire                     awvalid,    // address write valid
    input  wire                     wvalid,     // data    write valid
    input  wire [(pADDR_WIDTH-1):0] awaddr,     // address write data  => Write the Address of Coefficient
    input  wire [(pDATA_WIDTH-1):0] wdata,      // data    write data  => Write the Coefficient
    //axi-lite read
    output wire                     arready,    // address read ready
    input  wire                     rready,     // data    read ready
    input  wire                     arvalid,    // address read valid
    output wire                     rvalid,     // data    read valid
    input  wire [(pADDR_WIDTH-1):0] araddr,     // address read data   => Send the Address of Coefficient to Read
    output reg  [(pDATA_WIDTH-1):0] rdata,      // data    read data   => Coefficient of that Address
    // AXI-stream
    // input 
    input   wire                     ss_tvalid, 
    input   wire [(pDATA_WIDTH-1):0] ss_tdata, 
    input   wire                     ss_tlast, 
    output  reg                      ss_tready, 
    // output
    input   wire                     sm_tready, 
    output  reg                      sm_tvalid, 
    output  reg  [(pDATA_WIDTH-1):0] sm_tdata, 
    output  reg                      sm_tlast, 
    
    // bram for tap RAM
    output  reg  [3:0]               tap_WE,
    output  reg                      tap_EN,
    output  reg  [(pDATA_WIDTH-1):0] tap_Di,
    output  reg  [(pADDR_WIDTH-1):0] tap_A,
    input   wire [(pDATA_WIDTH-1):0] tap_Do,

    // bram for data RAM
    output  reg  [3:0]               data_WE,
    output  reg                      data_EN,
    output  reg  [(pDATA_WIDTH-1):0] data_Di,
    output  reg  [(pADDR_WIDTH-1):0] data_A,
    input   wire [(pDATA_WIDTH-1):0] data_Do,

    input   wire                     axis_clk,
    input   wire                     axis_rst_n
);
// fir state 
localparam IDLE = 0,
           WAIT = 1,
           IN   = 2,
           COMP = 3;
        //    OUT  = 4;
reg [1:0] state, state_n;
reg signed [31:0] Yn, Xn, Hn, Yn_n;
reg [(pADDR_WIDTH-1):0] tap_wa, tap_ra, tap_a_r;
reg last;
reg [5:0] cnt11, rst_cnt;
wire      cnt_en = (state == COMP && cnt11 != 11);
//===========Block level signal(ap_hs)=============
reg ap_start, ap_idle, ap_done;
reg ap_start_n, ap_idle_n, ap_done_n;
reg [31:0] data_length; // parameterize ?

// ap_start
always@(*)begin
    if (~&tap_wa && wready && wvalid && wdata == 32'h0000_0001 && state == IDLE && rst_cnt == 10) // host program ap_start 
        ap_start_n = 1'b1; 
    else if(state == IN)
        ap_start_n = 1'b0;
    else 
        ap_start_n = ap_start;
end

//ap_idle 
always@(*)begin
    if (~&tap_wa && wready && wvalid && wdata == 32'h0000_0001 && state == IDLE && rst_cnt == 10) // host program ap_start
        ap_idle_n = 1'b0;
    // else if(sm_tlast)
    else if(cnt11 == 11 && state == COMP && last) // reset ap_idle when last data transmitted out 
        ap_idle_n = 1'b1;
    else 
        ap_idle_n = ap_idle;
end

//ap_done 
always@(*)begin
    if(cnt11 == 11 && state == COMP && last)
    // if(sm_tlast)
        ap_done_n = 1'b1;
    else if(state == IN)
        ap_done_n = 1'b0;
    else    
        ap_done_n = ap_done;
end

always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)begin
        ap_start    <= 1'b0;
        ap_idle     <= 1'b1;
        ap_done     <= 1'b0;
        data_length <= 32'd0;
    end
    else begin
        ap_start    <= ap_start_n;
        ap_idle     <= ap_idle_n;
        ap_done     <= ap_done_n;
        data_length <= (tap_wa == 12'h10 && wready && wvalid) ? wdata : data_length;
    end
end

//===========axilite-read=============
// localparam AXI_R_IDLE = 0,
//            AXI_R_ADDR = 1,
//            AXI_R_WAIT = 2, 
//            AXI_R_DATA = 3; 
localparam AXI_R_IDLE = 3'b000,
           AXI_R_ADDR = 3'b010,
           AXI_R_WAIT = 3'b100, 
           AXI_R_DATA = 3'b001 ; 
reg [2:0] axi_r_state, axi_r_state_n;

always@(*)begin
    case(axi_r_state)
        AXI_R_IDLE : axi_r_state_n = (arvalid) ? AXI_R_ADDR : AXI_R_IDLE;
        AXI_R_ADDR : axi_r_state_n = AXI_R_WAIT;
        AXI_R_WAIT : axi_r_state_n = AXI_R_DATA;
        AXI_R_DATA : axi_r_state_n = (rvalid)  ? AXI_R_IDLE : AXI_R_DATA;
        default : axi_r_state_n = AXI_R_IDLE;
    endcase
end
assign {arready, rvalid} = axi_r_state[1:0];
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)
        axi_r_state <= AXI_R_IDLE;
    else 
        axi_r_state <= axi_r_state_n;
end
always@(*)begin
    case(tap_ra)
        12'h00 : rdata = {ap_idle, ap_done, ap_start};
        12'h10 : rdata = data_length;
        default: rdata = tap_Do;
    endcase
end

//==========axilite-write===========
reg [1:0] axi_w_state, axi_w_state_n;
localparam AXI_W_IDLE = 2'b00,
           AXI_W_ADDR = 2'b10, 
           AXI_W_DATA = 2'b01;
always@(*)begin
    case(axi_w_state)
        AXI_W_IDLE : axi_w_state_n = (awvalid) ? AXI_W_ADDR : AXI_W_IDLE;
        AXI_W_ADDR : axi_w_state_n = AXI_W_DATA;
        AXI_W_DATA : axi_w_state_n = (wvalid)  ? AXI_W_IDLE : AXI_W_DATA;
        default : axi_w_state_n = AXI_W_IDLE;
    endcase
end
assign {awready, wready} = axi_w_state[1:0];
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)
        axi_w_state <= AXI_W_IDLE;
    
    else 
        axi_w_state <= axi_w_state_n;
    
end

always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)begin
        tap_wa <= 0;
        tap_ra <= 0;
    end
    else begin
        tap_wa <= (awvalid && awready) ? awaddr : tap_wa;
        tap_ra <= (arvalid && arready) ? araddr : tap_ra;
    end
end
//==============StreamIn============
always@(*)begin
    if(state == IN)
        ss_tready = 1;
    else 
        ss_tready = 0;
end
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)
        last <= 0;
    else begin
        if(state == IDLE) 
            last <= 0;
        else if (state == IN)
            last <= ss_tlast;
        else 
            last <= last;
    end
end
// ======================= Stream-Out ===========
always @(*) begin
    if (cnt11 == 11 && state == COMP) begin
        sm_tvalid = 1;
        sm_tdata = Yn;
        sm_tlast = last;
    end
    else begin
        sm_tvalid = 0;
        sm_tdata = 0;
        sm_tlast = 0;
    end
end

//==============tapRAM===============

//tap_A
always@(*)begin
    if (wready && wvalid && state != COMP)
        tap_a_r = tap_wa;
    else if(axi_r_state == AXI_R_WAIT && ap_idle) 
        tap_a_r = tap_ra;
    else                  
        tap_a_r = (cnt11 << 2) + 12'h20; //offset = 'h20

    tap_A = (12'h20 <= tap_a_r && tap_a_r <= 12'h48) ? tap_a_r - 12'h20 : 0;
end

//tap_WE && tap_Di
always@(*)begin
    if(wready && wvalid && tap_wa != 12'h00 && tap_wa != 12'h10)begin
        tap_WE = 4'hf;
        tap_Di = wdata;
    end
    else begin
        tap_WE = 0;
        tap_Di = 0;
    end
end

//tap_EN
always@(*)begin
    // tap_EN = ((wready && wvalid) || (rready && rvalid) || (S0 <= state_n && state_n <= S10));
    tap_EN = ((wready && wvalid) || (rready && rvalid) || (state_n == COMP));
end

//============dataRAM============
reg [(pADDR_WIDTH-1):0] data_wa, data_wa_n, data_ra, data_ra_n;

//data_wa
always@(*)begin
    if(state == IN)
        data_wa_n = (data_wa == 10) ? 0 : data_wa + 1;
    else
        data_wa_n = data_wa;
end
//data_ra
always@(*)begin
    if(state == IN)begin
        data_ra_n = (data_wa == 10) ?  0 : data_wa + 1;
    end
    else if(state == COMP)begin
        data_ra_n = (data_ra == 10) ?  0 : data_ra + 1;
    end
    else    
        data_ra_n = data_ra;
end

always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)begin
        data_wa <= 0;
        data_ra <= 1;
    end
    else begin
        data_wa <= data_wa_n;
        data_ra <= data_ra_n;
    end
end
always@(*)begin
    data_WE = 0;
    data_A  = 0;
    data_Di = 0;
    data_EN = 1;
    if(state == COMP)begin
        data_EN = 1;
        data_A  = (data_ra << 2);
    end
    else if(rst_cnt <= 10 && state == IDLE)begin
        data_EN = 1;
        data_A  = (rst_cnt << 2);
        data_WE = 4'hf;
        data_Di = 32'd0;
    end
    else if(state == IN)begin 
        data_EN = 1;
        data_A  = (data_wa << 2);
        data_WE = 4'hf;
        data_Di = ss_tdata;
    end 


    // else begin
    //     data_WE = 0;
    //     data_A  = 0;
    //     data_Di = 0;
    //     data_EN = 1;
    // end
end

always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)begin
        cnt11 <= 0;
        rst_cnt <= 0;
    end
    else begin
        cnt11 <= (cnt11 == 11) ? ((state == COMP && (state_n == IDLE||state_n == IN)) ? 0 : cnt11) : state == COMP ? cnt11 + 1: cnt11;
        rst_cnt <= (rst_cnt == 10)? (state == COMP && state_n == IDLE) ? 0 : rst_cnt : rst_cnt + 1;
    end
end
// fir fsm
always@(*)begin
    case(state)
        IDLE : state_n = (wready && wvalid && wdata == 32'd1 && rst_cnt == 10) ? WAIT : IDLE;
        WAIT : state_n = (ss_tvalid) ? IN : WAIT;    
        IN   : state_n = (ss_tready) ? COMP: IN;     // 1T
        COMP : begin
            // state_n = (cnt11 == 11) ? OUT : COMP; // 13T
            if(cnt11 == 11)begin
                if(last)
                    state_n = IDLE;
                else if(sm_tready && ss_tvalid)
                    state_n = IN;
                else   
                    state_n = COMP;
            end
            else begin
                state_n = COMP;
            end
        end
        // OUT  : begin
        //     if (last)
        //         state_n = IDLE;
        //     else if(sm_tready && ss_tvalid)
        //         state_n = IN;
        //     else 
        //         state_n = OUT;
        // end
        default : state_n = IDLE;
    endcase
end
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)
        state <= IDLE;
    else 
        state <= state_n;
end

//===========Xn, Yn, Hn, Yn_n -> reslove Latch
always@(*)begin
    Xn = data_Do;
    Hn = tap_Do;
    Yn= Yn_n;
end
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)
        Yn_n <= 0;
    else begin
        if(state == IN)
            Yn_n <= 0;
        else if(state == COMP)
            Yn_n <= Yn + Xn * Hn;
        else 
            Yn_n <= Yn;
    end
end

endmodule 
