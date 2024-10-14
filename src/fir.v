module fir 
#(  parameter pADDR_WIDTH = 12,
    parameter pDATA_WIDTH = 32,
    parameter Tape_Num    = 11
)
(
    //axi-lite write
    output reg                      awready,    // address write ready
    output reg                      wready,     // data    write ready
    input  wire                     awvalid,    // address write valid
    input  wire                     wvalid,     // data    write valid
    input  wire [(pADDR_WIDTH-1):0] awaddr,     // address write data  => Write the Address of Coefficient
    input  wire [(pDATA_WIDTH-1):0] wdata,      // data    write data  => Write the Coefficient
    //axi-lite read
    output reg                      arready,    // address read ready
    input  wire                     rready,     // data    read ready
    input  wire                     arvalid,    // address read valid
    output reg                      rvalid,     // data    read valid
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
localparam IDLE = 12,
           RST  = 13,
           WAIT = 14,
           SIN  = 15,
           STOR = 16,
           S0   = 0,
           S1   = 1,
           S2   = 2,
           S3   = 3,
           S4   = 4,
           S5   = 5,
           S6   = 6,
           S7   = 7,
           S8   = 8,
           S9   = 9,
           S10  = 10,
           OUT  = 11;
reg [4:0] state, state_n;
reg signed [31:0] Yn, Xn, Hn;
reg [(pADDR_WIDTH-1):0] tap_wa, tap_ra, tap_a_r;
reg last;

//===========Block level signal(ap_hs)=============
reg ap_start, ap_idle, ap_done;
reg ap_start_n, ap_idle_n, ap_done_n;
reg [31:0] data_length; // parameterize ?

// ap_start
always@(*)begin
    if (~&tap_wa && wready && wvalid && wdata == 32'h0000_0001) // host program ap_start 
        ap_start_n = wdata; 
    else if(state == SIN)
        ap_start_n = 1'b0;
    else 
        ap_start_n = ap_start;
end

//ap_idle 
always@(*)begin
    if (~&tap_wa && wready && wvalid && wdata == 32'h0000_0001) // host program ap_start
        ap_idle_n = 1'b0;
    else if(state == OUT && last) // reset ap_idle when last data transmitted out 
        ap_idle_n = 1'b1;
    else 
        ap_idle_n = ap_idle;
end

//ap_done 
always@(*)begin
    if(state == OUT && last)
        ap_done_n = 1'b1;
    else if(state == SIN)
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
localparam AXI_R_IDLE = 0,
           AXI_R_ADDR = 1,
           AXI_R_WAIT = 2, 
           AXI_R_DATA = 3;
reg [1:0] axi_r_state, axi_r_state_n;

always@(*)begin
    case(axi_r_state)
        AXI_R_IDLE : axi_r_state_n = (arvalid) ? AXI_R_ADDR : AXI_R_IDLE;
        AXI_R_ADDR : axi_r_state_n = AXI_R_WAIT;
        AXI_R_WAIT : axi_r_state_n = AXI_R_DATA;
        AXI_R_DATA : axi_r_state_n = (rvalid)  ? AXI_R_IDLE : AXI_R_DATA;
        default : axi_r_state_n = AXI_R_IDLE;
    endcase
end
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)begin
        axi_r_state <= AXI_R_IDLE;
        arready     <= 0;
        rvalid      <= 0;
    end
    else begin
        axi_r_state <= axi_r_state_n;
        arready     <= (axi_r_state_n == AXI_R_ADDR);
        rvalid      <= (axi_r_state_n == AXI_R_DATA);
    end
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
localparam AXI_W_IDLE = 0,
           AXI_W_ADDR = 1, 
           AXI_W_DATA = 2;
always@(*)begin
    case(axi_w_state)
        AXI_W_IDLE : axi_w_state_n = (awvalid) ? AXI_W_ADDR : AXI_W_IDLE;
        AXI_W_ADDR : axi_w_state_n = AXI_W_DATA;
        AXI_W_DATA : axi_w_state_n = (wvalid)  ? AXI_W_IDLE : AXI_W_DATA;
        default : axi_w_state_n = AXI_W_IDLE;
    endcase
end
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)begin
        axi_w_state <= AXI_W_IDLE;
        awready     <= 0;
        wready      <= 0;
    end
    else begin
        axi_w_state <= axi_w_state_n;
        awready     <= (axi_w_state_n == AXI_W_ADDR);
        wready      <= (axi_w_state_n == AXI_W_DATA);
    end
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
    if(state == SIN)
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
        else if (state == SIN)
            last <= ss_tlast;
        else 
            last <= last;
    end
end
// =============StreamOut===========
always @(*) begin
    if (state == OUT) begin
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
    if (wready && wvalid)
        tap_a_r = tap_wa;
    else if(axi_r_state == AXI_R_WAIT && ap_idle) 
        tap_a_r = tap_ra; //not complete
    else                  
        tap_a_r = (state_n << 2) + 12'h20; //offset = 'h20
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
    tap_EN = ((wready && wvalid) || (rready && rvalid) || (S0 <= state_n && state_n <= S10));
end

//============dataRAM============
// reg [3:0] data_WE_r;
// reg       data_EN_r;
reg [(pADDR_WIDTH-1):0] data_wa, data_wa_n, data_ra, data_ra_n;
// reg [(pDATA_WIDTH-1):0] data_Di_r;
reg [3:0] data_rst_cnt;

//data_wa
always@(*)begin
    if(state == SIN)begin
        data_wa_n = (data_wa == 10) ? 0 : data_wa + 1;
    end
    else
        data_wa_n = data_wa;
end
//data_ra
always@(*)begin
    if(state == SIN)begin
        data_ra_n = (data_wa == 10) ?  0 : data_wa + 1;
    end
    else if(S0 <= state && state <= S10)begin
        data_ra_n = (data_ra == 10) ?  0 : data_ra + 1;
    end
    else    
        data_ra_n = data_ra;
end

always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)begin
        data_wa <= 0;
        data_ra <= 1;
        data_rst_cnt <= 0;
    end
    else begin
        data_wa <= data_wa_n;
        data_ra <= data_ra_n;
        data_rst_cnt <= (state_n != state) ? 0 : data_rst_cnt + 1;
    end
end
always@(*)begin
    data_WE = 0;
    data_A  = 0;
    data_Di = 0;
    data_EN = 1;
    if(S0 <= state_n && state_n <= S10)begin
        data_EN = 1;
        data_A  = (data_ra_n << 2);
    end
    else if(state == RST)begin
        data_EN = 1;
        data_A  = (data_rst_cnt << 2);
        data_WE = 4'hf;
        data_Di = 32'd0;
    end
    else if(state == SIN)begin 
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

// fir fsm
always@(*)begin
    case(state)
        IDLE : state_n = (wready && wvalid && wdata == 1) ? RST : IDLE;
        RST  : state_n = (data_rst_cnt == 4'd10) ? WAIT : RST;
        WAIT : state_n = (ss_tvalid) ? SIN : WAIT;
        SIN  : state_n = (ss_tready) ? STOR: SIN;
        STOR : state_n = S0;
        S0   : state_n = S1;
        S1   : state_n = S2;
        S2   : state_n = S3;
        S3   : state_n = S4;
        S4   : state_n = S5;
        S5   : state_n = S6;
        S6   : state_n = S7;
        S7   : state_n = S8;
        S8   : state_n = S9;
        S9   : state_n = S10;
        S10  : state_n = OUT;
        OUT  : begin
            if (last)
                state_n = IDLE;
            else if(sm_tready)
                state_n = WAIT;
            else 
                state_n = OUT;
        end
    endcase
end
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)
        state <= IDLE;
    else 
        state <= state_n;
end

//===========Xn, Yn, Hn
always@(*)begin
    Xn = data_Do;
    Hn = tap_Do;
end
always@(posedge axis_clk or negedge axis_rst_n)begin
    if(!axis_rst_n)
        Yn <= 0;
    else begin
        if(state == SIN)
            Yn <= 0;
        else if(S0 <= state_n && state_n <= S10)
            Yn <= Yn + Xn * Hn;
        else 
            Yn <= Yn;
    end
end

endmodule 
