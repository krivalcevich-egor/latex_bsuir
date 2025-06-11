library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_ARITH.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity MUL_N is
	 generic (N:natural:=16; 
	          qA:natural:=8);  -- fractional part of A (to remove from product)
    port ( A : in  STD_LOGIC_VECTOR (N-1 downto 0);
           B : in  STD_LOGIC_VECTOR (N-1 downto 0);
           P : out  STD_LOGIC_VECTOR (2*N-1 downto 0)	-- full multiplier
--			  P : out STD_LOGIC_VECTOR(N-1 downto 0)			-- half of output bits is needed
			  );
end MUL_N;

architecture Behavioral of MUL_N is
type abij is array(0 to N-1) of std_logic_vector(N-1 downto 0);
component ADD1 
    Port ( A : in  STD_LOGIC;
           B : in  STD_LOGIC;
           Ci : in  STD_LOGIC;
           S : out  STD_LOGIC;
           Co : out  STD_LOGIC);
end component;
component HA 
    Port ( A : in  STD_LOGIC;
           B : in  STD_LOGIC;
           S : out  STD_LOGIC;
           Co : out  STD_LOGIC);
end component;
signal ab,s,c: abij;
signal c_out: std_logic_vector(N-1 downto 0);
signal p_out: std_logic_vector(2*N-1 downto 0);
begin

row: for i in 0 to N-1 generate
		begin
		col: for j in 0 to N-1 generate
			  begin
				n8: if ((j=N-1)and(i/=N-1))OR((i=N-1)and(j/=N-1)) generate
						begin
						ab(i)(j)<= not(A(j) and B(i));
					 end generate;
				a8: if ((j/=N-1)and(i/=N-1))OR((i=N-1)and(j=N-1)) generate
					 begin
						ab(i)(j)<=(A(j) and B(i));
					 end generate;
			end generate;
		end generate;
s(0)<=ab(0);
c(0)<=(others=>'0');
mul: for i in 0 to N-2 generate
	  begin
	  sm: for j in 0 to N-1 generate
	      begin
			one: if (j=0) generate
					ad: HA port map (A=>s(i)(j),B=>c(i)(j),S=>p_out(i),Co=>c(i+1)(j));
				  end generate;
			oth: if (j/=0) generate
					ad: ADD1 port map (A=>s(i)(j),B=>ab(i+1)(j-1),Ci=>c(i)(j),S=>s(i+1)(j-1),Co=>c(i+1)(j));
				  end generate;
			end generate;
	  s(i+1)(N-1)<=ab(i+1)(N-1);
	  end generate;
mul_end:
	 for i in 0 to N-1 generate
	 begin
	 fst: if (i=0) generate 
			 ad: HA port map (A=>s(N-1)(i),B=>c(N-1)(i),S=>p_out(N-1+i),Co=>c_out(i));  
			end generate;
	 one: if (i=1) generate
		    ad: ADD1 port map (A=>s(N-1)(i),B=>'1',Ci=>c(N-1)(i),S=>p_out(N-1+i),Co=>c_out(i));  
			end generate;
	 oth: if ((i/=0)and (i/=1)) generate
		    ad: ADD1 port map (A=>s(N-1)(i),B=>c_out(i-1),Ci=>c(N-1)(i),S=>p_out(N-1+i),Co=>c_out(i));  
			end generate;
	 end generate;
p_out(2*N-1) <= c_out(N-1); 
P<=p_out;						-- full integer multiplier
--P<=p_out(2*(N-1) downto N-1);-- half of product fractional multiplier
--P<=p_out(N-1 downto 0);-- half of product fractional multiplier
--P<=p_out(N+(N/2)-1 downto N/2);-- half of product

--P<=p_out(N+qA-1 downto qA);-- half of product
end Behavioral;

