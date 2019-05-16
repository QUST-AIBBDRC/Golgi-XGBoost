function ebgw1= ebgw1(seq1,L)
%len 
seq1 = strrep(seq1,'X','');  % omit 'X'
seq1 = strrep(seq1,'Z','');  % omit 'Z'
seq1 = strrep(seq1,'B','');  % omit 'B'
seq1 = strrep(seq1,'O','');  % omit 'O'
seq1 = strrep(seq1,'J','');  % omit 'J'
seq1 = strrep(seq1,'U','');  % omit 'U'

%class
c= strrep(seq1,'A','X');
c1= strrep(c,'C','Z');
c2= strrep(c1,'D','B');
c3= strrep(c2,'E','B');
c4= strrep(c3,'I','X');
c5= strrep(c4,'K','O');
c6= strrep(c5,'L','X');
c7=strrep(c6,'M','X');
c8=strrep(c7,'N','Z');
 c9=strrep(c8,'P','X');
 c10=strrep(c9,'Q','Z');
 c11=strrep(c10,'R','O');
 c12=strrep(c11,'S','Z');
 c13=strrep(c12,'T','Z');
 c14=strrep(c13,'V','X');
 c15=strrep(c14,'W','X');
c16= strrep(c15,'Y','Z');
c17= strrep(c16,'G','X');
c18= strrep(c17,'H','O');
c19= strrep(c18,'F','X');
len=length(c19);


 seq2=zeros(1,len);
for i=1:len
    if (c19(i)=='X' || c19(i)=='Z')
            seq2(i)=1;
        elseif(c19(i)=='B' || c19(i)=='O')
           seq2(i)=0; 
    end
end
% cc1=findstr(c19,'X');
% cc2=findstr(c19,'Z');
% cc3=findstr(c19,'B');
% cc4=findstr(c19,'O');

for k=1:L
    chang=fix(k*(len/L));
    seq3=seq2(1,1:chang);
    %calculate frequency
    ebgw1(1,k)=sum(seq3==1)/chang;
    clear chang seq3
end
end
