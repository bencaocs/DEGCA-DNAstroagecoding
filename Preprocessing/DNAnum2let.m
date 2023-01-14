%DNA±äÎª0 1 2 3
function C = DNAnum2let(seq1)
B=char(seq1);
[m,l]=size(B);

for i=1:m
    C(i,:)=strrep(B(i,:),'T','0') ;
end
i=1;
for i=1:m
    C(i,:)=strrep(C(i,:),'C','1') ;
end
i=1;
for i=1:m
    C(i,:)=strrep(C(i,:),'G','2') ;
end
i=1;
for i=1:m
    C(i,:)=strrep(C(i,:),'A','3') ;
end
i=1;


    %C=fliplr(C);

end