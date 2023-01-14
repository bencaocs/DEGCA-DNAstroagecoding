function [adj,indset_label] = preprocess_data (data,d,maxiter)
% is processed as a training set suitable for DEGCA program. mat of output has two,

% One is adj, reflecting edge relationship, and the other is indset_label, reflecting whether this node is a point in MIS
data1=DNAnum2let(data);
data1=data1-'0';
hmjuzhen=data1;
% Calculates sparse matrix, that is, statistical edge
HammingDist = DistHammingMatrix(hmjuzhen);
L=HammingDist>=d;
L(logical(eye(size(L))))=1;
adj = sparse(~L);
%
Length =size(hmjuzhen,2);
count =100;

DNASet=hmjuzhen(1:100,:);
DNAsethoubu=hmjuzhen(101:size(hmjuzhen,1),:);
newDNA = Aco(DNASet,Length,d);

iter = 1;
iter_max =maxiter;
while iter <= iter_max
    %fprintf('%d gen,size is %d\n',iter,size(newDNA,1));
    ran=randperm(size(DNAsethoubu,1),100);
    DNASet =  DNAsethoubu(ran,:);

    len = size(DNASet,1);
    for i=1:len
        DisH=Hamming(DNASet(i,:),newDNA);
        len1=find(DisH<d);
       if isempty(len1) 
           newDNA=[DNASet(i,:);newDNA];
            elseif length(len1)==1  
                newDNA(len1,:)=[];
                newDNA=[DNASet(i,:);newDNA];
            end
    end
    iter= iter+1;
end



set = newDNA;
BestSet = DNAcode2(newDNA);
[c, ia, is] = intersect(data,BestSet,'rows');
indset_label=zeros(1,size(data,1));
for i=1:size(ia,1)
        indset_label(ia(i))=1;
end
save(size(data,2)+"-"+d+"-"+size(ia,1), 'adj', 'indset_label','BestSet','ia');

end
