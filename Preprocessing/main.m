%Batch preprocessing for DNA graph data
d=3;
maxiter=400;
%if .fasta/q

data_fastq=importdata('codewords8.fasta');
for j=size(data_fastq,1):-1:1
if mod(j,2)==1
    data_fastq(j)=[];
end
end
data=cell2mat(data_fastq);

%if data size is .mat 
%data=importdata('8(3352).mat');

preprocess_data(data,d,maxiter);
