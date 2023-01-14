function HammingDistMatrix= DistHammingMatrix(IntDNAMatrix,R)
%�������������DNA���������м�ĺ�������,���ؾ������(�Գ�)
%IntDNAMatrix:��������,��������DNA���о���
%R:�ṹ��,�������
%DistHammingMatrix:��������(�Գ�),a_ij=HammingDist(DNA_i,DNA_j)
HammingDistMatrix = pdist2(IntDNAMatrix,IntDNAMatrix,'hamming')*size(IntDNAMatrix,2);
end