function H = Hamming(x,IntDNAMatrix)
% ������������x��y�ĺ�������,��������ֵd
% x:����������
% y:����������
% x��y������ͬ����
 H = pdist2(x,IntDNAMatrix,'hamming')*size(IntDNAMatrix,2);
end
    


