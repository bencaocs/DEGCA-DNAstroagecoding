function H = Hamming(x,IntDNAMatrix)
% 计算整数向量x与y的汉明距离,返回整数值d
% x:整数行向量
% y:整数行向量
% x与y必须相同长度
 H = pdist2(x,IntDNAMatrix,'hamming')*size(IntDNAMatrix,2);
end
    


