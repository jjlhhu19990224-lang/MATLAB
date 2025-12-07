function [fnn1M, fnn2M, fnnTotalM] = FalseNearestNeighbors3Criteria(xV,tauV,mV,escape,theiler)
% -----------------------------------------------------------
% 计算三种虚假邻近点判据 FNN：
%   1) 判据 1：R_{m+1}^2 / R_m^2 > escape^2
%   2) 判据 2：|x(i+mτ)-x(j+mτ)| / R_m > escape
%   3) 联合判据：以上两者之一成立
%
% 不依赖 MATS，不调用 kdtreeidx，完全基于 KDTreeSearcher。
% -----------------------------------------------------------

if nargin < 4 || isempty(escape), escape = 10; end
if nargin < 5 || isempty(theiler), theiler = 0; end

xV = xV(:);
n = length(xV);

% 保留原始信号用于判据 2 的幅值计算
xOrig = xV;

% 归一化并添加微小噪声避免重复点
xmin = min(xV); xmax = max(xV);
rangeX = xmax - xmin;
if rangeX > 0
    xV = (xV - xmin) / rangeX;
else
    % 常值序列时直接返回零矩阵
    fnn1M = zeros(length(tauV), length(mV));
    fnn2M = fnn1M;
    fnnTotalM = fnn1M;
    return;
end
xV = xV + 1e-10*randn(size(xV));

ntau = length(tauV);
nm   = length(mV);

fnn1M     = NaN(ntau,nm);
fnn2M     = NaN(ntau,nm);
fnnTotalM = NaN(ntau,nm);

% -------- 主循环 --------
for itau = 1:ntau
    tau = tauV(itau);

    for im = 1:nm
        m = mV(im);

        nvec = n - m*tau;
        if nvec - theiler < 2
            break;
        end

        % 组建 m 维嵌入（归一化版用于搜索，原始版用于幅值判据）
        X     = zeros(nvec,m);
        Xorig = zeros(nvec,m);
        for k = 1:m
            X(:,m-k+1)     = xV(1+(k-1)*tau : nvec+(k-1)*tau);
            Xorig(:,m-k+1) = xOrig(1+(k-1)*tau : nvec+(k-1)*tau);
        end

        % 构建 KD 树
        kd = KDTreeSearcher(X);

        idxV  = NaN(nvec,1);
        distV = NaN(nvec,1);

        % 搜索最近邻
        for i = 1:nvec
            % 找两个最近邻（自身 + 最近的邻居）
            [ind,dist] = knnsearch(kd, X(i,:), "K", 3);

            % 排除自身
            ind = ind(2:end);
            dist = dist(2:end);

            % Theiler window
            valid = find(abs(ind - i) > theiler);
            if isempty(valid), continue; end

            idxV(i)  = ind(valid(1));
            distV(i) = dist(valid(1));
        end

        valid = find(~isnan(idxV));
        nproper = length(valid);

        if nproper < 0.1*nvec
            continue;
        end

        % --- 计算判据 ---
        Rm2Norm  = distV(valid).^2;
        Rm1pNorm = (xV(valid+m*tau) - xV(idxV(valid)+m*tau)).^2;

        nnratio = 1 + Rm1pNorm ./ Rm2Norm;      % 判据 1 指标

        % 判据 2：追加维度的原始幅值增量相对于原始空间距离过大
        Rm2Raw = sum((Xorig(valid,:) - Xorig(idxV(valid),:)).^2, 2);
        rawDist = abs(xOrig(valid+m*tau) - xOrig(idxV(valid)+m*tau));
        fnn2    = mean(rawDist ./ sqrt(Rm2Raw + eps) > escape);

        fnn1     = mean(nnratio > escape^2);
        fnnTotal = mean((nnratio > escape^2) | (rawDist ./ sqrt(Rm2Raw + eps) > escape));
        
        fnn1M(itau,im)     = fnn1;
        fnn2M(itau,im)     = fnn2;
        fnnTotalM(itau,im) = fnnTotal;
    end
end
end
