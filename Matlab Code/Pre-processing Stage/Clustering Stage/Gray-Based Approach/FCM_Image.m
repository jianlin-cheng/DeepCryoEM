function [ Unow, centers,m] = FCM_Image( X, M )
[row, col] = size(X);
m = 2;      % 2 fuzzification parameter
StopThresh = 0.0001;  % 0.001 stopping condition
MaxIter = 20;   % number of maximun iteration
diff    = inf;

% Initlize membership function
Upre = rand(row, col, M);
dep_sum = sum(Upre, 3);
dep_sum = repmat(dep_sum, [1,1, M]);
Upre = Upre./dep_sum;

% Initlize cluster centers
centers = zeros(M,1); 
for i=1:M
    centers(i,1) = sum(sum(Upre(:,:,i).*X))/sum(sum(Upre(:,:,i)));
end

for iter = 1:MaxIter    
% Updated the Equation 
    N       = size(X(:),1); %number of data points
    D       = pdist2(X(:), centers);
%     D       = pdist2(X(:), centers,'cityblock');

    for i = 1:N
        for j = 1:M
            if(D(i,j) ~= 0)
                U(i,j) = 1/sum((D(i,j)./D(i,:)).^(2/(m-1)));
            else
                U(i,j) = 1;
            end
        end
    end
    Unow=reshape(U,row, col, M);
%%
    %Update cluster centers
    centersPrev      = centers;
    Upre = Unow.^m;
    for i=1:M
        centers(i,1) = sum(sum(Upre(:,:,i).*X))/sum(sum(Upre(:,:,i)));
    end
    diff = norm(centersPrev - centers);
    if diff<StopThresh
        break;
    end
    fprintf('Iter = %d, Centers diff = %f\n', iter, diff);

end