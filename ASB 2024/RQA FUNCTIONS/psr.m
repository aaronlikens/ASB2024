function out =psr(data,tau,dim)

% Transpose row vectors to column vectors
if size(data,2) > size(data,1)
    data = data';
end

% Get the column number of the data
DIM = size(data,2);

% Embed the data
for i = 1:dim
    out(1:length(data)-(dim-1)*tau,1+DIM*(i-1):DIM*i) = data(1+(i-1)*tau:length(data)-(dim-i)*tau,:);
end

end