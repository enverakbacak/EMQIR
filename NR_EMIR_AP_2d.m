
clear all;
close all;
clc;

load('Barcelona/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
data = max(data,0);
load('Barcelona/features/features_64.mat');
features = features_64;
load('Barcelona/hashCodes/targets.mat');
targets = targets;
load('Barcelona/hashCodes/filenames.mat');
filenames = filenames;
N = length(filenames);


queryIndex1 = 1;
queryIndex2 = 472;

q_1 = data(queryIndex1,:); 
q_2 = data(queryIndex2,:); 
%q3 = data(queryIndex3,:); 

q1new = repmat(q_1,N,1);
q2new = repmat(q_2,N,1);
%q3new = repmat(q3,N,1);

dist_1 = xor(data, q1new);
dist_2 = xor(data, q2new);
%dist_3 = xor(data, q2new);

hamming_dist1 = sum(dist_1,2);
hamming_dist2 = sum(dist_2,2);
%hamming_dist3 = sum(dist_3,2);

n_hamming_dist1 = mat2gray(hamming_dist1);
n_hamming_dist2 = mat2gray(hamming_dist2);
%n_hamming_dist3 = mat2gray(hamming_dist3);
 
 
X = zeros(2,N);
X(1,:) = hamming_dist1;
X(2,:) = hamming_dist2;
%X(3,:) = n_hamming_dist3;
X = (X)';
input = unique(X, 'rows');
hold off; plot(X(:,1),X(:,2),'k.'); hold on;

% queries location and Optimum point location
q1 = X(queryIndex1,:); 
q2 = X(queryIndex2,:);
line(q1,q2);
Cmn = (q1(:) + q2(:)).'/(2);

% https://www.mathworks.com/matlabcentral/answers/44943-properties-of-the-plot-marker-shapes
plot(q1 , q2, 'gd' , 'LineWidth', 20); 
plot(Cmn(1), Cmn(2), 'ro' , 'LineWidth', 10);


Dissim = EuDist2(Cmn, input, 'euclid');
Dissim = Dissim';
[DissimSorted, DissimIndex] = sort(Dissim,'ascend'); % short & get indexes(before shorting)


%%%%%%%%%%%%% Choose First P Shortest distances %%%%%%%%%%%%%%%%%%%%%%%%
P = 10;
DissimIndex_P = DissimIndex(1:P, :); 
Retrieved_PP_indexes  =    ismember(X, input(DissimIndex_P,:),'rows'); 
Retrieved_Items       =    find(Retrieved_PP_indexes); 

plot(X(Retrieved_Items,1),X(Retrieved_Items,2),'ys', 'LineWidth', 2); 
r=DissimSorted(P);
x0=Cmn(1);
y0=Cmn(2);
theta = linspace(0,2*pi,100);
h = plot(x0 + r*cos(theta),y0 + r*sin(theta),'g', 'MarkerSize',5);
%axis equal
%syms x y
%fimplicit((x-x0).^2 + (y-y0).^2 -r^2)
%axis equal
%}


%%%%%%%%%%%%%%%%%%%%  Re-Ranking, rearrange P items by features  %%%%%%%%%%% 
% Add queries to Feature Pareto space, for creating Cmn_f
Retrieved_Items(end+1,:) = queryIndex1;
Retrieved_Items(end+1,:) = queryIndex2;

rtr_idx2_features = features(Retrieved_Items(:,1), :);

f1 = features(queryIndex1,:); 
f2 = features(queryIndex2,:);

dist_f1 = pdist2(f1 , rtr_idx2_features , 'euclid' );
dist_f2 = pdist2(f2 , rtr_idx2_features , 'euclid' );

[M2,B] = size(rtr_idx2_features(:,1));
YY = zeros(2,M2);
YY(1,:) = dist_f1;
YY(2,:) = dist_f2;
YY = (YY)';
for i = 1:M2
    
    plot(YY(:, 1), YY(:,2)  ,'.')    
end


qf1 = YY(end-1,:);  
qf2 = YY(end,:);  
  
Cmn_f = (qf1(:) + qf2(:)).'/2;
plot(Cmn_f(1), Cmn_f(2), 's' , 'LineWidth', 2);

Dissim_f                = EuDist2(YY, Cmn_f, 'euclid');
Dissim_ff               = [Retrieved_Items, Dissim_f];
DissimSorted_f          = sortrows(Dissim_ff, 2);

Retrieved_Items_Ranked  = DissimSorted_f(:,1);

% Now remove last two rows (qf1&qf2) from Retrieved items
Retrieved_Items_Ranked(end) = [];
Retrieved_Items_Ranked(end) = [];


%%%%%%%%%%%%%%%%%%%%%%% Metrics %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

predicted_labels  = targets(Retrieved_Items_Ranked,:); % When doing Reranking
union_of_query_labels = or(targets(queryIndex1, :), targets(queryIndex2, : ));
    
diff = ismember( predicted_labels, union_of_query_labels  , 'rows'); 
if isempty( diff)
   diff = 0;
end
num_nz = nnz( diff(:,1) );
s = size(diff(:,1), 1);
    
for j=1:s;
        
    % Cummulative sum of the true-positive elements
    CUMM = cumsum(diff);          
    Precision_AT_K(j,1) = ( CUMM(j,1)  ) ./ j;              
    Recall_AT_K(j,1) = ( CUMM(j,1)  ) ./ (num_nz); % ???????????                    
       
end

avg_Precision = sum(Precision_AT_K(:,1) .* diff(:,1) ) / num_nz;
avg_Precision(isnan(avg_Precision))=0;
% avg_Precision_OLD = sum(Precision_AT_K(:,1) ) / s;
acc = num_nz / s;   % accuracy of the best cluster 

%{
%plot(Recall_AT_K, Precision_AT_K);

hold off;
x = linspace(0,s);
plot( Precision_AT_K )
ylabel('Precision@k' ,'FontSize', 12)
xlabel('Number of Rterieved Items' ,'FontSize', 12) 
hold on;
%}

