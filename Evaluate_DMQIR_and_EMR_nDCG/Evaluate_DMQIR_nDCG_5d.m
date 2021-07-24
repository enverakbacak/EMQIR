tic
close all;
clear all;
clc;



load('Barcelona/hashCodes/hashCodes_32.mat');
data = hashCodes_32;
data = max(data,0);
%load('Barcelona/features/features_64.mat');
%features = features_64;
load('Barcelona/hashCodes/targets.mat');
targets = targets;
load('Barcelona/hashCodes/filenames.mat');
queryIndex = xlsread('Barcelona/qGroups_5d.xls');
%}

%{
load('Lamda/hashCodes/hashCodes_16.mat');
data = hashCodes_16;
data = max(data,0);
%load('Lamda/features/features_64.mat');
%features = features_64;
load('Lamda/hashCodes/targets.mat');
targets = targets;
load('Lamda/hashCodes/filenames.mat');
queryIndex = xlsread('Lamda/qGroups_5d.xls');
%}


queryIndex = transpose( queryIndex ); 
queryIndex1 = queryIndex(1,:);        % First element of Query Pair
queryIndex2 = queryIndex(2,:);        % Second element of Query Pair
queryIndex3 = queryIndex(3,:);
queryIndex4 = queryIndex(4,:); 
queryIndex5 = queryIndex(5,:);
N = length(filenames);

     
maxFront = 3;    
    
    for l = 1:240                  % Number of Query Pairs
        
        q1 = data(queryIndex1,:);         % q1 & q2 are query pairs in the loop
        q2 = data(queryIndex2,:); 
        q3 = data(queryIndex3,:);  
        q4 = data(queryIndex4,:); 
        q5 = data(queryIndex5,:); 
                 
        q1_rep{l,:} = repmat(q1(l,:),N,1); % Make query matrix size to the same as data matrix size
        q2_rep{l,:} = repmat(q2(l,:),N,1);   
        q3_rep{l,:} = repmat(q3(l,:),N,1);
        q4_rep{l,:} = repmat(q4(l,:),N,1);   
        q5_rep{l,:} = repmat(q5(l,:),N,1);
        
        xor_data_q1new{l,:} = xor(data, q1_rep{l,:}); % xor of data and query matrices
        xor_data_q2new{l,:} = xor(data, q2_rep{l,:});  
        xor_data_q3new{l,:} = xor(data, q3_rep{l,:}); 
        xor_data_q4new{l,:} = xor(data, q4_rep{l,:});  
        xor_data_q5new{l,:} = xor(data, q5_rep{l,:});  
        
        hamming_dist1{l,:} = sum(xor_data_q1new{l,:},2); % sum up rows to get hamming distances
        hamming_dist2{l,:} = sum(xor_data_q2new{l,:},2);
        hamming_dist3{l,:} = sum(xor_data_q3new{l,:},2);
        hamming_dist4{l,:} = sum(xor_data_q4new{l,:},2);
        hamming_dist5{l,:} = sum(xor_data_q5new{l,:},2);
        
        norm_hamming_dist1{l,:} =  hamming_dist1{l,:} / max( hamming_dist1{l,:}(:) ); % Normalize hamming  distances between 0&1
        norm_hamming_dist2{l,:} =  hamming_dist2{l,:} / max( hamming_dist2{l,:}(:) );
        norm_hamming_dist3{l,:} =  hamming_dist3{l,:} / max( hamming_dist3{l,:}(:) );
        norm_hamming_dist4{l,:} =  hamming_dist4{l,:} / max( hamming_dist4{l,:}(:) );
        norm_hamming_dist5{l,:} =  hamming_dist5{l,:} / max( hamming_dist5{l,:}(:) );
        %dist1{l,:} = mat2gray(dist1{l,:}); % Normalize hamming  distances between 0&1
        %dist2{l,:} = mat2gray(dist2{l,:});     
        
        X = zeros(5,N);
        X(1,:) = norm_hamming_dist1{l,:};
        X(2,:) = norm_hamming_dist2{l,:};
        X(3,:) = norm_hamming_dist3{l,:};
        X(4,:) = norm_hamming_dist4{l,:};
        X(5,:) = norm_hamming_dist5{l,:};
    
        X = (X)';
        [K,~] = size(unique(X,'rows')); % The number of PPs = K

                
       
        
        [pf_idx] = pareto_fronts(X, maxFront);
        
        
        q1_label{l,:} = targets(queryIndex1(:,l), : ); % Label vector of Query 1
        q2_label{l,:} = targets(queryIndex2(:,l), : ); % Label vector of Query 2
        q3_label{l,:} = targets(queryIndex3(:,l), : ); % Label vector of Query 3
        q4_label{l,:} = targets(queryIndex4(:,l), : ); % Label vector of Query 4
        q5_label{l,:} = targets(queryIndex5(:,l), : ); % Label vector of Query 5
        
        b{l,:} = or(q1_label{l,:} , q2_label{l,:});    % beta in the equation 7 
        b{l,:} = or( b{l,:},q3_label{l,:});
        b{l,:} = or( b{l,:},q4_label{l,:});
        b{l,:} = or( b{l,:},q5_label{l,:});
        
        absolute_b{l,:} = nnz(b{l,:});                 % Number of non-zero elements in the beta, nnz is a Matlab Func.
        
         
        for j  =1:maxFront
            
            Labels{l,j} = targets(pf_idx{j,1}(:,6),:);
                                     
            [R(l,j) , C(l,j)] = size(Labels{l,j});        
            
              
           
            
            switch (mod(R(l,j) ,2) == 1)
                case 1
                     e_left(l,j) = (round(R(l,j) / 2) - 1) ;  
                     e_rigth(l,j)= (round(R(l,j) / 2)    ) ;
                case 0
                     e_left(l,j)  =  R(l,j) / 2 ;
                     e_rigth(l,j) =  R(l,j) / 2 ;
            end
            
            
           
            % ALL MQUR scores for each query pairs,  each front X Number of retrieved items               
            for e = 1:R(l,j)               
                MQUR_ALL{l,:}(j,e) =  nnz( and(Labels{l,j}(e,:) , b{l,:} ) ) /  absolute_b{l,:};
                                           
            end                   
            %MQUR score of the first retrieved items in each front
            MQUR_1(l,j) = MQUR_ALL{l,:}(j,1);
                        
            
             
            G{l,j} = [];
            rigth_G{l,j} = [];
            left_G{l,j}  = [];
            i_G{l,j} =[];
            i_rigth_G{l,j} = [];
            i_left_G{l,j}  = [];
            
            rigth_DG{l,j} = [];
            left_DG{l,j}  = [];
            i_DG{l,j} = [];
            i_rigth_DG{l,j} = [];
            i_left_DG{l,j}  = [];
            
            rigth_DCG{l,j} = [];
            left_DCG{l,j}  = [];
            i_rigth_DCG{l,j} = [];
            i_left_DCG{l,j}  = [];
            
            
             for s = 1:R(l,j)     
               G{l,j}(:,s) = MQUR_ALL{l,:}(j,s);
               i_G{l,j}(:,s) = 1;
             end
             
             switch (mod(R(l,j) ,2) == 1)
                case 1                    
                    rigth_G{l,j}   =   G{l,j}(: , (e_rigth(l,j)   )  : R(l,j));
                    i_rigth_G{l,j} = i_G{l,j}(: , (e_rigth(l,j)   )  : R(l,j));
                    
                    left_G{l,j}    =   G{l,j}(: , 1 : e_left(l,j) ); 
                    left_G{l,j}    =   fliplr(left_G{l,j});
                    i_left_G{l,j}   = i_G{l,j}(: , 1 : e_left(l,j) );
             
                    rigth_DG{l,j}(:,1)   =   rigth_G{l,j}(:,1);
                    i_rigth_DG{l,j}(:,1) = i_rigth_G{l,j}(:,1);
                    
             
             
                    left_DG{l,j}(:,1)   =   left_G{l,j}(:,1);
                    i_left_DG{l,j}(:,1) = i_left_G{l,j}(:,1);
             
                    for d = 2 : e_rigth(l,j)
                         rigth_DG{l,j}(:,d)   =   rigth_G{l,j}(:,d) / log2(d); %%%%%%%%%%%%%%%
                         i_rigth_DG{l,j}(:,d) = i_rigth_G{l,j}(:,d) / log2(d); 
                    end                    
                    for  f = 2 : e_left(l,j)   
                         left_DG{l,j}(:,f)   =   left_G{l,j}(:,f) / log2(f);
                         i_left_DG{l,j}(:,f) = i_left_G{l,j}(:,f) / log2(f);
                    end
                    
                    
                 case 0
                    rigth_G{l,j}   =   G{l,j}(: , (e_rigth(l,j) +1)  : R(l,j));
                    i_rigth_G{l,j} = i_G{l,j}(: , (e_rigth(l,j) +1)  : R(l,j));
                    
                    left_G{l,j}    =   G{l,j}(: , 1 : e_left(l,j) ); 
                    left_G{l,j}    =   fliplr(left_G{l,j});
                    i_left_G{l,j}   = i_G{l,j}(: , 1 : e_left(l,j) );
             
                    rigth_DG{l,j}(:,1)   =   rigth_G{l,j}(:,1);
                    i_rigth_DG{l,j}(:,1) = i_rigth_G{l,j}(:,1);
                    %rigth_DG{l,j}(:,e_rigth(l,j)) =  MQUR_ALL{l,:}(j,e_rigth(l,j));
             
             
                    left_DG{l,j}(:,1)   =   left_G{l,j}(:,1);
                    i_left_DG{l,j}(:,1) = i_left_G{l,j}(:,1);
             
             
                    for h = 2 : e_rigth(l,j)
                         rigth_DG{l,j}(:,h)   =   rigth_G{l,j}(:,h) / log2(h);
                         i_rigth_DG{l,j}(:,h) = i_rigth_G{l,j}(:,h) / log2(h); 
                    end
                    for m = 2 : e_left(l,j)
                         left_DG{l,j}(:,m)   =   left_G{l,j}(:,m) / log2(m);
                         i_left_DG{l,j}(:,m) = i_left_G{l,j}(:,m) / log2(m);
                    end
             end
             
                                    
             rigth_DCG{l,j} = cumsum(rigth_DG{l,j}); 
             i_rigth_DCG{l,j} = cumsum( i_rigth_DG{l,j});
             n_rigth_DCG{l,j} = rigth_DCG{l,j} ./ i_rigth_DCG{l,j};
                                  
           
             left_DCG{l,j} = cumsum(left_DG{l,j}); 
             i_left_DCG{l,j} = cumsum( i_left_DG{l,j});
             n_left_DCG{l,j} = left_DCG{l,j} ./ i_left_DCG{l,j};
             
             flip_n_left_DCG{l,j} = fliplr(n_left_DCG{l,j});
             n_DCG{l,j} = horzcat(flip_n_left_DCG{l,j}, n_rigth_DCG{l,j});
                 
            
                 
        end
    end
    
    toc
   

    for ll=1:l
    for jj=1:j
        [R_1(ll,jj) , C_1(ll,jj)] = size(n_rigth_DCG{ll,jj}); % Find size of each matrix in the n_rigth_DCG array
        [R_2(ll,jj) , C_2(ll,jj)] = size(n_left_DCG{ll,jj}); % Find size of each matrix in the n_left_DCG array
        
    end
end
max_1 = max(C_1(:)); % Find max size of n_rigth_DCG row
max_2 = max(C_2(:)); % Find max size of n_left_DCG row

for ll=1:l
    for jj=1:j
        n_rigth_DCG_zp{ll,jj}(1,:) =  [ n_rigth_DCG{ll,jj}(1,:) ,(zeros(1 ,max_1 - C_1(ll,jj) ))]; % Zero padding of all elements  in the n_rigth_DCG
        n_left_DCG_zp{ll,jj}(1,:)  =  [ n_left_DCG{ll,jj}(1,:)  ,(zeros(1 ,max_2 - C_2(ll,jj) ))]; % Zero padding of all elements  in the n_left_DCG
        
    end
end


n_rigth_DCG_zp_mean =[];
n_left_DCG_zp_mean =[];

for jj=1:j
    n_rigth_DCG_zp_sum  = 0;
    n_left_DCG_zp_sum  = 0;
    
    for ll=1:l
           
            n_rigth_DCG_zp_sum =  n_rigth_DCG_zp_sum  + n_rigth_DCG_zp{ll,jj}(1,:) ; % Sum of all rows for each righ front
            n_left_DCG_zp_sum =   n_left_DCG_zp_sum   + n_left_DCG_zp{ll,jj}(1,:) ;  % Sum of all rows for each left front
            
           
    end
    n_rigth_DCG_zp_mean(end+1,:) =  n_rigth_DCG_zp_sum/ll; % Mean value of each rigth front
    n_left_DCG_zp_mean(end+1,:) =  n_left_DCG_zp_sum/ll;   % Mean value of each left front
    
end

flip_n_left_DCG_zp_mean = fliplr(n_left_DCG_zp_mean);
n_DCG_mean_fronts = horzcat(flip_n_left_DCG_zp_mean, n_rigth_DCG_zp_mean);

n_DCG_mean = mean(n_DCG_mean_fronts ,1); % Mean value of first 10 fronts

plot(n_DCG_mean); 

ylabel('Mean Value of nDCG for DMQR')
xlabel('Number of Retrived Items on the Fronts') 