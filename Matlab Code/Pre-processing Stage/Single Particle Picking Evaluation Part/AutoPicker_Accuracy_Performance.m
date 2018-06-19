%% CryoEm Project ...
% AutoPicker: Fully Automated Single Particle Picking in Cryo-EM Images...
%
%==========================================================================
% Adil Al-Azzawi & Jianlin Cheng
% University of Missouri-Columbia 
% aaadn5@mail.missouri.edu
%==========================================================================
%
%% Step(1): Initialization...
clear all ; close all; clc
%
disp('_______________________________________________________________________');
disp('                                                                       ');
disp('                CLUSTERING - EVALUATION - PERFORMANCE                  ');
disp('_______________________________________________________________________');
disp(' ');
%
%% Step(2): Prepeare Clutering Images using (K-means and Our Approach)...
% Do K-Means Clustering... 
opt=input('Have you already do the MRI Brain Tumor Clusters images using K-means [Y=1|N=0] : ');
if opt==0
       K_means_Clustering_Approach;
else
end
% Do Our Clustering... 
opt1=input('Have you already do the MRI Brain Tumor Clusters images using Our Approach [Y=1|N=0] : ');
if opt1==0
      Our_Clustering_Algorithm;
end
%
%% Step(3): Clustering Evaluation Performance ...
MRI_image='C:\Users\Adil Al-Azzawi\Desktop\Lamia Project\Step(3)_MRI_Tumor_Detection\Dataset\MRI Dataset\Upnormal\MRI';
%
Ground_Truth_images='C:\Users\Adil Al-Azzawi\Desktop\Lamia Project\Step(3)_MRI_Tumor_Detection\Dataset\MRI Dataset\Upnormal\GT';
Ours_Clustering_images='C:\Users\Adil Al-Azzawi\Desktop\Lamia Project\Step(3)_MRI_Tumor_Detection\Dataset\Our Clustering Images';
Kmenas_Clustering_images='C:\Users\Adil Al-Azzawi\Desktop\Lamia Project\Step(3)_MRI_Tumor_Detection\Dataset\K-means Clustering Images';
%
code_dir='C:\Users\Adil Al-Azzawi\Desktop\Lamia Project\Step(3)_MRI_Tumor_Detection\Matlab Code';

% Our K-Means Results
    OURS_VS_GT_ACCURACY=zeros(1,40);
    OURS_VS_GT_SEN=zeros(1,40);
    OURS_VS_GT_SPEC=zeros(1,40);
    OURS_VS_GT_PRE=zeros(1,40);
    OURS_VS_GT_TP=zeros(1,40);
    OURS_VS_GT_FP=zeros(1,40);
    OURS_VS_GT_MISS=zeros(1,40);  
    OURS_VS_GT_DIC=zeros(1,40); 
    OURS_VS_GT_F_SCORE=zeros(1,40);

% K means
    KMEANS_VS_GT_ACCURACY=zeros(1,40);
    KMEANS_VS_GT_SEN=zeros(1,40);
    KMEANS_VS_GT_SPEC=zeros(1,40);
    KMEANS_VS_GT_PRE=zeros(1,40);
    KMEANS_VS_GT_TP=zeros(1,40);
    KMEANS_VS_GT_FP=zeros(1,40);
    KMEANS_VS_GT_MISS=zeros(1,40);
    KMEANS_VS_GT_DIC=zeros(1,40); 
    KMEANS_VS_GT_F_SCORE=zeros(1,40); 
%%
cd(MRI_image);
D1 = dir('*.jpg');
% Change the direction to the image folder
   for i = 1:numel(D1)
        close all;
        % start to read cancer image
        clc; 
        disp('_________________________________________________________');        
        disp(' '); 
        disp('          P  E  R  F  O  R  M  A  N  C  E                ');
        disp(' C  L  U  S  T  E  R  I  N  G  -  A  C  C  U  R  A  C  Y ');
        disp('_________________________________________________________');
        disp(' ');
        fprintf('The MRI Cluster Image no : %d\n',i');  
 
        MRI_image = (imread(D1(i).name));
        MRI_image=imresize(MRI_image,0.5);
        
        % change direction to Ground Truth images
        cd (Ground_Truth_images);
        G_T_MRI_image = (imread(D1(i).name));
        [row,col,MAP]=size(G_T_MRI_image);
        if MAP==3
            G_T_MRI_image=rgb2gray( G_T_MRI_image );
        end
        % convert to binary
        G_T_MRI_image =imbinarize(G_T_MRI_image);
%         G_T_MRI_image=double(G_T_MRI_image);
        G_T_MRI_image=(imresize(G_T_MRI_image,0.5));
        G_T_MRI_image=double(G_T_MRI_image);
%         imshow(G_T_MRI_image);

        % change direction to Ground Truth images
        cd (Ours_Clustering_images);        
        Our_MRI_images = imread(D1(i).name);

        % convert image to double
        Our_MRI_images =imbinarize(Our_MRI_images);
        [R,C,~]=size(Our_MRI_images);
        Our_MRI_images=double(Our_MRI_images);

%         figure; imshow(Our_MRI_images);

        % change direction to K_Means images
        cd (Kmenas_Clustering_images);
        KM_MRI_images = (imread(D1(i).name));
        KM_MRI_images1=KM_MRI_images;
        %
        cd(code_dir);
        % First Results Ground Truth and Ours
        Our_MRI_images(Our_MRI_images>0)=1; 
        G_T_MRI_image(G_T_MRI_image>0)=1; 
        [confusion_matrix] = performance_results( Our_MRI_images,G_T_MRI_image );
        
        % DIC evaluation Results
        [ Dic1 ] = DICE_evaluation( confusion_matrix );
        
        % Evaluations Ours Vs. Ground Truth
        [acc1,spec1,sen1,pre1,FP_rate1,TP_rate1,Miss_class1,F_1_score1] = evaluation( confusion_matrix );
        
        OURS_VS_GT_ACCURACY(i)=acc1;
        OURS_VS_GT_TP(i)=TP_rate1;
        OURS_VS_GT_FP(i)=FP_rate1;
        OURS_VS_GT_PRE(i)=pre1;
        OURS_VS_GT_SEN(i)=sen1;
        OURS_VS_GT_SPEC(i)=spec1;
        OURS_VS_GT_MISS(i)=Miss_class1;
        OURS_VS_GT_DIC(i)=Dic1; 
        OURS_VS_GT_F_SCORE(i)=F_1_score1; 

        % Second Results K-menas Vs. Ground Truth
         KM_MRI_images(KM_MRI_images>0)=1; 
        [confusion_matrix] = performance_results(KM_MRI_images,G_T_MRI_image );
        % DIC evaluation Results
        [ Dic2 ] = DICE_evaluation( confusion_matrix );
        
        % Evaluations Ours Vs. Ground Truth
        [  acc2,spec2,sen2,pre2,FP_rate2,TP_rate2,Miss_class2,F_1_score2] = evaluation( confusion_matrix );
        
        KMEANS_VS_GT_ACCURACY(i)=acc2;
        KMEANS_VS_GT_TP(i)=TP_rate2;
        KMEANS_VS_GT_FP(i)=FP_rate2;
        KMEANS_VS_GT_PRE(i)=pre2;
        KMEANS_VS_GT_SEN(i)=sen2;
        KMEANS_VS_GT_SPEC(i)=spec2;
        KMEANS_VS_GT_MISS(i)=Miss_class2;
        KMEANS_VS_GT_DIC(i)=Dic2; 
        KMEANS_VS_GT_F_SCORE(i)=F_1_score2; 
      
        % change direction to the code
        cd (code_dir);
        figure;
        subplot(1,4,1); imshow(MRI_image);title('Original Skin Cancer Image');
        subplot(1,4,2); imshow(G_T_MRI_image);title('Ground Truth Lesion Image');
        subplot(1,4,3); imshow(Our_MRI_images);title('Our Clustred Lesion Image');
        subplot(1,4,4); imshow(KM_MRI_images1);title('K-Means Lesion Image');
%       pause;
    cd ('C:\Users\Adil Al-Azzawi\Desktop\Lamia Project\Step(3)_MRI_Tumor_Detection\Dataset\MRI Dataset\Upnormal\MRI');
   end
%%
    cd(code_dir);
    OURS_VS_GT_TP = sorting( OURS_VS_GT_TP,1 );
    OURS_VS_GT_FP = sorting( OURS_VS_GT_FP,1 );
    OURS_VS_GT_ACCURACY = sorting( OURS_VS_GT_ACCURACY,1); 
    OURS_VS_GT_PRE=sorting( OURS_VS_GT_PRE,1); 
    OURS_VS_GT_SEN=sorting( OURS_VS_GT_SEN,1); 
    OURS_VS_GT_SPEC=sorting( OURS_VS_GT_SPEC,1);
    OURS_VS_GT_DIC=sorting( OURS_VS_GT_DIC,1);

    KMEANS_VS_GT_TP = sorting( KMEANS_VS_GT_TP,2);
    KMEANS_VS_GT_FP = sorting( KMEANS_VS_GT_FP,2); 
    KMEANS_VS_GT_ACCURACY = sorting( KMEANS_VS_GT_ACCURACY,2); 
    KMEANS_VS_GT_PRE=sorting( KMEANS_VS_GT_PRE,1); 
    KMEANS_VS_GT_SEN=sorting( KMEANS_VS_GT_SEN,1); 
    KMEANS_VS_GT_SPEC=sorting( KMEANS_VS_GT_SPEC,1); 
    KMEANS_VS_GT_DIC=sorting( KMEANS_VS_GT_DIC,1);   

    %
    disp('=========================================================');
    disp('|                                                        |'); 
    disp('|          P E R F O R M A N C E - R E S U L T S         |');
    disp('|                                                        |'); 
    disp('=========================================================');
    disp(' ');
    disp('_________________________________________________________');
    disp(' '); 
    disp('   Our K-Means Approach Vs The Ground Truth Clustering   ');
    disp('_________________________________________________________');
    disp(' '); 
    fprintf('Average Accuracy  =\t %0.2f\n',(mean(OURS_VS_GT_ACCURACY))*100);
    fprintf('Average Sensitivity  =\t %0.2f\n',(mean(OURS_VS_GT_SEN))*100);
    fprintf('Average Specificity  =\t %0.2f\n',(mean(OURS_VS_GT_SPEC))*100);
    fprintf('Average Precision  =\t %0.2f\n',(mean(OURS_VS_GT_PRE))*100);
    fprintf('Average True  Positive Recognition Rate =\t %0.2f\n',(mean(OURS_VS_GT_TP))*100);
    fprintf('Average False Postive  Recognition Rate =\t %0.2f\n',(mean(OURS_VS_GT_FP))*100);
    fprintf('Average Misclassification Rate (Overall, how often is it wrong) =\t %0.2f\n',(mean(OURS_VS_GT_MISS))*100);
    fprintf('Average DIC Evaluation Results =\t %0.4f\n',mean(OURS_VS_GT_DIC));
    fprintf('\n');
    pause;  
    disp('_________________________________________________________');
    disp(' '); 
    disp('         K-Means Vs The Ground Truth Clustering          ');
    disp('_________________________________________________________');
    disp(' ');
    fprintf('Average Accuracy  =\t %0.2f\n',(mean(KMEANS_VS_GT_ACCURACY))*100);
    fprintf('Average Sensitivity  =\t %0.2f\n',(mean(KMEANS_VS_GT_SEN))*100);
    fprintf('Average Specificity  =\t %0.2f\n',(mean(KMEANS_VS_GT_SPEC))*100);
    fprintf('Average Precision  =\t %0.2f\n',(mean(KMEANS_VS_GT_PRE))*100);
    fprintf('Average True  Positive Recognition Rate =\t %0.2f\n',(mean(KMEANS_VS_GT_TP))*100);
    fprintf('Average False Postive  Recognition Rate =\t %0.2f\n',(mean(KMEANS_VS_GT_FP))*100);
    fprintf('Average Misclassification Rate (Overall, how often is it wrong) =\t %0.2f\n',(mean(KMEANS_VS_GT_MISS))*100); 
    fprintf('Average DIC Evaluation Results =\t %0.4f\n',mean(KMEANS_VS_GT_DIC));
    fprintf('\n');
    pause;
    
%% Plot the Results...
    figure;
    plot(OURS_VS_GT_MISS)
    title('The Misclassification Rate Curve (Our K-Means Vs. K-means)')
    hold on;
    plot(KMEANS_VS_GT_MISS)
    xlabel('Number Of MRI Images');
    ylabel('Misclassification Rate Rate');
    legend('Our Improving','K-Means')
    
    figure;
    plot(OURS_VS_GT_DIC)
    title('The DIC Evaluation Score (Our K-Means Vs. K-means)')
    hold on;
    plot(KMEANS_VS_GT_DIC)
    xlabel('Number Of MRI Images');
    ylabel('DIC Evaluation Score');
    legend('Our Improving','K-Means')

    figure;
    plot(OURS_VS_GT_PRE)
    title('ROC (Our K-Means Vs. K-means)')
    hold on;
    plot(KMEANS_VS_GT_PRE)
    xlabel('Recall(1-Specifity)'); 
    ylabel('Secsitivity');
    legend('Our Improving','K-Means')
    
    % Plot Our Results  
    figure;
    plot(OURS_VS_GT_ACCURACY)
    title('The ROC Curve (Our K-Means Vs. K-means)')
    hold on;
    plot(KMEANS_VS_GT_ACCURACY)
    xlabel('Number Of MRI Images');
    ylabel('Accuracy Rate');
    legend('Our Improving','K-Means')
   

    % Time Consuming...
    load Our_time;
    our_average_time=mean(our_time);
    fprintf('The Mean Average time that Our Approach needs to cluster all MRI images is : %f\n',our_average_time);
    %
    load Kmeans_time;
    kmeans_average_time=mean(k_means_time);
    fprintf('The Mean Average time that K-Means needs to cluster all MRI images is : %f\n',kmeans_average_time);
%%
