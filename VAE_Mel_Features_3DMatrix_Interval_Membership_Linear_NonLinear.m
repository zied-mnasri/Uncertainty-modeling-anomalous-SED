
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%% Evaluate Interval-Valued Fuzzy Membership functions %%%%%%%%%%%%%%%%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w0=input('Give weight of Class0: ')
framedur=1; %input('Give frame duration (sec): '); 
shift=0.5; %input('Give shift rate (between 0.01 and 0.5): ');
numEpochs=100;%input('Give number of epochs: ');
method= input('Give method (0: Baseline, 1: Uncertainty on VAE''s Error (using linear membership), 2: Uncertainty on membership (Using non-linear membership) : ');

%%%%%%%%%%%%%%%%% Load data   %%%%%%%%%%%%%%%%%%%%%%
load (strcat(['Data\Matrix_signal_frame_Mel-log-features_',num2str(framedur),'_sec_shift_',num2str(shift),'.mat']));    
load (strcat(['Data\Matrix_signal_frame_labels_',num2str(framedur),'_sec_shift_',num2str(shift),'.mat']));     

%%%%%%%%%%%%%%%%% Mel-log Features %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
M=size(mel_log_features);
Features=reshape(mel_log_features,M(1),M(2),M(3),M(4)*M(5),1);

%%%%%%%%%%%%%%%%% Load labels %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Labels=reshape(frame_event,M(4)*M(5),1);
Labels=Labels.';
Events(Labels<0)=0;
Events(Labels>=0)=1;
numFrames=size(Events,2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%% Evaluate VAE using (NotEvent Only) model
h=1;
for k = 1:numFrames
    display(['Processing file No ', num2str(k),' of ',num2str(numFrames)])
        Mel_features=Features(:,:,:,k);
        [M1,M2,M3]=size(Mel_features);
        Mmax1=2^(nextpow2(M1));
        Mmax2=2^(nextpow2(M2));
        Mel_features=[Mel_features;zeros(Mmax1-M1,M2,3)];
        M1=size(Mel_features,1);
        Mel_features=[Mel_features,zeros(Mmax1,Mmax2-M2,3)];
        M2=size(Mel_features,2);
        Mel_features=reshape(Mel_features,sqrt(Mmax1*Mmax2),sqrt(Mmax1*Mmax2),M3);
        specData(:,:,:,h)=Mel_features;
        h=h+1;
end
[M,N,P,Q]=size(specData);
numfiles=Q;
randIdx=randperm(numfiles);
Trainsize=floor(0.8*numfiles);
Testsize=floor(0.9*numfiles);

XTrain=specData(:,:,:,randIdx(1:Trainsize));
XTest=specData(:,:,:,randIdx(Trainsize+1:Testsize));
XEval=specData(:,:,:,randIdx(Testsize+1:end));

XTrain=dlarray(double(XTrain),'SSCB');
XTest=dlarray(double(XTest),'SSCB');
XEval=dlarray(double(XEval),'SSCB');

Events(Labels<0)=0;
Events(Labels>=0)=1;
YTrain=Events(randIdx(1:Trainsize));
YTest=Events(randIdx(Trainsize+1:Testsize));
YEval=Events(randIdx(Testsize+1:end));

%%%%%%%%%%%%%%%%% Load trained models   %%%%%%%%%%%%%%%%%%%%%%
load(['Models\VAE_3DMatrix_Model_NotEvent_',num2str(framedur),'_sec_','shift_',num2str(shift),'_epochs_',num2str(numEpochs),'.mat']);

%%%%%%%%%% VAE-class0-based reconstruction
display('Evaluate model for NotEvent Only!')

for idx=1:size(XEval,4)
        X0 = XEval(:,:,:,idx);
        [z0, ~, ~] = sampling(encoderNet0, X0);
        XPred0 = sigmoid(forward(decoderNet0, z0));      
        X0 = gather(extractdata(X0));
        XPred0 = gather(extractdata(XPred0));
        for h=1:size(X0,3)
            Pred_err_NotEvent(idx,h)=sqrt(mse(X0(:,:,h)-XPred0')); 
        end
end
Pred_err_NotEvent_norm=(Pred_err_NotEvent-min(Pred_err_NotEvent))./(max(Pred_err_NotEvent)-min(Pred_err_NotEvent));
eps0=sqrt(sum(Pred_err_NotEvent_norm.^2,2))/size(Pred_err_NotEvent_norm,2);

%%%%%%%%%%%%%%%%%%%%%%%  VAE_Thresholds  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
w1=1-w0;
p0=1-w0;
p1=1-w1;

%%%%%%%%%%%%%%%%%%% Evaluate Interval-Fuzzy membership functions %%%%%%%%%%
tau0=(1-w0)*max(eps0);
sigma=std(eps0);
med=mean(eps0);
gamma0=w0;
gamma1=1-w0;
%switch method
    
    if (method==0)  %%%% Evaluate Baseline method (VAE trained on NotEvent class only)%%%%%%%%
        YPred=(eps0>=tau0);
    elseif method==1 %Using linear membership function
            a0=(1-w0)*tau0;
            b0=2*(1-w0)*tau0;
            a1=(1-w1)*tau0;
            b1=2*(1-w1)*tau0;

            for k=1:length(YEval)
                if eps0(k)>a0
                    mu0L(k)=0;
                else
                    mu0L(k)=(a0 - eps0(k))/a0;
                end
            end

            for k=1:length(YEval)
                if eps0(k)<a0
                    mu0U(k)=1;
                elseif (eps0(k)>b0)
                    mu0U(k)=0;
                else
                    mu0U(k)=(b0- eps0(k))/a0;
                end
            end

            for k=1:length(YEval)
                if eps0(k)>a1
                    mu1L(k)=0;
                else
                    mu1L(k)=(a1 - eps0(k))/a1;
                end
            end

            for k=1:length(YEval)
                if eps0(k)<a1
                    mu1U(k)=1;
                elseif (eps0(k)>b1)
                    mu1U(k)=0;
                else
                    mu1U(k)=(b1- eps0(k))/a1;
                end
            end
        
        elseif method==2 %Non-linear membership function
            a=tau0;
            b=2*tau0;

            for k=1:length(YEval)
                if eps0(k)>a
                    mu0L(k)=0;
                else
                    mu0L(k)=(1 - eps0(k)/a)^w0;
                end
            end

            for k=1:length(YEval)
                if eps0(k)<a
                    mu0U(k)=1;
                elseif (eps0(k)>b)
                    mu0U(k)=0;
                else
                    mu0U(k)=(2- eps0(k)/a)^(1/w0);
                end
            end

            for k=1:length(YEval)
                if eps0(k)>a
                    mu1L(k)=0;
                else
                    mu1L(k)=(1- eps0(k)/a)^w1;
                end
            end

            for k=1:length(YEval)
                if eps0(k)<a
                    mu1U(k)=1;
                elseif (eps0(k)>b)
                    mu1U(k)=0;
                else
                    mu1U(k)=(2- eps0(k)/a)^(1/w1);
                end
            end

        end
        
%%%%%%%%%%%%%%%%%Interval comparison %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (method>0)
for k=1:length(YEval)

        P01(k)=(max(0,mu0U(k)-mu1L(k))-max(0,mu0L(k)-mu1U(k)))/(mu0U(k)-mu0L(k)+mu1U(k)-mu1L(k));
        if isnan(P01(k))
            P01(k)=0.5;
        end

        P10(k)=(max(0,mu1U(k)-mu0L(k))-max(0,mu1L(k)-mu0U(k)))/(mu0U(k)-mu0L(k)+mu1U(k)-mu1L(k));
        if isnan(P10(k))
            P10(k)=0.5;
        end

    end
    
% The predicted class corresponds to the most preferred/probable/largest interval of membership function components [muL,muU]
% If the interval [mu0L,mu0U] is less preferred than [mu1L,mu1U], then YPred=0; else YPred=1;
% P10=P([mu1L,mu1U]>[mu0L,mu0U]) and P01=P([mu0L,mu0U]>[mu1L,mu1U])
% YPred(k)=1 --> P10(k)>P01(k) --> Prob([mu1L,mu1U]>[mu0L,mu0U]) > Prob([mu0L,mu0U]>[mu1L,mu1U])
% YPred(k)=0 --> P10(k)<P01(k) --> Prob([mu1L,mu1U]>[mu0L,mu0U]) < Prob([mu0L,mu0U]>[mu1L,mu1U])
    %if method>0
   YPred=(P10>P01).';
%end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
close all
figure
subplot(411)
[eps0_sort,Idx_sort0]=sort(eps0);
mu0L_sort=mu0L(Idx_sort0);
plot(eps0_sort,mu0L_sort);
hold on
mu0U_sort=mu0U(Idx_sort0);
plot(eps0_sort,mu0U_sort)
legend('mu0L','mu0U')
xlabel('Error (eps0)')
title('Membership to Normal Events (mu0)')

subplot(412)
[eps1_sort,Idx_sort1]=sort(eps0);
mu1L_sort=mu1L(Idx_sort1);
plot(eps1_sort,mu1L_sort);
hold on
mu1U_sort=mu1U(Idx_sort1);
plot(eps1_sort,mu1U_sort)
legend('mu1L','mu1U')
xlabel('Error (eps1)')
title('Membership to Outlier Events (mu1)')

subplot(413)
P01_sort=P01(Idx_sort0);
plot(eps0_sort,P01_sort)
hold on
P10_sort=P10(Idx_sort0);
plot(eps0_sort,P10_sort)
legend('P01','P10')
xlabel('Error (eps0 or eps1)')
title('Sorted probability by error')

subplot(414)
[P01_sort,Idx_sortP01]=sort(P01);
eps0_sort2=eps0(Idx_sortP01);
plot(P01_sort,eps0_sort2)
hold on
[P10_sort,Idx_sortP10]=sort(P10);
%eps1_sort2=eps1(Idx_sortP01);
%plot(P10_sort,eps1_sort2)
%legend('eps0','eps1')
xlabel('Probability (P01 or P10)')
title('Sorted error by probability')
end
%%%%%%%%%%%%%%%%%%%%%%%% Confusion matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
numClasses=max(YEval)-min(YEval)+1;
for i=1:numClasses
    for j=1:numClasses
        CM(i,j)=0;
        for k=1:length(YPred)
           if ((YPred(k)==i-1)&(YEval(k)==j-1))
               CM(i,j)=CM(i,j)+1;
           end
        end
    end
end

for i=1:numClasses
    Prec(i)=CM(i,i)/sum(CM(i,:));
    Rec(i)=CM(i,i)/sum(CM(:,i));
    F1(i)=2*Prec(i)*Rec(i)/(Prec(i)+Rec(i));    
end

Acc=trace(CM)/length(YPred);
display('Perforamnce measures for VAE+Interval-FS (Acc, Prec1, Prec2, Rec1, Rec2, F1-1, F1-2)')
display(['Accuracy_FS = ',num2str(Acc)])
display(['Precision_FS = ',num2str(Prec)])
display(['Recall_FS = ',num2str(Rec)])
display(['F1score_FS = ',num2str(F1)])

%%%%%%%%%% Plot ROC and compute AUC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[tpr,fpr,thresholds] = roc(YEval,YPred')
figure
grid on
res=plotroc(YEval,YPred');
auc=trapz([fpr,1],[tpr,1]);% calculate the area under the curve 
display(['AUC = ',num2str(auc)])
