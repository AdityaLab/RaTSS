function grad_desc(save_dir,save_file, maxIteration, beta)
deltaP=load(strcat(save_dir,save_file,'_B.txt'));
%deltaP=deltaP';
m= size(deltaP,1);
deltaP=deltaP(:,1);
%fprintf('size B:%d',m);
c1=max(abs(deltaP));
%disp(c1);
deltaP = deltaP/c1;
% for trial using min-max normalization for delpta P
%c1= min(deltaP);
%c2=max(deltaP)-min(deltaP);
%deltaP=deltaP/c1-c1/c2;

%disp(size(deltaP));
%disp(deltaP);
maxIteration= str2num(maxIteration);
beta= str2double(beta);
%l1=[0.1 0.12 0.15 0.18 0.19 0.2 0.22 0.25 0.3];
%l2=[0.1 0.12 0.15 0.18 0.19 0.2 0.22 .25 0.3];
l1=[0.01 0.03 0.05 0.1 0.12 0.15 0.18 0.19 0.2];
%l2=[0.01 0.03 0.05 0.1 0.12 0.15 0.18 0.19 0.2];
%l1=[0.1 0.001 0.002 0.003 0.004 0.005 0.007 0.008 0.009];
%l2=[1.8 2 2.1];
%l1=[0.94 0.98 1.2];
%l2=[0.94 0.98 1.2];
%l1=[0.03]; 
%l2=[0.05];
%l1=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.94];
l2=[0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.94];
%l1=[0.6];
%l2=[0.94];

alpha_n=zeros(m,1);
funVal_n = zeros(maxIteration,1);
l2_diff=100.0;
it_n=0;
cost_n=0;
l1_final=0.1;
l2_final=0.1;
for lambda1=l1
	for lambda2=l2
		alpha = rand(m,1);
		alpha(:,1)=0.0001;
		funVal = zeros(maxIteration,1);
		for iteration=1:maxIteration
			%gradient considering absolute
			%alpha_change= -deltaP.*sign(deltaP).*sign(alpha) + lambda1*sign(alpha)+lambda2*alpha;
			%gradient considering not absolute
			alpha_change= -deltaP + lambda1*sign(alpha)+lambda2*alpha;
			%line search
			t= backtrack_line_search(beta,alpha,alpha_change,deltaP,lambda1,lambda2);
			alpha = alpha-t*alpha_change;
			funVal(iteration) = -alpha'*deltaP + lambda1*norm(alpha,1)+lambda2*0.5*norm(alpha,2)^2;
            %disp(size(funVal(iteration)));
			%funVal(iteration) = -abs(alpha'*deltaP) + lambda1*norm(alpha,1)+lambda2*0.5*(norm(alpha,2)^2-1);
			%stopping criterion
			if iteration > 1
				if norm(alpha_change,2) < 1*10^-4
					break
				end
			end
			if iteration > 100
				if norm(alpha_change,2) < 1*10^-4 || funVal(iteration-1) == funVal(iteration)...
                || abs(funVal(iteration-1) - funVal(iteration)) < 1*10^-3
					break
				end
			end
		end
		diff = abs(1-norm(alpha,2)^2);
		fprintf('l2_norm=%.4f, lambda1=%.4f lambda2=%.4f it=%d\n',norm(alpha,2)^2,lambda1,lambda2,iteration);
		if diff >0 && diff<l2_diff
			l2_diff=diff;
			alpha_n=alpha;
			l1_final = lambda1;
			l2_final=lambda2;
			funval_n=funVal;
			it_n=iteration;
			cost_n=funVal(it_n);
		end
	end
end
%filename=strcat(save_dir,save_file,'_',mat2str(l1_final),'_',mat2str(l2_final));
filename=strcat(save_dir,save_file);
normID = fopen(strcat(filename,'_normDiff.txt'),'w');
fprintf(normID,'l2_norm=%.4f, diff=%.4f cost=%.4f it=%d lmbd1=%.2f lmbd2=%.2f\n',...
		norm(alpha_n,2)^2,l2_diff,cost_n,it_n,l1_final,l2_final);
fclose(normID);
%disp(alpha_n);
%{
fprintf('l2_norm=%.4f, diff=%.4f cost=%.4f it=%d lmbd1=%.2f lmbd2=%.2f\n',...
		norm(alpha_n,2)^2,l2_diff,cost_n,it_n,l1_final,l2_final);
[out,idx] = sort(alpha_n,'descend');
disp('alpha sorted:');
disp(idx);
fprintf('iteration:%d\n',it_n);
disp(funVal_n(it_n));
%}
%cost_file= strcat(filename,'_cost.txt');
%dlmwrite(cost_file,funVal_n,',');
alpha_file = strcat(filename,'_alpha.txt');
dlmwrite(alpha_file, alpha_n,',');
exit

function [t]=backtrack_line_search(beta, alpha, alpha_change,deltaP,lambda1,lambda2)
		t=1;
		rate=0.5;
		while 1
			temp_alpha = alpha-t*alpha_change;
			fxt = -temp_alpha'*deltaP + lambda1*norm(temp_alpha,1)+lambda2*(norm(temp_alpha,2)^2-1); 
			fx= -alpha'*deltaP + lambda1*norm(alpha,1)+lambda2*(norm(alpha,2)^2-1);
			fx = fx - rate*t*alpha_change'*alpha_change;
			%fprintf('t=%.4f fxt=%.4f fx=%.4f\n',t,fxt,fx);
			if (fx-fxt)<=1*10^-4 || t<=1*10^-3
				break
			end
			t= beta*t;
		end
		%fprintf('final-t=%.4f\n',t);
return