function netbp
%NETBP  Uses backpropagation to train a network 

%%%%%%% DATA %%%%%%%%%%%
x1 = [0.1,0.3,0.1,0.2,0.4,0.6,0.5,0.9,0.4,0.7,0.3,0.4,0.2,0.1,0.5];
x2 = [0.1,0.4,0.5,0.2,0.2,0.3,0.6,0.2,0.4,0.6,0.9,0.7,0.6,0.8,0.8];
y = [ones(1,5) zeros(1,10); zeros(1,5) ones(1,5) zeros(1,5);ones(1,5) zeros(1,5) zeros(1,5)];

figure(1)
hold on
xlim([0 1])
ylim([0 1])
scatter(x1(1:5),x2(1:5),'red','x', 'LineWidth', 3)
scatter(x1(6:10),x2(6:10),'blue','o','LineWidth', 3)
scatter(x1(11:15),x2(11:15),'yellow','v', 'LineWidth', 3)
hold off

%Initialize weights and biases 
rng(5000);
W2 = 0.5*randn(2,2); W3 = 0.5*randn(3,2); W4 = 0.5*randn(3,3);
b2 = 0.5*randn(2,1); b3 = 0.5*randn(3,1); b4 = 0.5*randn(3,1);

% Forward and Back propagate 
eta = 0.05;                % learning rate
Niter = 1e6;               % number of SG iterations 
savecost = zeros(Niter,1); % value of cost function at each iteration
for counter = 1:Niter
    k = randi(10);         % choose a training point at random
    x = [x1(k); x2(k)];
    % Forward pass
    a2 = activate(x,W2,b2);
    a3 = activate(a2,W3,b3);
    a4 = activate(a3,W4,b4);
    % Backward pass
    delta4 = a4.*(1-a4).*(a4-y(:,k));
    delta3 = a3.*(1-a3).*(W4'*delta4);
    delta2 = a2.*(1-a2).*(W3'*delta3);
    % Gradient step
    W2 = W2 - eta*delta2*x';
    W3 = W3 - eta*delta3*a2';
    W4 = W4 - eta*delta4*a3';
    b2 = b2 - eta*delta2;
    b3 = b3 - eta*delta3;
    b4 = b4 - eta*delta4;
    % Monitor progress
    newcost = cost(W2,W3,W4,b2,b3,b4);   % display cost to screen
    savecost(counter) = newcost;
end


% Show decay of cost function  
save costvec
figure(2)
semilogy(1:1e4:Niter,savecost(1:1e4:Niter))

%We shadow the different sections of the plot

  function costval = cost(W2,W3,W4,b2,b3,b4)
     costvec = zeros(10,1); 
     for i = 1:10
         x =[x1(i);x2(i)];
         a2 = activate(x,W2,b2);
         a3 = activate(a2,W3,b3);
         a4 = activate(a3,W4,b4);
         costvec(i) = norm(y(:,i) - a4,2);
     end
     costval = norm(costvec,2)^2;
  end % of nested function

end
