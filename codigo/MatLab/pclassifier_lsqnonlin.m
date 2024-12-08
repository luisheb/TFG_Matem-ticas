function pclassifier_lsqnonlin

%%%%%%% DATA %%%%%%%%%%%
x1 = [0.1,0.3,0.1,0.2,0.4,0.6,0.5,0.9,0.4,0.7,0.3,0.4,0.2,0.1,0.5];
x2 = [0.1,0.4,0.5,0.2,0.2,0.3,0.6,0.2,0.4,0.6,0.9,0.7,0.6,0.8,0.8];
y = [ones(1,5) zeros(1,10); zeros(1,5) ones(1,5) zeros(1,5); zeros(1,10) ones(1,5)];

figure(1)
clf
a1 = subplot(1,1,1);
plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
hold on
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
plot(x1(11:15),x2(11:15),'yv','MarkerSize',12,'LineWidth',4)
a1.XTick = [0 1];
a1.YTick = [0 1];
a1.FontWeight = 'Bold';
a1.FontSize = 10;
xlim([0,1])
ylim([0,1])


rng(5000);
Pzero = 0.5*randn(27,1);

[finalP,finalerr] = lsqnonlin(@neterr,Pzero);

% Check this function 
finalW2 = zeros(2,2);
finalW3 = zeros(3,2);
finalW4 = zeros(3,3);
finalW2(:) = finalP(1:4);
finalW3(:) = finalP(5:10);
finalW4(:) = finalP(11:19);
finalb2 = finalP(20:21);
finalb3 = finalP(22:24);
finalb4 = finalP(25:27);

N = 500;
Dx = 1/N;
Dy = 1/N;
xvals = 0:Dx:1;
yvals = 0:Dy:1;
for k1 = 1:N+1
    xk = xvals(k1);
    for k2 = 1:N+1
        yk = yvals(k2);
        xy = [xk;yk];
        a2 = activate(xy,finalW2,finalb2);
        a3 = activate(a2,finalW3,finalb3);
        a4 = activate(a3,finalW4,finalb4);
        Aval(k2,k1) = a4(1);
        Bval(k2,k1) = a4(2);
        Cval(k2,k1) = a4(3);
     end
end
[X,Y] = meshgrid(xvals,yvals);

figure(3)
clf
a2 = subplot(1,1,1);

Mval = ones(size(Aval));

Mval(Cval >= Aval & Cval >= Bval) = 2;
Mval(Aval >= Bval & Aval >= Cval) = 1;
Mval(Bval >= Aval & Bval >= Cval) = 0;

contourf(X, Y, Mval, [-1 0.75 1.5 2.5]); 

hold on

colormap([1 1 1; 0.8 0.8 0.8; 0.5 0.5 0.5]);

plot(x1(1:5),x2(1:5),'ro','MarkerSize',12,'LineWidth',4)
plot(x1(6:10),x2(6:10),'bx','MarkerSize',12,'LineWidth',4)
plot(x1(11:15),x2(11:15),'yv','MarkerSize',12,'LineWidth',4)
a2.XTick = [0 1];
a2.YTick = [0 1];
a2.FontWeight = 'Bold';
a2.FontSize = 10;
xlim([0,1])
ylim([0,1])

print -dpng classifier_lsq.png

  function costvec = neterr(Pval)
  % return a vector whose two-norm squared is the cost function  
     W2 = zeros(2,2);
     W3 = zeros(3,2);
     W4 = zeros(3,3);
     W2(:) = Pval(1:4);
     W3(:) = Pval(5:10);
     W4(:) = Pval(11:19);
     b2 = Pval(20:21);
     b3 = Pval(22:24);
     b4 = Pval(25:27);

     costvec = zeros(15,1); 
     for i = 1:15
         x = [x1(i);x2(i)];
         a2 = activate(x,W2,b2);
         a3 = activate(a2,W3,b3);
         a4 = activate(a3,W4,b4);
         costvec(i) = norm(y(:,i) - a4,2);
     end
  end % of nested function

end