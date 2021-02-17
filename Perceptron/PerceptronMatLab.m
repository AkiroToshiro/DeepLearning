net = newp([-9 0; 0 9], 3);


P1 = {[0; 0] [0; 1] [1; 0] [1; 1]};
Y = sim(net,P1)
% [and; or; no and]
T1 = {[0; 0; 1] [0; 1; 0] [0; 1; 0] [1; 1; 0]};
net.adaptParam.passes = 20;
net = adapt(net,P1,T1);
Y = sim(net,P1)

P2 = [[0; 0] [0; 1] [1; 0] [1; 1]];
T2 = [[0; 0; 1] [0; 1; 0] [0; 1; 0] [1;1;0]];

net = init(net);
Y = sim(net,P2)
net.trainParam.epochs = 20;
net = train(net,P2,T2);
Y = sim(net,P2)

wts = net.IW{1,1}, bias = net.b{1, 1, 1}

net.inputweights{1,1}.initFcn = 'rands';
net.biases{1}.initFcn = 'rands';
wts = net.IW{1,1}, bias = net.b{1, 1, 1}

net = init(net);

net = train(net,P2,T2);
Y = sim(net,P2)

