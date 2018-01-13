function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

h=sigmoid(X*theta);
% You need to return the following variables correctly 
%J = 0;
%grad = zeros(size(theta));
theta(1)=0;
th_sq=theta'*theta;
cr_term=(lambda/(2*m))*th_sq;
J = (1/m)*((-y'*log(h))-((1-y)'*log(1-h)))+cr_term;

e=h-y;
gr_term=(lambda/m)*theta;
grad=1/m.*(X'*e)+gr_term;


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
