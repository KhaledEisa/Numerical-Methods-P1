## Numerical Methods P1
Numerical methods app that implements the bisection, secant, newton raphson, lagrange, basic mid-point, basic trapezoidal rule, and basic simpson's rule using python

##Project Description & Steps:
# Bisection
For the bisection we get a function ex: x^3 + 4x^2 - 10 = 0, and then we get the x interval x {1, 2}, and the error = 10^(-2). The process should be as follows we calculate the Pi which is (ai + bi)/2 then we subsitute in f(x) and check if the value is below 0 then the value of ai is pi and bi stays the same. In case the f(x) value is greater than 0 the value of bi is pi and ai stays the same the error should be calculated in each iteration in 0.00054 not 1.6e-6 etc.


For the secant we get 2 points x0 and x1, and f(x) = x^3 + 4x^2 - 10, and the error or we get the number of iterations to do without error. then we calculate the new x which is ((X(i-1)*f(xi))- (x(i) * f(xi-1))) / (f(xi)-f(xi-1)). The table contains i, xi, f(xi), and the comparison |2-1| <! error



For the newton raphson we get the f(x) function and x0 and the error. The process is to calculate the first and second derivative to f(x) and substitute with the x0 and print them to the user as f(x) = , f'(x) = , f''(x) =. then we print checking the convergence and which is |f(x0)*f''(x0)| < f'(x0)^2. If the comparison is true then we continue to calculate the new x's which use the rool
xi+1 = xi - f(xi)/f'(xi). Then we show the table which contains i, xi, |xi+1 - xi| to check against the error. the user can input an interval so ask him prior if he'll use interval or a single point. If he uses and interival take the interval from, to. and choose the midpoint in the interval to solve the newton raphson.

For lagrange interpolation the user inputs either a table x0, f0, x1, f1 etc. but you must ask the user prior if he'll enter a table or function. and if he says table take from him the counter of the elements of the table count to be inserted and take from him the x's and f(x)'s. if he chooses a function take the function f(x) and the values of x to be used and they can be a range or multiple random value so ask him prior as well. after taking the inputs you must print the rule which is to caluclate P(x) = summation(Lk(x)fk(x)) where it starts from 0 to n which is the count of data. Then print the L functions which has a rule which is. As an example if we have 3 points x={0, 1, 2} and f(x)={3, 2, 7} for L0(x0) = (((x-1)*(x-2))/ ((0-1)*(0-2))) and print out the result of the multiplication which is another function f(x). Do for each point the L function calculation and print them. after having the L functions multiply each L function to it's corresponding f(x) value and show the function before multiplication and after then the user migh ask you to calculate the P of a value or P' of a value or both so ask him after calculating the functions and getting the P(x) if he wants to substitute or not. The derivative should be caluclated in the process.

Then we have 3 basic rules to implement which are: Basic Mid-Point, Basic Trapezoidal Rule, Basic Simpson's Rule. The rule for each of them and the inputs are as follows:
The user inputs the integration f(x) "before integration because it'll be calculated within the process" and he'll provide the a and b values which the integration will be made upon.
For the basic mid-point: (b-a)*f((a+b)/2)
For the basic trapezoidal rule ((b-a)/2)*(f(a)+f(b)))
For the basic Simpson's rule ((b-a)/6)*(f(a)+(4*f((a+b)/2))+f(b)) 
Show the functions and then calulcate the results and show them.
The user can choose to check which of the 3 functions work so the user might enter the integration and the value he is looking for. The code will run the 3 functions. If all of the functions solve it and get the same result the user inputted he gets a message that says any method would work on such function. But if only one works then we output which function it was, etc.



Allow the code to input any function f(x) whether it has sin, cos, tan, or e^-2, or a combination xe^-2 etc.

And tell me which language is preffered in such project
