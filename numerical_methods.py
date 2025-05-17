import sympy
import math 

def get_float_input(prompt, allow_empty=False):
    while True:
        try:
            val_str = input(prompt).strip()
            if allow_empty and not val_str:
                return None
            return float(val_str)
        except ValueError:
            print("Invalid input. Please enter a valid number.")

def get_int_input(prompt, min_val=None, allow_empty=False):
    while True:
        try:
            val_str = input(prompt).strip()
            if allow_empty and not val_str:
                return None
            val = int(val_str)
            if min_val is not None and val < min_val:
                print(f"Value must be at least {min_val}.")
                continue
            return val
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

def get_function_from_string(prompt="Enter the function f(x) (e.g., x**3 + 4*x**2 - 10): "):
    """
    Parses a function string using sympy and returns:
    - symbolic_expr: The sympy symbolic expression.
    - callable_func: A Python function that can be evaluated.
    - x_sym: The sympy symbol for 'x'.
    Returns (None, None, None) on failure.
    """
    while True:
        func_str = input(prompt).strip()
        if not func_str:
            print("Function string cannot be empty.")
            continue
        try:
            x_sym = sympy.symbols('x')
            # Define common functions for sympy parsing
            # For security in a real app, you might want a more restricted namespace
            local_dict = {
                "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
                "exp": sympy.exp, "ln": sympy.log, "log": sympy.log, # ln and log are common for natural log
                "log10": lambda arg: sympy.log(arg, 10),
                "sqrt": sympy.sqrt,
                "pi": sympy.pi, "e": sympy.E,
                "x": x_sym # ensure x is available
            }
            # Sympy's parse_expr is safer than eval for mathematical expressions
            symbolic_expr = sympy.parse_expr(func_str, local_dict=local_dict, transformations='all')

            # Check if 'x' is the only free symbol, or handle other cases if needed
            if x_sym not in symbolic_expr.free_symbols and symbolic_expr.free_symbols:
                print(f"Warning: Expression does not seem to depend on 'x'. Free symbols: {symbolic_expr.free_symbols}")
            
            # Create a callable function
            # Using 'evalf=True' for lambdify can sometimes help with numerical stability for complex exprs
            # but for pure symbolic, it's often better to lambdify then call .evalf() on results if needed.
            callable_func = sympy.lambdify(x_sym, symbolic_expr, modules=['math', 'sympy'])
            
            # Test the function
            try:
                callable_func(1.0) 
            except Exception as e:
                print(f"Error during initial test of parsed function: {e}")
                print("Please ensure your function is valid and uses 'x' as the variable.")
                # Potentially loop again or ask for re-entry
                continue # Or return None, None, None if you want to exit the loop on error

            return symbolic_expr, callable_func, x_sym
        except (sympy.SympifyError, SyntaxError, TypeError, NameError) as e:
            print(f"Error parsing function string '{func_str}': {e}")
            print("Please use Python/Sympy compatible math syntax (e.g., x**2 for x^2, exp(x) for e^x, log(x) for ln(x)).")
        except Exception as e: # Catch any other unexpected errors during parsing
            print(f"An unexpected error occurred while parsing '{func_str}': {e}")

def print_table(headers, data_rows, column_widths=None):
    """Prints a formatted table."""
    if not headers or not data_rows:
        return

    if column_widths is None:
        # Auto-calculate widths if not provided
        column_widths = [len(h) for h in headers]
        for row in data_rows:
            for i, cell in enumerate(row):
                column_widths[i] = max(column_widths[i], len(str(cell)))
    
    header_line = " | ".join(f"{h:<{column_widths[i]}}" for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * column_widths[i] for i in range(len(headers)))
    
    print("\n" + header_line)
    print(separator_line)
    
    for row in data_rows:
        row_line = " | ".join(f"{str(cell):<{column_widths[i]}}" for i, cell in enumerate(row))
        print(row_line)
    print("-" * len(separator_line)) # Footer line

# --- Numerical Method Implementations ---
# 1. Bisection Method (Corrected Error)
def bisection_solver():
    print("\n--- Bisection Method ---")
    symbolic_f, f, x_sym = get_function_from_string()
    if not symbolic_f: return

    a = get_float_input("Enter the start of the interval (a): ")
    b = get_float_input("Enter the end of the interval (b): ")
    error_tol_str = input("Enter the error tolerance (e.g., 0.01 or 10^-2 for |b_i - a_i| < tol): ").strip()
    try:
        error_tol = float(sympy.sympify(error_tol_str).evalf())
    except (sympy.SympifyError, TypeError):
        print("Invalid error tolerance format.")
        return

    # Initial check for signs
    f_a_initial = f(a)
    f_b_initial = f(b)
    if f_a_initial * f_b_initial >= 0:
        print("Bisection method fails: f(a) and f(b) must have opposite signs.")
        print(f"f({a}) = {f_a_initial:.6f}, f({b}) = {f_b_initial:.6f}")
        return

    iterations_data = []
    headers = ["Iter", "a_i", "b_i", "p_i", "f(p_i)", "|b_i - a_i|"] # Changed Error header
    
    ai, bi = a, b
    # f_ai will store f(ai) to avoid re-computation if ai doesn't change
    f_ai = f_a_initial 
    
    pi = (ai + bi) / 2 # Declare pi outside loop for final print if max_iter reached

    for i in range(1, 101): # Max 100 iterations safeguard
        current_interval_width = bi - ai # This is the error as per your request

        # Calculate p_i only once per iteration
        pi = (ai + bi) / 2
        f_pi = f(pi)

        iterations_data.append([
            i, 
            f"{ai:.6f}", 
            f"{bi:.6f}", 
            f"{pi:.6f}", 
            f"{f_pi:.6f}",
            f"{abs(current_interval_width):.{max(1, int(-math.log10(error_tol)) + 1 if error_tol > 0 else 6)}f}" # abs for safety, format based on tol
        ])

        # Stopping condition: width of interval containing the root
        if abs(current_interval_width) < error_tol or f_pi == 0:
            print_table(headers, iterations_data)
            print(f"\nConverged after {i} iterations.")
            # The root is known to be within [ai, bi], p_i is the midpoint.
            # The best estimate for the root is p_i.
            # The maximum error of p_i from the true root is current_interval_width / 2.
            print(f"Approximate root p ≈ {pi:.{int(-math.log10(error_tol)) + 2 if error_tol > 0 else 6}f}")
            print(f"The root lies in the interval [{ai:.6f}, {bi:.6f}] of width {abs(current_interval_width):.6f}")
            print(f"f(p) ≈ {f(pi):.6e}")
            return

        # Update interval
        if f_ai * f_pi < 0: # Root is in [ai, pi]
            bi = pi
            # f_ai remains the same
        else: # Root is in [pi, bi]
            ai = pi
            f_ai = f_pi # Update f_ai as ai has changed to pi
        
    # If loop finishes, max iterations reached
    print_table(headers, iterations_data)
    print("\nMaximum iterations reached without desired tolerance.")
    print(f"Last interval [{ai:.6f}, {bi:.6f}]")
    print(f"Last midpoint p ≈ {pi:.6f}, f(p) ≈ {f(pi):.6e}")
    print(f"Current interval width |b_i - a_i| = {abs(bi - ai):.6f}")

# 2. Secant Method (Corrected Error Formatting and UnboundLocalError fix)
def secant_solver():
    print("\n--- Secant Method ---")
    symbolic_f, f, x_sym = get_function_from_string()
    if not symbolic_f: return

    x_initial_0 = get_float_input("Enter initial point x0: ")
    x_initial_1 = get_float_input("Enter initial point x1: ")

    error_tol_str = input("Enter error tolerance (e.g., 0.001) OR press Enter for number of iterations: ").strip()
    num_iterations_to_perform = None
    error_tol = None # Initialize to None

    if error_tol_str:
        try:
            error_tol = float(sympy.sympify(error_tol_str).evalf())
            if error_tol <= 0:
                print("Error tolerance must be a positive value.")
                return
        except (sympy.SympifyError, TypeError, ValueError): # Added ValueError for float conversion
            print("Invalid error tolerance format.")
            return
    else:
        num_iterations_to_perform = get_int_input("Enter number of *new points to calculate* (iterations): ", min_val=1)

    iterations_data = []
    headers = ["Iter (calc x_k+1)", "x_k-1", "x_k", "f(x_k-1)", "f(x_k)", "x_k+1", "Error |x_k+1 - x_k|"]
    
    xk_minus_1 = x_initial_0
    xk = x_initial_1
    
    max_loops = num_iterations_to_perform if num_iterations_to_perform else 100

    for i in range(1, max_loops + 1):
        try:
            f_xk = f(xk)
            f_xk_minus_1 = f(xk_minus_1)
        except Exception as e:
            print(f"Error evaluating function: {e}")
            if iterations_data:
                print_table(headers, iterations_data)
            return

        if abs(f_xk - f_xk_minus_1) < 1e-12:
            if iterations_data:
                 print_table(headers, iterations_data)
            print("\nDenominator f(xk) - f(xk-1) is too small. Method may fail or has converged.")
            print(f"Last xk = {xk:.6f}, f(xk) = {f_xk:.6e}")
            return

        try:
            xk_plus_1 = (xk_minus_1 * f_xk - xk * f_xk_minus_1) / (f_xk - f_xk_minus_1)
        except ZeroDivisionError:
            if iterations_data:
                print_table(headers, iterations_data)
            print("\nDivision by zero in Secant formula. f(xk) and f(xk-1) might be equal.")
            return
            
        current_error = abs(xk_plus_1 - xk)

        # Determine formatting for error string
        error_str_format = ""
        if error_tol and error_tol > 0:
            # Calculate number of decimal places based on tolerance magnitude
            # Add 1 or 2 for a bit more precision than the tolerance itself
            decimal_places_for_error = max(1, int(-math.log10(error_tol)) + 1) 
            error_str_format = f"{current_error:.{decimal_places_for_error}f}"
        else: # Fixed iterations or invalid tolerance
            error_str_format = f"{current_error:.6f}" # Default to 6 decimal places

        iterations_data.append([
            i,
            f"{xk_minus_1:.6f}",
            f"{xk:.6f}",
            f"{f_xk_minus_1:.6f}",
            f"{f_xk:.6f}",
            f"{xk_plus_1:.6f}",
            error_str_format # Use the pre-formatted string
        ])
        
        if error_tol is not None and current_error < error_tol:
            print_table(headers, iterations_data)
            decimal_places_for_root = max(1, int(-math.log10(error_tol)) + 2) if error_tol > 0 else 6
            print(f"\nConverged after {i} iteration(s) calculating new points (tolerance met).")
            print(f"Approximate root x = {xk_plus_1:.{decimal_places_for_root}f}")
            print(f"f(x) = {f(xk_plus_1):.6e}")
            return
        
        xk_minus_1 = xk
        xk = xk_plus_1
        
        if num_iterations_to_perform and i == num_iterations_to_perform:
            print_table(headers, iterations_data)
            print(f"\nCompleted {num_iterations_to_perform} iteration(s) calculating new points.")
            print(f"Approximate root x = {xk:.6f}")
            print(f"f(x) = {f(xk):.6e}")
            return

    if error_tol:
        print_table(headers, iterations_data)
        print("\nMaximum iterations (100) reached in tolerance mode without convergence.")
        print(f"Last calculated x = {xk:.6f}, f(x) = {f(xk):.6e}")

# 3. Newton-Raphson Method
def newton_raphson_solver():
    print("\n--- Newton-Raphson Method ---")
    symbolic_f, f, x_sym = get_function_from_string()
    if not symbolic_f: return

    # Calculate derivatives
    try:
        symbolic_f_prime = sympy.diff(symbolic_f, x_sym)
        f_prime = sympy.lambdify(x_sym, symbolic_f_prime, modules=['math', 'sympy'])
        symbolic_f_double_prime = sympy.diff(symbolic_f_prime, x_sym)
        f_double_prime = sympy.lambdify(x_sym, symbolic_f_double_prime, modules=['math', 'sympy'])
    except Exception as e:
        print(f"Error calculating derivatives: {e}")
        return

    print(f"f(x) = {symbolic_f}")
    print(f"f'(x) = {symbolic_f_prime}")
    print(f"f''(x) = {symbolic_f_double_prime}")

    x0_input_type = input("Use a single point x0 (p) or an interval (i)? [p/i]: ").strip().lower()
    x0 = 0.0
    if x0_input_type == 'i':
        a = get_float_input("Enter start of interval (a): ")
        b = get_float_input("Enter end of interval (b): ")
        x0 = (a + b) / 2
        print(f"Using midpoint x0 = {x0:.6f} from interval [{a}, {b}]")
    elif x0_input_type == 'p':
        x0 = get_float_input("Enter initial guess x0: ")
    else:
        print("Invalid choice for x0 input.")
        return

    error_tol_str = input("Enter the error tolerance (e.g., 0.001): ").strip()
    try:
        error_tol = float(sympy.sympify(error_tol_str).evalf())
    except (sympy.SympifyError, TypeError):
        print("Invalid error tolerance format.")
        return

    # Evaluate at x0
    try:
        val_f_x0 = f(x0)
        val_f_prime_x0 = f_prime(x0)
        val_f_double_prime_x0 = f_double_prime(x0)
    except Exception as e:
        print(f"Error evaluating functions at x0 = {x0}: {e}")
        return

    print(f"\nAt x0 = {x0:.6f}:")
    print(f"  f(x0) = {val_f_x0:.6f}")
    print(f"  f'(x0) = {val_f_prime_x0:.6f}")
    print(f"  f''(x0) = {val_f_double_prime_x0:.6f}")

    # Convergence Check
    print("\nChecking convergence condition |f(x0)*f''(x0)| < f'(x0)^2:")
    try:
        lhs_conv = abs(val_f_x0 * val_f_double_prime_x0)
        rhs_conv = val_f_prime_x0**2
        print(f"  |{val_f_x0:.3e} * {val_f_double_prime_x0:.3e}| = {lhs_conv:.3e}")
        print(f"  ({val_f_prime_x0:.3e})^2 = {rhs_conv:.3e}")
        if lhs_conv < rhs_conv:
            print("  Condition MET: {lhs_conv:.3e} < {rhs_conv:.3e}. Proceeding with iterations.")
        else:
            print("  Condition NOT MET: {lhs_conv:.3e} >= {rhs_conv:.3e}. Convergence is not guaranteed by this criterion.")
            # Allow user to proceed or stop
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                return
    except Exception as e: # Catch potential errors if f_prime_x0 is 0 etc.
        print(f"  Error during convergence check: {e}")
        print("  Cannot reliably check convergence. Proceeding with caution.")


    iterations_data = []
    headers = ["Iter", "x_i", "|x_{i+1} - x_i|"]
    xi = x0

    for i in range(1, 101): # Max 100 iterations
        f_xi = f(xi)
        f_prime_xi = f_prime(xi)

        if abs(f_prime_xi) < 1e-12: # Avoid division by zero
            print_table(headers, iterations_data)
            print(f"\nDerivative f'(xi) is close to zero at xi = {xi:.6f}. Method fails.")
            return

        xi_plus_1 = xi - f_xi / f_prime_xi
        current_error = abs(xi_plus_1 - xi)

        iterations_data.append([
            i,
            f"{xi:.6f}",
            f"{current_error:.6f}"
        ])

        if current_error < error_tol:
            print_table(headers, iterations_data)
            print(f"\nConverged after {i} iterations.")
            print(f"Approximate root x = {xi_plus_1:.{int(-math.log10(error_tol)) + 2}f}")
            print(f"f(x) = {f(xi_plus_1):.6e}")
            return
        
        xi = xi_plus_1
    
    print_table(headers, iterations_data)
    print("\nMaximum iterations reached without desired tolerance.")
    print(f"Last approximate root x = {xi:.6f}")
    print(f"f(x) = {f(xi):.6e}")


# 4. Lagrange Interpolation
def lagrange_solver():
    print("\n--- Lagrange Interpolation ---")
    x_sym = sympy.symbols('x') # For symbolic polynomial output

    input_mode = input("Enter data as a table (t) or from a function (f)? [t/f]: ").strip().lower()
    
    x_points_num = [] # Numerical values for x
    y_points_num = [] # Numerical values for y (f(x))
    f_k_sym = [] # Symbolic values for f_k if from table, or actual values

    if input_mode == 't':
        n_points = get_int_input("Enter the number of data points (n+1): ", min_val=2)
        print("Enter data points (x_k, f_k):")
        for k in range(n_points):
            xk = get_float_input(f"  x_{k}: ")
            fk_str = input(f"  f_{k} (can be numeric or symbolic expression if desired, usually numeric): ").strip()
            try:
                # Try to make it numeric first
                fk_num = float(fk_str)
                x_points_num.append(xk)
                y_points_num.append(fk_num)
                f_k_sym.append(sympy.sympify(fk_num)) # Store as sympy number
            except ValueError:
                # If not numeric, try to parse as symbolic (less common for f_k from table)
                try:
                    fk_s = sympy.parse_expr(fk_str)
                    x_points_num.append(xk)
                    # For y_points_num, we'd need a way to evaluate this symbolic fk if needed later
                    # This scenario is more complex; typically table input means numeric y_points.
                    # For now, let's assume numeric f_k for table input for simplicity of y_points_num.
                    # If you need symbolic f_k from table, y_points_num would need to be handled differently.
                    print("Warning: Symbolic f_k from table input is advanced. Assuming numeric for evaluation.")
                    return # Or handle this case more robustly
                except sympy.SympifyError:
                    print(f"Invalid input for f_{k}. Please enter a number.")
                    return


    elif input_mode == 'f':
        symbolic_f_expr, f_callable, _ = get_function_from_string("Enter the function f(x) for interpolation: ")
        if not symbolic_f_expr: return

        x_values_mode = input("Enter x-values as a range (r) or multiple individual values (m)? [r/m]: ").strip().lower()
        if x_values_mode == 'r':
            start_x = get_float_input("Enter start of x range: ")
            end_x = get_float_input("Enter end of x range: ")
            num_x = get_int_input("Enter number of points in range: ", min_val=2)
            if start_x >= end_x and num_x > 1 :
                print("Start of range must be less than end for multiple points.")
                return
            if num_x == 1:
                 x_points_num = [start_x]
            else:
                 x_points_num = [start_x + i * (end_x - start_x) / (num_x - 1) for i in range(num_x)]
        elif x_values_mode == 'm':
            num_x = get_int_input("Enter number of x-values: ", min_val=2)
            print("Enter x-values:")
            for k in range(num_x):
                x_points_num.append(get_float_input(f"  x_{k}: "))
        else:
            print("Invalid choice for x-values input.")
            return
        
        # Calculate y_points and f_k_sym from the function
        for x_val in x_points_num:
            try:
                y_val = f_callable(x_val)
                y_points_num.append(y_val)
                f_k_sym.append(symbolic_f_expr.subs(x_sym, x_val)) # Substitute to get the value or expression at x_k
            except Exception as e:
                print(f"Error evaluating f({x_val}): {e}")
                return
    else:
        print("Invalid input mode.")
        return

    if len(set(x_points_num)) != len(x_points_num): # Check for duplicate x values
        print("Error: x-values must be distinct for Lagrange interpolation.")
        return

    n = len(x_points_num) -1 # Degree of polynomial is n (n+1 points)
    
    print(f"\nInterpolating with {n+1} points (degree {n} polynomial).")
    print(f"Data points (x_k, f_k):")
    for k in range(n + 1):
        print(f"  x_{k} = {x_points_num[k]}, f_{k} = {y_points_num[k] if input_mode == 'f' or isinstance(y_points_num[k], (int, float)) else f_k_sym[k]}")


    print(f"\nP(x) = Σ [L_k(x) * f_k] (for k from 0 to {n})")

    L_k_symbolic_list = []
    P_x_symbolic = sympy.sympify(0)

    for k in range(n + 1):
        numerator_sym = sympy.sympify(1)
        denominator_num = 1.0 # Numerical denominator for L_k

        print(f"\n--- L_{k}(x) ---")
        L_k_formula_str_num = []
        L_k_formula_str_den = []

        for j in range(n + 1):
            if k == j:
                continue
            numerator_sym *= (x_sym - x_points_num[j])
            denominator_num *= (x_points_num[k] - x_points_num[j])
            L_k_formula_str_num.append(f"(x - {x_points_num[j]})")
            L_k_formula_str_den.append(f"({x_points_num[k]} - {x_points_num[j]})")
        
        L_k_sym = sympy.simplify(numerator_sym / denominator_num)
        L_k_symbolic_list.append(L_k_sym)

        print(f"  L_{k}(x) = [{'*'.join(L_k_formula_str_num)}] / [{'*'.join(L_k_formula_str_den)}]")
        print(f"  L_{k}(x) simplified = {L_k_sym}")
        
        term_sym = L_k_sym * f_k_sym[k] # f_k_sym[k] is either number or symbolic f(x_k)
        print(f"  L_{k}(x) * f_{k} = ({L_k_sym}) * ({f_k_sym[k]}) = {sympy.expand(term_sym)}")
        P_x_symbolic += term_sym

    P_x_symbolic = sympy.expand(P_x_symbolic) # Expand the final polynomial
    print(f"\n--- Resulting Interpolating Polynomial P(x) ---")
    print(f"P(x) = {P_x_symbolic}")

    # Lambdify P(x) for evaluation
    try:
        P_callable = sympy.lambdify(x_sym, P_x_symbolic, modules=['math', 'sympy'])
    except Exception as e:
        print(f"Could not create callable function for P(x): {e}")
        P_callable = None
    
    # Symbolic Derivative P'(x)
    try:
        P_prime_x_symbolic = sympy.diff(P_x_symbolic, x_sym)
        P_prime_callable = sympy.lambdify(x_sym, P_prime_x_symbolic, modules=['math', 'sympy'])
        print(f"\nP'(x) = {P_prime_x_symbolic}")
    except Exception as e:
        print(f"Could not calculate or lambdify P'(x): {e}")
        P_prime_x_symbolic = None
        P_prime_callable = None


    while True:
        eval_choice = input("\nEvaluate P(x) or P'(x) at a point? (p for P(x), d for P'(x), b for both, n for none): ").strip().lower()
        if eval_choice == 'n':
            break
        elif eval_choice in ['p', 'd', 'b']:
            val_to_eval = get_float_input("Enter the x-value for evaluation: ")
            if eval_choice == 'p' or eval_choice == 'b':
                if P_callable:
                    try:
                        result_p = P_callable(val_to_eval)
                        print(f"  P({val_to_eval}) = {result_p:.6f}")
                    except Exception as e:
                        print(f"  Error evaluating P({val_to_eval}): {e}")
                else:
                    print("  P(x) callable function not available.")
            
            if eval_choice == 'd' or eval_choice == 'b':
                if P_prime_callable:
                    try:
                        result_p_prime = P_prime_callable(val_to_eval)
                        print(f"  P'({val_to_eval}) = {result_p_prime:.6f}")
                    except Exception as e:
                        print(f"  Error evaluating P'({val_to_eval}): {e}")
                else:
                    print("  P'(x) callable function not available.")
            if eval_choice != 'b': # If not 'both', break after p or d
                 break
        else:
            print("Invalid choice.")


# 5. Numerical Integration Rules
def numerical_integration_solver():
    print("\n--- Numerical Integration (Basic Rules) ---")
    symbolic_f, f, x_sym = get_function_from_string("Enter the integrand f(x): ")
    if not symbolic_f: return

    a = get_float_input("Enter the lower limit of integration (a): ")
    b = get_float_input("Enter the upper limit of integration (b): ")

    if a == b:
        print("Integral from a to a is 0.")
        return
    if a > b:
        print("Warning: Lower limit a is greater than upper limit b. Result will be -(Integral from b to a).")
        # We can proceed, the formulas will handle sign, or swap and negate at the end.

    # Evaluate f(a), f(b), f((a+b)/2)
    try:
        f_a = f(a)
        f_b = f(b)
        f_mid = f((a + b) / 2)
    except Exception as e:
        print(f"Error evaluating function at limits or midpoint: {e}")
        return

    print(f"\nFor f(x) = {symbolic_f} from a={a} to b={b}:")
    print(f"  f(a) = f({a}) = {f_a:.6f}")
    print(f"  f(b) = f({b}) = {f_b:.6f}")
    print(f"  f((a+b)/2) = f({(a+b)/2:.4f}) = {f_mid:.6f}")


    # Basic Mid-Point Rule
    mid_point_val = (b - a) * f_mid
    print(f"\n1. Basic Mid-Point Rule:")
    print(f"   I ≈ (b-a) * f((a+b)/2) = ({b-a}) * f({(a+b)/2:.4f})")
    print(f"     = ({b-a}) * {f_mid:.6f} = {mid_point_val:.6f}")

    # Basic Trapezoidal Rule
    trapezoidal_val = ((b - a) / 2) * (f_a + f_b)
    print(f"\n2. Basic Trapezoidal Rule:")
    print(f"   I ≈ ((b-a)/2) * (f(a) + f(b)) = ({(b-a)/2}) * ({f_a:.6f} + {f_b:.6f})")
    print(f"     = ({(b-a)/2}) * {(f_a + f_b):.6f} = {trapezoidal_val:.6f}")

    # Basic Simpson's Rule
    simpson_val = ((b - a) / 6) * (f_a + 4 * f_mid + f_b)
    print(f"\n3. Basic Simpson's Rule:")
    print(f"   I ≈ ((b-a)/6) * (f(a) + 4*f((a+b)/2) + f(b))")
    print(f"     = ({(b-a)/6:.6f}) * ({f_a:.6f} + 4*{f_mid:.6f} + {f_b:.6f})")
    print(f"     = ({(b-a)/6:.6f}) * {(f_a + 4*f_mid + f_b):.6f} = {simpson_val:.6f}")

    # Comparison feature
    compare_choice = input("\nDo you want to compare these results against an expected value? (y/n): ").strip().lower()
    if compare_choice == 'y':
        expected_val_str = input("Enter the expected integral value: ")
        try:
            expected_val = float(sympy.sympify(expected_val_str).evalf())
            print(f"\nComparing with expected value: {expected_val:.6f}")
            # Use a small tolerance for floating point comparison
            comp_tol = 1e-5 
            results_match = {
                "Mid-Point": abs(mid_point_val - expected_val) < comp_tol,
                "Trapezoidal": abs(trapezoidal_val - expected_val) < comp_tol,
                "Simpson's": abs(simpson_val - expected_val) < comp_tol,
            }
            matching_methods = [name for name, matches in results_match.items() if matches]

            if not matching_methods:
                print("None of the basic methods matched the expected value within tolerance.")
            elif len(matching_methods) == 3:
                print("All three basic methods (Mid-Point, Trapezoidal, Simpson's) matched the expected value.")
            else:
                print(f"The following method(s) matched the expected value: {', '.join(matching_methods)}")
        except (sympy.SympifyError, TypeError, ValueError):
            print("Invalid format for expected value.")


# --- Main Menu ---
def main_menu():
    while True:
        print("\n========= Numerical Methods Solver =========")
        print("Choose a method:")
        print("  1. Bisection Method")
        print("  2. Secant Method")
        print("  3. Newton-Raphson Method")
        print("  4. Lagrange Interpolation")
        print("  5. Numerical Integration (Basic Rules)")
        # Add Simple Iteration if desired - requires careful g(x) formulation by user
        print("  q. Quit")

        choice = input("Enter your choice: ").strip().lower()

        if choice == '1':
            bisection_solver()
        elif choice == '2':
            secant_solver()
        elif choice == '3':
            newton_raphson_solver()
        elif choice == '4':
            lagrange_solver()
        elif choice == '5':
            numerical_integration_solver()
        elif choice == 'q':
            print("Exiting solver. Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_menu()