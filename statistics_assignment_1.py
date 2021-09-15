def binom_coefficient(n, k):
    binom_coef = 1.0
    for l in range(k):
        tmp_numer = n - l
        tmp_denom = k - l
        binom_coef *= tmp_numer / tmp_denom
    return binom_coef

def count_even(n):
    # Use recursion. #
    c_even = 0.0
    n_half = int(n / 2)
    for l in range(n_half+1):
        c_even += binom_coefficient(n, 2*l)
    return c_even

def tot_count_subsets(m):
    c_subsets = sum(
        [2**l for l in range(1, m+1)])
    return c_subsets

def tot_count_even(m):
    eta_even = sum([
        count_even(l) for l in range(1, m+1)])
    return eta_even

def probability_even(m):
    # Include the empty set. #
    tmp_numer = tot_count_even(m) + 1
    tmp_denom = tot_count_subsets(m) + 1
    p_even = tmp_numer / tmp_denom
    return p_even

def probability_even_0_1(m):
    # Include the empty set. #
    tmp_numer = 2**m
    tmp_denom = 2**(m+1) - 1
    p_even = tmp_numer / tmp_denom
    return p_even

def probability_even_0_1_2(m):
    # Include the empty set. #
    tmp_numer = 3**(m+1) + 2*m + 1
    tmp_denom = 2 * (3**(m+1) - 1)
    p_even = tmp_numer / tmp_denom
    return p_even

n_maximum  = 100
p_sub_even = [
    probability_even_0_1(l) for l in range(n_maximum)]

p_sub_even_2 = [
    probability_even_0_1_2(l) for l in range(n_maximum)]
